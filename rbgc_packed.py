
from __future__ import annotations
import math
import torch
from torch import nn
from torch.utils.cpp_extension import load_inline
_XNOR_EXT = None

def load_xnor_ext(verbose: bool = False):
    """Compile/load the packed uint32 XNOR-popcount CUDA extension."""
    global _XNOR_EXT
    if _XNOR_EXT is not None:
        return _XNOR_EXT

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the packed XNOR-popcount layer.")

    cpp_src = r"""
#include <torch/extension.h>

torch::Tensor pack_weight(torch::Tensor weight);
torch::Tensor xnor_forward(torch::Tensor x_u32, torch::Tensor w_pack, int64_t D);
torch::Tensor unpack_selected_pm1(torch::Tensor x_u32, torch::Tensor bit_index);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_weight", &pack_weight, "Pack sign(weight) into uint32 words");
    m.def("xnor_forward", &xnor_forward, "Packed uint32 XNOR-popcount forward");
    m.def("unpack_selected_pm1", &unpack_selected_pm1,
          "Unpack selected packed bits directly to float32 +/-1");
}
"""

    cuda_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be float32")
#define CHECK_INT(x) TORCH_CHECK(x.scalar_type() == torch::kInt32, #x " must be int32")

// The bit positions come from arange(D) or randperm(D), so they are in bounds.
// Each thread writes one contiguous output element. All bit manipulation stays
// in registers: no [N,G] int64 words or float32 0/1 intermediates are allocated.
__global__ void unpack_selected_pm1_kernel(
    const int* __restrict__ X,
    const int64_t* __restrict__ bit_index,
    float* __restrict__ Y,
    int64_t N,
    int64_t W32,
    int64_t G
) {
    const int64_t count = N * G;
    for (int64_t pos = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         pos < count;
         pos += (int64_t)blockDim.x * gridDim.x) {
        const int64_t row = pos / G;
        const int64_t j = bit_index[pos - row * G];
        const unsigned int word = (unsigned int)X[row * W32 + (j >> 5)];
        const unsigned int bit = (word >> (unsigned int)(j & 31)) & 1u;
        Y[pos] = bit ? 1.0f : -1.0f;
    }
}

torch::Tensor unpack_selected_pm1(torch::Tensor x_u32, torch::Tensor bit_index) {
    CHECK_CUDA(x_u32);
    CHECK_CONTIGUOUS(x_u32);
    CHECK_INT(x_u32);
    CHECK_CUDA(bit_index);
    CHECK_CONTIGUOUS(bit_index);
    TORCH_CHECK(bit_index.scalar_type() == torch::kInt64, "bit_index must be int64");
    TORCH_CHECK(x_u32.dim() == 2, "x_u32 must be [N,W32]");
    TORCH_CHECK(bit_index.dim() == 1, "bit_index must be [G]");
    TORCH_CHECK(x_u32.device() == bit_index.device(), "input devices must match");
    const c10::cuda::CUDAGuard device_guard(x_u32.device());
    const int64_t N = x_u32.size(0);
    const int64_t W32 = x_u32.size(1);
    const int64_t G = bit_index.numel();
    TORCH_CHECK(W32 > 0, "x_u32 must have at least one packed word");
    auto out = torch::empty({N, G}, x_u32.options().dtype(torch::kFloat32));
    if (N == 0 || G == 0) return out;
    const int threads = 256;
    int64_t blocks64 = (N * G + threads - 1) / threads;
    const int blocks = (int)(blocks64 < 65535 ? blocks64 : 65535);
    unpack_selected_pm1_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x_u32.data_ptr<int>(), bit_index.data_ptr<int64_t>(), out.data_ptr<float>(),
        N, W32, G);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

__global__ void pack_weight_kernel(
    const float* __restrict__ W,
    int* __restrict__ Wp,
    int D,
    int H,
    int W32
) {
    int word = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    if (word >= W32 || h >= H) return;

    unsigned int out = 0u;
    int base = word * 32;
    #pragma unroll
    for (int b = 0; b < 32; ++b) {
        int d = base + b;
        if (d < D && W[d * H + h] > 0.0f) {
            out |= (1u << b);
        }
    }
    Wp[word * H + h] = (int)out;
}

torch::Tensor pack_weight(torch::Tensor weight) {
    CHECK_CUDA(weight);
    CHECK_CONTIGUOUS(weight);
    CHECK_FLOAT(weight);
    TORCH_CHECK(weight.dim() == 2, "weight must be [D,H]");

    int D = (int)weight.size(0);
    int H = (int)weight.size(1);
    int W32 = (D + 31) / 32;

    auto out = torch::empty(
        {W32, H},
        torch::TensorOptions().dtype(torch::kInt32).device(weight.device())
    );

    dim3 block(16, 16);
    dim3 grid((W32 + block.x - 1) / block.x,
              (H + block.y - 1) / block.y);
    pack_weight_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        weight.data_ptr<float>(),
        out.data_ptr<int>(),
        D, H, W32
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

__global__ void xnor_forward_kernel(
    const int* __restrict__ X,
    const int* __restrict__ Wp,
    float* __restrict__ Y,
    int M,
    int W32,
    int H,
    int D
) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || h >= H) return;

    int matches = 0;
    for (int w = 0; w < W32; ++w) {
        unsigned int x = (unsigned int)X[m * W32 + w];
        unsigned int y = (unsigned int)Wp[w * H + h];
        matches += __popc(~(x ^ y));
    }

    int total_bits = W32 * 32;
    int padding_bits = total_bits - D;
    if (padding_bits > 0) matches -= padding_bits;

    float binary_dot = 2.0f * (float)matches - (float)D;
    Y[m * H + h] = binary_dot / sqrtf((float)D);
}

torch::Tensor xnor_forward(
    torch::Tensor x_u32,
    torch::Tensor w_pack,
    int64_t D64
) {
    CHECK_CUDA(x_u32);
    CHECK_CONTIGUOUS(x_u32);
    CHECK_INT(x_u32);
    CHECK_CUDA(w_pack);
    CHECK_CONTIGUOUS(w_pack);
    CHECK_INT(w_pack);
    TORCH_CHECK(x_u32.dim() == 2, "x_u32 must be [N,W32]");
    TORCH_CHECK(w_pack.dim() == 2, "w_pack must be [W32,H]");
    TORCH_CHECK(x_u32.size(1) == w_pack.size(0), "W32 mismatch");

    int M = (int)x_u32.size(0);
    int W32 = (int)x_u32.size(1);
    int H = (int)w_pack.size(1);
    int D = (int)D64;

    auto out = torch::empty(
        {M, H},
        torch::TensorOptions().dtype(torch::kFloat32).device(x_u32.device())
    );

    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x,
              (H + block.y - 1) / block.y);
    xnor_forward_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        x_u32.data_ptr<int>(),
        w_pack.data_ptr<int>(),
        out.data_ptr<float>(),
        M, W32, H, D
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}


"""

    _XNOR_EXT = load_inline(
        name="rbgc_clean_xnor_u32_v1",
        cpp_sources=cpp_src,
        cuda_sources=cuda_src,
        functions=None,
        with_cuda=True,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=verbose,
    )
    return _XNOR_EXT


def unpack_selected_pm1(
    x_u32: torch.Tensor,
    bit_index: torch.Tensor,
    backend: str = "fused",
    ext_verbose: bool = False,
) -> torch.Tensor:
    """Extract valid selected bits as contiguous float32 [N,G] signs.

    Callers must supply int64 indices in [0, 32*x_u32.shape[1]). The training
    sampler additionally restricts them to [0,D). The fused unpacker does not
    consume random numbers.
    """
    if backend == "fused":
        return load_xnor_ext(verbose=bool(ext_verbose)).unpack_selected_pm1(
            x_u32.contiguous(), bit_index.contiguous()
        )
    raise ValueError("RBGC clean uses only the fused CUDA unpacker")


def sampled_ste_weight_gradient(
    x_u32: torch.Tensor,
    weight: torch.Tensor,
    grad_output: torch.Tensor,
    bit_index: torch.Tensor,
    D: int,
    backend: str = "fused",
    ext_verbose: bool = False,
) -> torch.Tensor:
    """Original sampled clipped STE, with only the unpack implementation changed."""
    x_pm1 = unpack_selected_pm1(x_u32, bit_index, backend, ext_verbose)
    sampled = int(bit_index.numel())
    if sampled == 0:
        raise ValueError("sampled STE needs at least one selected bit")
    grad_selected = (
        x_pm1.T.matmul(grad_output.float())
        * (float(D) / float(sampled))
        / math.sqrt(float(D))
    )
    grad_weight = torch.zeros_like(weight)
    grad_weight.index_copy_(0, bit_index, grad_selected.to(weight.dtype))
    return grad_weight * (weight.abs() <= 1.0).to(weight.dtype)


class PackedXNORLinearFn(torch.autograd.Function):
    """Packed XNOR forward with sampled-bit straight-through backward."""

    @staticmethod
    def forward(
        ctx,
        x_u32: torch.Tensor,
        weight: torch.Tensor,
        D: int,
        grad_bits: int,
        ext_verbose: bool,
        unpack_backend: str = "fused",
    ) -> torch.Tensor:
        ext = load_xnor_ext(verbose=bool(ext_verbose))
        x_u32 = x_u32.contiguous()
        weight = weight.contiguous()

        packed_weight = ext.pack_weight(weight)
        output = ext.xnor_forward(x_u32, packed_weight, int(D))

        ctx.save_for_backward(x_u32, weight)
        ctx.D = int(D)
        ctx.grad_bits = int(grad_bits)
        ctx.unpack_backend = unpack_backend
        ctx.ext_verbose = bool(ext_verbose)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_u32, weight = ctx.saved_tensors
        D = ctx.D
        grad_bits = ctx.grad_bits
        device = grad_output.device

        if grad_bits <= 0 or grad_bits >= D:
            bit_index = torch.arange(D, device=device, dtype=torch.long)
        else:
            bit_index = torch.randperm(D, device=device)[:grad_bits]

        grad_weight = sampled_ste_weight_gradient(
            x_u32, weight, grad_output, bit_index, D,
            ctx.unpack_backend, ctx.ext_verbose,
        )
        # Preserve direct calls of the original five-argument Function.apply.
        return (None, grad_weight, None, None, None, None)[:len(ctx.needs_input_grad)]


class PackedXNORLinear(nn.Module):
    """Binary feature transform X_pack @ sign(W), computed by XNOR-popcount."""

    def __init__(
        self,
        in_bits: int,
        out_channels: int,
        grad_bits: int = 8192,
        ext_verbose: bool = False,
        unpack_backend: str = "fused",
    ):
        super().__init__()
        self.in_bits = int(in_bits)
        self.out_channels = int(out_channels)
        self.grad_bits = int(grad_bits)
        self.ext_verbose = bool(ext_verbose)
        if unpack_backend != "fused":
            raise ValueError(f"unknown unpack backend: {unpack_backend}")
        self.unpack_backend = unpack_backend

        # Official GCN uses a trainable [input_features, channels] kernel.
        self.weight = nn.Parameter(torch.empty(self.in_bits, self.out_channels))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x_u32: torch.Tensor) -> torch.Tensor:
        return PackedXNORLinearFn.apply(
            x_u32,
            self.weight,
            self.in_bits,
            self.grad_bits,
            self.ext_verbose,
            self.unpack_backend,
        )

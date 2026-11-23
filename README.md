# NCSC: Neural Camera Source Clustering

This release provides inference and evaluation code with pretrained weights. 
## Requirements

- Python 3 and CUDA-enabled PyTorch
- NumPy, SciPy, and scikit-learn
- A compatible CUDA toolkit (`nvcc`), C++ compiler, and Ninja for the packed projection extension

After installing CUDA-enabled PyTorch, install the remaining Python dependencies:

```bash
pip install numpy scipy scikit-learn ninja
```

Keep `test_NCSC.py`, `rbgc_packed.py`, and `rbgc_utils.py` in the same directory. The CUDA extension is compiled automatically on first use. CPU-only inference is not supported by this implementation.

## Pretrained Weights

Pretrained weights are available through cloud storage:

- **Download:** [Pretrained checkpoints](https://1drv.ms/u/c/60a66d760ef28686/IQDMNb5gEF34S4R9dBejXvnHAYU7Pd0jW9IaRVe_gKwzkZQ?e=YhA9Zn)

Use a complete `checkpoint.pt` containing both the backbone (`model`) and consolidation (`coarsener`) weights. Model dimensions and saved graph/shuffling settings are read automatically from the checkpoint.

## Data Preparation

We use [prnu-python](https://github.com/polimi-ispl/prnu-python/tree/master) by Luca Bondi, Paolo Bestagini, and Nicolò Bonettini for noise residual extraction. Please refer to its example and acknowledge the original implementation when using it.

The test script consumes precomputed inputs:

- **Packed signs:** a raw `uint32` binary file with shape `[N, ceil(D/32)]`. Bit `d % 32` of word `d // 32` represents residual coordinate `d`, with 1 encoding +1 and 0 encoding -1.
- **Reliability-aware kNN affinities:** an `.npz` file containing neighbor indices and similarities, both shaped `[N, K]`, for example under the keys `nbrs` and `sims`. Neighbor indices are zero-based input row indices.
- **Labels (optional):** a one-dimensional `.npy` array for evaluation only.


## Inference and Evaluation

```bash
python test_NCSC.py \
  --checkpoint ./weights/checkpoint.pt \
  --packed_u32 ./data/packed_u32.dat \
  --reliable_knn_npz ./data/reliable_knn.npz \
  --labels ./data/labels.npy \
  --device cuda:0 \
  --out_dir ./results/test
```

Omit `--labels` to run inference without evaluation. Choose a new or empty output directory. Add `--save_assignments` to save the fine-level and consolidated probability matrices. The number of input rows is inferred from the packed file; `--rows` can optionally be supplied as a consistency check.

If labels refer to a larger original dataset, provide `--source_indices ./data/source_indices.npy --labels_order original`. Use `--label_column` for multi-column labels. Run `python test_NCSC.py --help` for all options.

## Outputs

| File | Contents |
| --- | --- |
| `prediction.npy` | Final cluster labels from `SC`, restored to input row order. |
| `prediction_S.npy` | Fine-level cluster labels before consolidation. |
| `C_final.npy` | Learned consolidation map. |
| `metrics.json` | Evaluation metrics when labels are supplied; use `SC` for final results and `S` for the fine-level reference. Scores are stored on a 0–1 scale. |
| `summary.json` | Model settings and occupied group counts. |

With `--save_assignments`, `S_final.npy` and `SC_final.npy` are also saved in input row order. Cluster IDs are arbitrary group labels, not camera model names.

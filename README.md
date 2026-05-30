# HierAdaptMR

PyTorch implementation of **[HierAdaptMR: Cross-Center Cardiac MRI Reconstruction with Hierarchical Feature Adapters](https://arxiv.org/abs/2508.13026)**.

HierAdaptMR tackles the domain shift that arises when a single MRI reconstruction model is deployed across multiple imaging centers, vendors, and contrasts. A shared **PromptMR-plus** backbone is frozen (or lightly fine-tuned) and augmented with lightweight **hierarchical feature adapters** that specialize the model to each center / vendor / contrast without retraining the whole network. The method was developed for the **CMRxRecon2025** multi-center cardiac MRI reconstruction challenge.

## Key Ideas

- **Shared backbone, specialized adapters.** A single PromptMR-plus reconstruction network is reused everywhere; small residual adapters inject center- and contrast-specific corrections.
- **Hierarchical conditioning.** Metadata (center → vendor → modality / contrast) is parsed directly from the filename and used to select the right adapter at inference time.
- **Lightweight & stable.** Each `FeatureAdapter` is a small Conv–BN–ReLU residual block gated by a learnable, clamped scalar `alpha`, so it nudges the backbone output rather than overriding it.
- **Hierarchical stratified sampling.** Training can use a balanced subset that samples patients evenly across the center / vendor / modality hierarchy to avoid over-fitting to large centers.

## Repository Layout

```
HierAdaptMR-main/
├── train_adapter.py                  # Train hierarchical feature adapters on the frozen backbone
├── finetun.py                        # Fine-tuning variant (backbone + adapters, lower LR)
├── train_with_distribution.sbatch    # SLURM job: multi-GPU (torchrun, 4×A100) training
├── train_no_distribution.sbatch      # SLURM job: single-process training
├── mri_network/
│   ├── promptmrplusV2.py             # PromptMR-plus reconstruction backbone
│   ├── multi_center_adapter.py       # FeatureAdapter + MultiCenterAdaptivePromptMR
│   └── MultiScaleSSIMLoss.py         # Adaptive multi-scale SSIM loss
├── data_loading/
│   ├── mri_data.py                   # CMRxRecon / fastMRI dataset classes
│   ├── data_module.py                # Data module wiring datasets to loaders
│   ├── transforms.py                 # k-space transforms (CmrxReconDataTransform)
│   ├── subsample.py                  # Under-sampling mask functions (CmrxRecon25MaskFunc)
│   └── volume_sampler.py             # Volume-wise sampling for distributed training
└── data_preprocessing/
    ├── data_preprocessing.py         # Convert raw .mat k-space into preprocessed .h5
    ├── split_data.py                 # Train/val split generation
    └── *.py                          # k-space inspection / shape-checking utilities
```

## Model

`MultiCenterAdaptivePromptMR` (`mri_network/multi_center_adapter.py`) wraps the PromptMR-plus backbone:

1. Under-sampled multi-coil k-space and the sampling mask are passed to the backbone to produce an initial reconstruction.
2. Center and contrast metadata are extracted from each sample's filename.
3. The matching `FeatureAdapter` applies a residual correction, `x + clamp(alpha) * adapter(x)`, with UNet-style normalization and padding around the backbone.

This keeps the backbone reusable across all centers while the adapters absorb center-specific appearance differences.

## Data

The pipeline targets the **CMRxRecon2025** multi-coil cardiac dataset. Filenames encode the hierarchy, e.g.:

```
Center001_UIH_30T_umr780_Cine_P001_cine_lax_3ch.h5
```

parsed as `center / vendor / field-strength / scanner / modality / patient / sequence`.

Preprocessing (`data_preprocessing/`) converts raw `.mat` k-space into `.h5` volumes and produces train/val splits. Point the training scripts at the resulting preprocessed directory.

## Setup

```bash
conda create -n ruruCMR python=3.11
conda activate ruruCMR
pip install torch numpy h5py tqdm tensorboard fastmri pyyaml scipy
```

> Adjust the PyTorch / CUDA build to your hardware. The project depends on `fastmri` utilities and an `mri_utils.losses.SSIMLoss` module.

## Training

Train the adapters on top of a pretrained backbone:

```bash
python train_adapter.py \
  --data_path /path/to/preprocessed \
  --experiments_output /path/to/output \
  --pretrained /path/to/backbone_checkpoint.pth.tar \
  --gpus 4 --batch_size 1 --max_epochs 20 --lr 2e-4
```

Useful arguments (defaults in parentheses):

| Argument | Description |
|---|---|
| `--data_path` | Preprocessed dataset root |
| `--experiments_output` | Where checkpoints / TensorBoard logs are written |
| `--pretrained` | Pretrained PromptMR-plus backbone checkpoint |
| `--use_subset` (True) / `--subset_ratio` (0.3) | Use hierarchical stratified subset for training |
| `--gpus` (4) / `--batch_size` (1) | Devices and per-GPU batch size |
| `--lr` (2e-4) / `--lr_step_size` (3) / `--lr_gamma` (0.9) | Optimizer schedule |
| `--max_epochs` (20) | Training length |
| `--num_low_frequencies` ([20]) / `--num_adj_slices` (5) | Sampling / k-t reconstruction settings |
| `--task_type` (`regular_task1`) | Challenge task variant |

`finetun.py` exposes the same interface with a lower default learning rate (1.5e-4) for joint backbone + adapter fine-tuning.

### Multi-GPU (SLURM)

`train_with_distribution.sbatch` launches distributed training with `torchrun` across 4 GPUs:

```bash
sbatch train_with_distribution.sbatch
```

Edit the partition, account, conda environment, and working directory at the top of the script to match your cluster.

## Monitoring

Training logs scalars and sample reconstructions to TensorBoard:

```bash
tensorboard --logdir /path/to/output
```

## Citation

```bibtex
@article{xu2025hieradaptmr,
  title={HierAdaptMR: Cross-Center Cardiac MRI Reconstruction with Hierarchical Feature Adapters},
  author={Xu, Ruru and Oksuz, Ilkay},
  booktitle={International Workshop on Statistical Atlases and Computational Models of the Heart},
  pages={299--310},
  year={2025},
  organization={Springer}
}
```

## Acknowledgements

Built on the **PromptMR-plus** reconstruction backbone and **fastMRI** tooling, and developed for the **CMRxRecon2025** challenge.

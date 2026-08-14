# IE 643 Model Street Reconstruction

Phase 1 is notebook-first. Use the root notebook:

- `../Training_Reconstruction_Model_Final.ipynb`

The notebook builds CIFAR-style ResNet variants, records pretrained-weight provenance, reconstructs class images from model weights, evaluates SSIM/PSNR/MSE/cosine/confidence/runtime, and writes Phase 1 metrics under:

- `../results/phase1_baseline/`

Phase 2 adds objective-ablation runs in the same notebook. It compares registered reconstruction objectives before any API, UI, or deployment work.

## Setup

Use Python 3.8 or newer. Install a PyTorch build that matches your CPU/GPU setup, then install the remaining dependencies:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Quick Run

Open the notebook from the repo root:

```powershell
jupyter notebook ..\Training_Reconstruction_Model_Final.ipynb
```

Run the notebook once with the default settings. The smoke tests run without downloading CIFAR-10 or starting a heavy benchmark.

To run the Phase 1 baseline benchmark:

- Set `RUN_BENCHMARK = True` in the final notebook cell.
- Keep the first run on `resnet18`, classes `(0, 1, 8)`, and seeds `(7, 21, 42)`.
- Set `download_cifar10=True` only if CIFAR-10 is not already available under the repo `data/` directory.
- Each completed run saves a side-by-side reconstructed vs nearest actual reference image with SSIM, PSNR, MSE, cosine, confidence, and runtime in the plot.

To run the Phase 2 objective ablation after reviewing Phase 1 metrics:

- Set `RUN_PHASE2_ABLATION = True`.
- Keep `phase2_objective_names = ("phase1_registered", "deepinv_margin", "plug_in_aug_sigmoid")` for the first comparison.
- Review `phase2_objective_ablation_metrics.csv` and the objective-specific reconstruction grids under `../results/phase1_baseline/`.

To run the Phase 3 cross-model check after choosing a Phase 2 objective:

- Set `RUN_PHASE3_CROSS_MODEL = True`.
- Set `PHASE3_OBJECTIVE_NAME` to the chosen objective, for example `plug_in_aug_sigmoid`.
- Keep the CIFAR-10 checkpoints under `../pre_trained_model_weights/` for full ResNet-18/34/50/101/152 coverage. The notebook reports missing checkpoints as skipped rows if any file is absent.
- Review `phase3_cross_model_metrics.csv`, the model summary table, the class success/failure table, and `phase3_frozen_config.json` before moving to API/UI/deployment.

## Weight Sources

The notebook supports:

- `local_cifar10`: project checkpoints under `../pre_trained_model_weights/`, such as `resnet18_cifar10_trained.pth`.
- MNIST checkpoints are inventoried from `../pre_trained_model_weights/` for later optional evaluation, but Phase 3 remains CIFAR-first.
- `torchvision_imagenet`: official torchvision ImageNet weights where layer shapes are compatible.
- `random`: smoke-test mode only.

Every result row records the weight source, checkpoint path, loaded-layer count, skipped-layer count, and metric values so later Phase 2 improvements can be compared honestly.

The notebook uses the same safe checkpoint-loading pattern as `Reconstruction_deep_learning_final.py`: unwrap `state_dict` or `model`, strip `module.`, keep shape-compatible layers, and load with `strict=False`.

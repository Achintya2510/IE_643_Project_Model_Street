# IE 643 Model Street: Model Inversion And Image Reconstruction

Research-style course project on reconstructing class-faithful CIFAR-10 images from pretrained ResNet weights using model inversion. The project compares baseline reconstruction against objective-level upgrades and cross-model ResNet variants.

## What This Repo Contains

- Final notebook: `Training_Reconstruction_Model_Final.ipynb`
- Core reconstruction code: `src_model_street/Reconstruction_deep_learning_final.py`
- Setup notes: `src_model_street/README.md`
- Experiment plan: `docs/plans/reconstruction-performance-first-upgrade.md`
- Metrics and visual outputs: `results/phase1_baseline/`

Large local datasets, pretrained weights, raw PDFs/slides, and cached tensor outputs are intentionally ignored for GitHub.

## Method

The reconstruction pipeline evaluates ResNet-18/34/50/101/152 checkpoints using:

- safe checkpoint loading from pretrained ResNet weights,
- multi-seed reconstruction search,
- augmentation-aware objective variants,
- BN priors, margin/class loss, TV/L2 regularization, and sigmoid image parameterization,
- nearest-reference comparison against CIFAR-10 class examples.

## Metrics

The project tracks:

- SSIM
- PSNR
- MSE
- nearest-reference embedding cosine
- target confidence
- runtime
- actual-vs-reconstructed visual comparison grids

## Key Results

Phase 2 objective upgrade, `plug_in_aug_sigmoid`, improved over the Phase 1 ResNet-18 baseline:

| Metric | Phase 1 Baseline | Upgraded Objective | Change |
| --- | ---: | ---: | ---: |
| SSIM | 0.014671 | 0.018260 | +24.46% |
| PSNR | 7.801809 | 9.494478 | +21.70% |
| MSE | 0.171696 | 0.132619 | -22.76% |
| Nearest cosine | 0.706924 | 0.713195 | +0.89% |

Phase 3 cross-model analysis completed 38 runs across ResNet-18/34/50/101/152. Deeper models improved class-faithful inversion: ResNet-34 achieved the strongest complete-run nearest-reference cosine mean of `0.927610`, while ResNet-101 achieved target confidence `1.0` across completed runs.

## Reproducing

Install dependencies:

```powershell
cd D:\IE_643_Project\src_model_street
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Open the final notebook from the project root:

```powershell
cd D:\IE_643_Project
jupyter notebook Training_Reconstruction_Model_Final.ipynb
```

To rerun full experiments, place CIFAR-10 data under `data/` and pretrained checkpoints under `Pre_trained_model_weights/`.

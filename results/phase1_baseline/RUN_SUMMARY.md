# Phase 1 vs Phase 2 Metric Run Summary

Run date: 2026-08-07

Scope:

- Model: ResNet-18
- Dataset reference: CIFAR-10 same-class nearest reference
- Classes: airplane `0`, automobile `1`, ship `8`
- Seeds: `7`, `21`, `42`
- Phase 3 cross-model evaluation: not run in this pass

## Aggregate Metrics

| Method | SSIM | PSNR | MSE | Nearest cosine | Target confidence | Avg runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 1 baseline | 0.014671 | 7.801809 | 0.171696 | 0.706924 | 0.173119 | 39.57s |
| Phase 2 `phase1_registered` | 0.010179 | 7.618725 | 0.176471 | 0.706797 | 0.168809 | 37.32s |
| Phase 2 `deepinv_margin` | 0.013437 | 6.846428 | 0.209223 | 0.707540 | 0.199585 | 35.51s |
| Phase 2 `plug_in_aug_sigmoid` | 0.018260 | 9.494478 | 0.132619 | 0.713195 | 0.129274 | 114.86s |

## Best Objective

`plug_in_aug_sigmoid` is the current best Phase 2 objective on aggregate pixel/structure metrics:

- SSIM improved by `+0.003589` or `+24.46%`.
- PSNR improved by `+1.692669 dB` or `+21.70%`.
- MSE reduced by `0.039077` or `22.76%`.
- Nearest embedding cosine improved by `+0.006271` or `+0.89%`.
- Runtime increased by about `190.29%`.
- Target confidence decreased by about `25.33%`, so Phase 2 still needs objective balancing.

Per-run improvement counts for `plug_in_aug_sigmoid` over Phase 1 baseline:

- SSIM improved in `5/9` runs.
- PSNR improved in `7/9` runs.
- MSE improved in `7/9` runs.
- Nearest cosine improved in `6/9` runs.

## Visual Quality Caveat

The side-by-side figures show that Phase 2 improves numeric reconstruction metrics but the generated images are still noisy and not semantically clean. This is acceptable as a first metric-improvement pass, but Phase 2/3 should not be presented as final visual reconstruction quality yet.

## Artifacts

- `phase1_baseline_metrics.csv`
- `phase2_objective_ablation_metrics.csv`
- `phase2_metric_improvement_summary.csv`
- `phase2_per_run_improvement_counts.csv`
- `phase2_metric_comparison_summary.png`
- `figures/*.png` side-by-side reconstructed vs nearest actual reference images with metrics


## Phase 3 Cross-Model Checkpoint Comparison - stopped after sufficient evidence

Run status: stopped after 38 completed rows on 2026-08-08 because the completed set was enough for checkpoint comparison.

Completed evidence:
- ResNet18: 9/9 rows
- ResNet34: 9/9 rows
- ResNet50: 9/9 rows
- ResNet101: 9/9 rows
- ResNet152: 2/9 rows, partial only

Metric winners among complete variants:
- Best mean SSIM: ResNet18
- Best mean PSNR: ResNet18
- Best mean MSE: ResNet34
- Best mean nearest-reference embedding cosine: ResNet34

Interpretation:
ResNet18 should not be described as universally bad because it still wins SSIM/PSNR in the completed numeric table, but it is weak for class-faithful semantic inversion because target confidence and embedding alignment are much lower than deeper checkpoints. ResNet34 is the strongest practical upgrade. ResNet50 and ResNet101 are useful deep-model comparisons, with ResNet101 stronger than ResNet50 in the completed averages, but neither dominates all reconstruction metrics. ResNet152 is only partial evidence and should be described as inconclusive or preliminary rather than final decline.

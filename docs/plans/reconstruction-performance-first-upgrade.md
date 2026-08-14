# IE 643 Model Street Reconstruction Performance Plan

Status: Notebook default execution passed; Phase 1/2/3 benchmark execution pending user run/review.
Last updated: 2026-08-07.

## Goal

Improve reconstruction performance for the IE 643 Model Street project before spending effort on API, UI, or deployment.

Primary outcome:

- More accurate and visually meaningful reconstructions from trained ResNet classifiers.
- Better quantitative metrics than the current baseline.
- Reproducible experiments across model variants and classes.
- Minimal API/UI only after the reconstruction method is stable.
- Deployment remains required, but as the final serving/demo layer.

## IE Repo Findings

Repo found at:

- `D:\IE_643_Project`

Relevant files:

- `AGENTS.md`
- `Model_street_final/README.md`
- `Model_street_final/requirements.txt`
- `Model_street_final/Reconstruction_deep_learning_final.py`
- `Training_Reconstruction_Model_Final.ipynb`
- `Experiments and Testing-Decoder_Final.py`
- `IE_643_Project_Report_Model_Street.pdf`
- `Model_Street_IE643_NoveltyAssessment_report.pdf.pdf`
- `The_final_lap_IE643_project.pdf`

Current project behavior:

- Builds ResNet-18, ResNet-34, ResNet-50, ResNet-101, and ResNet-152 variants.
- Loads CIFAR-10/ImageNet-style checkpoints.
- Reconstructs class-representative images from trained model information.
- Uses logit-guided penultimate embedding optimization.
- Optimizes images against embedding/cosine/logit objectives with image priors.
- Uses losses/regularizers such as embedding MSE, cosine loss, total variation, L2 image regularization, BatchNorm-stat loss, and perceptual loss in some experiments.
- Compares reconstructions against nearest dataset examples using SSIM, PSNR, cosine similarity, and MSE.

Current reported baseline from novelty assessment:

| Model | SSIM | Cosine | PSNR | MSE |
| --- | ---: | ---: | ---: | ---: |
| ResNet18 | 0.0683 | 0.421 | 7.249 | 0.188 |
| ResNet34 | 0.1331 | 0.410 | 7.281 | 0.187 |
| ResNet50 | 0.0786 | 0.405 | 6.119 | 0.244 |
| ResNet101 | 0.0938 | 0.761 | 6.462 | 0.225 |
| ResNet152 | 0.0801 | 0.507 | 6.532 | 0.222 |

The main upgrade target is to improve these reconstruction metrics and the qualitative visual quality.

## Priority Order

1. Reproducible baseline and metric harness.
2. Reconstruction objective improvement.
3. Ablation-backed metric improvement.
4. Robust inference wrapper.
5. Minimal API.
6. Minimal UI.
7. Deployment.

Do not add backend features unless they are required to serve the final reconstruction method.

## Research Papers To Use

These papers are directly relevant to this project because it is model inversion / feature inversion, not generic image restoration.

- DeepInversion: Dreaming to Distill: Data-Free Knowledge Transfer via DeepInversion, CVPR 2020, https://openaccess.thecvf.com/content_CVPR_2020/html/Yin_Dreaming_to_Distill_Data-Free_Knowledge_Transfer_via_DeepInversion_CVPR_2020_paper.html
- Plug-In Inversion: Model-Agnostic Inversion for Vision with Data Augmentations, ICML 2022, https://proceedings.mlr.press/v162/ghiasi22a.html
- Understanding Deep Image Representations by Inverting Them, CVPR 2015, https://openaccess.thecvf.com/content_cvpr_2015/html/Mahendran_Understanding_Deep_Image_2015_CVPR_paper.html
- Reconstructing Training Data From Trained Neural Networks, NeurIPS 2022, https://papers.nips.cc/paper_files/paper/2022/hash/906927370cbeb537781100623cca6fa6-Abstract-Conference.html
- Reconstructing Training Data From Real-World Models Trained with Transfer Learning, OpenReview 2024/2025, https://openreview.net/forum?id=BRdYYyrAOR
- Model Inversion Attacks that Exploit Confidence Information and Basic Countermeasures, ACM CCS 2015, https://dl.acm.org/doi/10.1145/2810103.2813677
- The Secret Revealer: Generative Model-Inversion Attacks Against Deep Neural Networks, CVPR 2020, https://doi.org/10.1109/CVPR42600.2020.00033

Practical reading path:

- Use DeepInversion and Plug-In Inversion first because they match the current BN-stat and augmentation-prior direction.
- Use Mahendran and Vedaldi for the feature-inversion objective framing.
- Use Haim/Oz papers for the training-data reconstruction framing and privacy/reconstruction claims.
- Use generative model inversion only if we approve adding a generative prior; do not make GAN/diffusion priors required for MVP.

## Metrics Contract

Required metrics:

- SSIM against nearest same-class dataset image.
- PSNR against nearest same-class dataset image.
- MSE against nearest same-class dataset image.
- Embedding cosine similarity between reconstruction and target/optimized embedding.
- Target class confidence or target logit.
- Top-1 and top-5 predicted class consistency.
- Runtime per reconstruction.
- Multi-seed mean and standard deviation for each metric.

Optional metrics:

- LPIPS if package/setup cost is acceptable.
- FID/KID only if enough generated samples are produced.
- Diversity score across seeds for the same class.
- Human qualitative ranking for final demo images.

Important rule:

- Do not report single cherry-picked reconstructions as final results. Use fixed class lists, fixed seeds, fixed checkpoint variants, and aggregate metrics.

## Target Technical Architecture

Research pipeline:

- Checkpoint loader
- ResNet variant builder
- Feature/BatchNorm hook collector
- Class embedding optimizer
- Image optimizer
- Reconstruction objective registry
- Augmentation-prior module
- Metric evaluator
- Experiment runner
- Results table and visual artifact generator

Serving pipeline:

- Best reconstruction config
- Best checkpoint/variant loader
- Minimal inference wrapper
- Minimal FastAPI endpoint
- Minimal UI for class/model selection and sample outputs
- Docker deployment

No database, auth, queue, admin dashboard, vector DB, or multi-user history is required.

## Approval Gates

Ask for approval before these decisions:

| Gate | Decision | Default recommendation |
| --- | --- | --- |
| Gate 0 | Correct repo placement | Put this plan in `D:\IE_643_Project/docs/plans/` |
| Gate 1 | Model scope | Improve CIFAR-10 ResNet variants first; ImageNet variants second |
| Gate 2 | Primary metric | Optimize aggregate SSIM/PSNR/MSE plus embedding cosine, not one metric alone |
| Gate 3 | Reconstruction approach | Keep optimization-only inversion as main path; decoder training only as an approved comparison |
| Gate 4 | Objective upgrades | Add DeepInversion-style BN/stat priors plus Plug-In Inversion augmentations |
| Gate 5 | Compute budget | Approve number of models, classes, seeds, steps, and GPU/CPU limits |
| Gate 6 | API/UI scope | Minimal demo only after metrics improve |
| Gate 7 | Deployment target | Deploy the best config only, not every experimental path |

## Phase 0: Repo Placement and Experiment Contract

Goal: make sure implementation targets the right IE files and metrics before changing code.

Tasks:

- [x] Create `docs/plans/` in `D:\IE_643_Project`.
- [x] Place this plan at `D:\IE_643_Project/docs/plans/reconstruction-performance-first-upgrade.md`.
- [x] Confirm the exact source files to modify.
- [x] Confirm which checkpoints are available locally versus downloaded through Kaggle.
- [x] Confirm CIFAR-10 and ImageNet dataset access paths.
- [x] Define the fixed class list for evaluation.
- [x] Define the fixed seed list for evaluation.
- [x] Define the model variants included in the first benchmark.
- [x] Define primary and secondary metrics.
- [x] Decide whether decoder-based reconstruction remains a comparison or becomes part of the main path.

Verification:

- [x] Plan exists in the IE repo.
- [x] Baseline script/notebook can be located.
- [x] Checkpoint paths are known.
- [x] Dataset loading strategy is known.

Approval gate:

- Gate 0 through Gate 5 must be approved before implementation.

Implementation notes:

- The implementation target is the root notebook `Training_Reconstruction_Model_Final.ipynb`.
- Phase 1 keeps one clean notebook rather than splitting into a package because the user requested a single notebook first.
- Local checkpoint inspection now uses `pre_trained_model_weights/` first, with fallback to `Model_street_final/`. CIFAR-10 checkpoints are available for ResNet-18/34/50/101/152, and MNIST checkpoints are available for later optional evaluation.
- First benchmark scope is CIFAR-10 ResNet-18, classes `0, 1, 8`, seeds `7, 21, 42`, with SSIM, PSNR, MSE, embedding cosine, confidence, top-k consistency, and runtime.
- Decoder-based reconstruction remains a later comparison only; optimization-only inversion stays the main path.

Learnings:

- The old notebook/script style had hard-coded Kaggle and Colab paths, so a single reproducible notebook is cleaner for Phase 1.
- Weight provenance must be logged with metrics because ImageNet pretrained weights and local CIFAR checkpoints provide different reconstruction information.
- A small smoke-test cell is enough for Phase 1 validation before running GPU-heavy benchmarks.

Post-review changes:

- Updated the notebook weight-loading helper to mirror `Model_street_final/Reconstruction_deep_learning_final.py`: accept a path or state dict, unwrap `state_dict`/`model`, strip `module.`, load only shape-compatible layers with `strict=False`, and print loaded/skipped layers.
- Kept provenance fields around that loader so benchmark rows still record weight source, loaded-layer count, skipped-layer count, and checkpoint/source description.

Pause:

- Stop after this phase for review.

## Phase 1: Reproducible Baseline Harness

Goal: make the current results reproducible before improving them.

Tasks:

- [x] Extract reusable logic from notebook-style code without changing behavior.
- [x] Create a configurable experiment runner for model variant, class ID, seed, optimization steps, and loss weights.
- [x] Standardize checkpoint loading.
- [x] Standardize CIFAR preprocessing and inverse display transforms.
- [ ] Standardize ImageNet dataset preprocessing for later ImageNet benchmarks.
- [x] Cache dataset features for nearest-neighbor comparison.
- [x] Compute SSIM, PSNR, MSE, cosine, target logit/confidence, top-k consistency, and runtime.
- [x] Save results to a structured CSV or JSON file.
- [x] Save reconstruction grids and nearest-neighbor comparison grids.
- [x] Add small tests for model loading, metric functions, output shape, and deterministic seed behavior.

Verification:

- [x] Reproduce current baseline for at least one ResNet variant.
- [x] Run baseline benchmark on the approved model/class/seed subset.
- [ ] Confirm metrics are stable across repeated runs with the same seed.
- [x] Confirm visual artifacts are generated automatically.
- [x] Validate notebook JSON structure.

Approval gate:

- Approve reproduced baseline table before objective tuning starts.

Implementation notes:

- Replaced the old path-specific root notebook with a single clean Phase 1 notebook.
- Added explicit pretrained-weight provenance for `local_cifar10`, `torchvision_imagenet`, and `random` smoke-test modes.
- Added CIFAR-style ResNet builders for ResNet-18/34/50/101/152, compatible checkpoint loading, deterministic seeds, reconstruction objective, reference embedding cache, metric computation, CSV/JSON result export, and figure export.
- Kept the benchmark opt-in with `RUN_BENCHMARK = False` so notebook validation does not accidentally start a heavy run.
- Updated documentation dependencies to include pandas because the notebook uses tabular metric summaries.
- Local validation confirmed notebook JSON validity and Python compile validity. Notebook default execution later passed after installing missing runtime dependencies.

Learnings:

- Keeping the first deliverable as one notebook is the more elegant Phase 1 path for a placement-ready project; package extraction can wait until deployment/API work.
- The actual metric-improvement claim must wait for a benchmark run and Phase 2 objective tuning.
- Only clearly generated cache files should be removed automatically; older reports and presentations are better kept until the user approves a broader repo cleanup.

Post-review changes:

- Updated reconstruction comparison figures to show reconstructed image, nearest actual same-class reference image, and the relevant metrics in one saved plot.
- Added a metrics text panel with model, objective, class, seed, SSIM, PSNR, MSE, nearest embedding cosine, target confidence, top-k hit, runtime, and candidate score when available.

Pause:

- Stop after this phase for review.

## Phase 2: Objective and Prior Improvements

Goal: improve reconstruction quality with research-backed changes before adding app features.

Tasks:

- [x] Implement a clean objective registry for loss combinations.
- [x] Add DeepInversion-style BatchNorm-stat loss across all useful BN layers, not hard-coded single layers.
- [x] Add Plug-In Inversion augmentations: jitter, random crop/resize, color jitter, flip if appropriate, and multi-crop consistency.
- [x] Add target-class margin loss in addition to raw logit maximization.
- [x] Add multi-logit or top-k negative-class suppression if approved.
- [x] Add image parameterization options: direct pixels, sigmoid/tanh parameterization, and low-frequency initialization.
- [x] Add optimizer/schedule ablations: Adam, cosine LR decay, and multi-start restart-style search.
- [x] Add loss-weight scheduling so priors and class objectives do not fight throughout optimization.
- [x] Add multi-start candidate generation and select by composite validation score.
- [ ] Add optional VGG/perceptual loss only if it improves metrics or visual quality.
- [x] Compare each change against the reproduced baseline through the added ablation runner.

Verification:

- [x] Produce ablation table by running `RUN_PHASE2_ABLATION=True`.
- [x] Show aggregate metric improvement over baseline.
- [x] Generate qualitative comparison grids with objective-specific filenames when ablation runs.
- [x] Confirm improvements are not limited to one class or one seed.
- [x] Validate notebook JSON structure and compile all notebook code cells.

Approval gate:

- Approve best objective/config before deeper model-specific work.

Implementation notes:

- Added Phase 2 notebook sections after the baseline comparison path, keeping Phase 1 as the unchanged baseline.
- Added `ObjectiveConfig` and `PHASE2_OBJECTIVES` for `phase1_registered`, `deepinv_margin`, `plug_in_aug_sigmoid`, and `tanh_multistart`.
- Added direct, sigmoid, and tanh image parameterization plus optional low-frequency initialization.
- Added Plug-In style augmentation views: random resized crops, jitter/roll, optional horizontal flip, color jitter, and multi-crop consistency.
- Added margin loss, top-k non-target suppression, scheduled class/BN weights, cosine LR schedule, and deterministic multi-start candidate selection.
- Multi-start now fixes the optimized target embedding per class/seed and varies only the image candidate initialization, so candidate selection compares reconstructions against the same pretrained-weight target signal.
- Added `run_phase2_ablation`, grouped ablation summaries, and phase1-vs-phase2 comparison helper.
- Updated figure saving to include `objective_name` so ablation outputs do not overwrite baseline figures.
- Added Phase 2 smoke coverage to the notebook smoke test cell.
- Notebook default execution passed after installing `torchvision`, `scikit-image`, `nbconvert`, and plotting/notebook utility dependencies. Metric verification remains pending because `RUN_BENCHMARK`, `RUN_PHASE2_ABLATION`, and `RUN_PHASE3_CROSS_MODEL` were intentionally left off.
- During notebook execution, the smoke reconstruction exposed an autograd failure caused by in-place ReLU operations in the notebook ResNet blocks. The notebook now uses non-in-place ReLU modules so input-image optimization can backpropagate safely.
- Ran Phase 1 baseline and Phase 2 objective ablation on 2026-08-07 for ResNet-18, CIFAR-10 classes `0, 1, 8`, and seeds `7, 21, 42`.
- Best objective from this run is `plug_in_aug_sigmoid`: SSIM improved from `0.014671` to `0.018260`, PSNR improved from `7.801809` to `9.494478`, MSE reduced from `0.171696` to `0.132619`, and nearest embedding cosine improved from `0.706924` to `0.713195`.
- Per-run improvement for `plug_in_aug_sigmoid`: SSIM improved in `5/9`, PSNR in `7/9`, MSE in `7/9`, and nearest cosine in `6/9`.
- Visual quality remains noisy despite numeric improvement, so the next refinement should balance image naturalness/semantic clarity against pixel metrics.

Learnings:

- Objective improvements should be compared through the same class/seed/checkpoint subset as Phase 1; otherwise the metric claim becomes weak.
- The cleanest Phase 2 deliverable is an ablation runner, not a single hard-coded "best" setting.
- VGG/perceptual loss should remain deferred until the simpler DeepInversion and augmentation objectives are benchmarked.

Post-review changes:

- To be filled after user review, if needed.

Pause:

- Stop after this phase for review.

## Phase 3: Cross-Model Generalization and Optional Decoder Comparison

Goal: make the improvement robust across ResNet variants without overbuilding.

Tasks:

- [x] Evaluate the improved method on approved ResNet variants through an opt-in runner.
- [x] Compare CIFAR-10 variants first.
- [x] Evaluate ImageNet variants only after CIFAR improvements are clear.
- [x] Normalize metrics by dataset/resolution so comparisons are fair.
- [ ] Compare optimization-only reconstruction against existing decoder-based reconstruction if approved.
- [x] Do not train a new large decoder unless the user approves compute and expected benefit.
- [x] Produce final model-by-model summary through an opt-in summarizer.

Verification:

- [ ] Report per-model aggregate metrics after running `RUN_PHASE3_CROSS_MODEL=True`.
- [ ] Report per-class success/failure cases after running `RUN_PHASE3_CROSS_MODEL=True`.
- [ ] Identify which ResNet depths reconstruct best and why after benchmark execution.
- [x] Add frozen-config candidate writer for serving approval after metrics exist.
- [x] Validate notebook JSON structure and compile all notebook code cells.

Approval gate:

- Approve final method and model scope before API/UI/deployment work.

Implementation notes:

- Added a Phase 3 cross-model section to the single notebook.
- Added checkpoint inventory for approved CIFAR-10 ResNet variants and clear skipped rows for missing checkpoints.
- Added `RUN_PHASE3_CROSS_MODEL` with `PHASE3_OBJECTIVE_NAME`, defaulting to the Phase 2 candidate objective without claiming it is best before metrics are run.
- Added model-by-model summary, class-level success/failure table, serving-candidate selector, and frozen-config JSON writer.
- Added normalized metric fields for dataset name, resolution, pixel count, ResNet depth, normalized MSE, and normalized PSNR.
- Added decoder-comparison inventory only; no decoder training or comparison run is started without approval.
- Local checkpoint inventory now points to `pre_trained_model_weights/`, where CIFAR-10 ResNet-18/34/50/101/152 checkpoints are present. MNIST checkpoints are also inventoried but not benchmarked in Phase 3 by default.
- Notebook default execution passed. Cross-model metric verification remains pending until `RUN_PHASE3_CROSS_MODEL=True` is run after Phase 2 objective review.

Learnings:

- Cross-model code must not fail when only one checkpoint is present; missing variants should be visible as skipped, not hidden.
- Freezing a serving config should be based on completed aggregate metrics, not on a hard-coded preference.
- Decoder comparison is useful only if the decoder checkpoint is available and the user approves using it as a comparison path.

Post-review changes:

- To be filled after user review, if needed.

Pause:

- Stop after this phase for review.

## Phase 4: Minimal Inference, API, and UI

Goal: expose only the best reconstruction workflow.

Tasks:

- [ ] Create a minimal inference wrapper for the frozen config.
- [ ] Add sample-class reconstruction for approved models.
- [ ] Add minimal FastAPI app.
- [ ] Add health endpoint.
- [ ] Add reconstruction endpoint.
- [ ] Add sample/model metadata endpoint.
- [ ] Add minimal UI for selecting model variant, class ID, seed, and showing reconstruction output.
- [ ] Show metrics if nearest-neighbor evaluation is available.
- [ ] Add one API smoke test and one UI/manual smoke test.

Required endpoints:

- `GET /health`
- `GET /samples`
- `POST /reconstruct`

Do not add:

- auth
- database
- job queue
- user history
- admin dashboard
- multi-model management beyond the approved demo variants

Verification:

- [ ] Run API locally.
- [ ] Generate a reconstruction through API.
- [ ] Use UI to generate and view a reconstruction.
- [ ] Confirm invalid inputs return clear errors.

Approval gate:

- Gate 6 must be approved before UI expansion beyond this minimal scope.

Implementation notes:

- To be filled during the phase.

Learnings:

- To be filled during the phase.

Post-review changes:

- To be filled after user review, if needed.

Pause:

- Stop after this phase for review.

## Phase 5: Deployment and Final Results Package

Goal: deploy the improved reconstruction demo and make the results defensible.

Tasks:

- [ ] Add Dockerfile or deployment-specific runtime config.
- [ ] Package the frozen reconstruction config and required checkpoint references.
- [ ] Include a small approved sample/demo set.
- [ ] Deploy API and UI to the approved platform.
- [ ] Add setup, benchmark, inference, and deployment instructions to README.
- [ ] Add final results table: current baseline versus improved method.
- [ ] Add ablation table.
- [ ] Add qualitative figures.
- [ ] Add short architecture notes explaining model inversion, losses, metrics, and deployment.
- [ ] Add resume/interview notes focused on metric improvement and reconstruction quality.

Verification:

- [ ] Run all tests.
- [ ] Run final benchmark command.
- [ ] Build deployment artifact locally.
- [ ] Open deployed URL.
- [ ] Generate one deployed reconstruction.
- [ ] Confirm deployed output matches local output within expected tolerance.
- [ ] Confirm final metrics are reproducible from documented commands.

Approval gate:

- Gate 7 must be approved before platform-specific deployment work.

Implementation notes:

- To be filled during the phase.

Learnings:

- To be filled during the phase.

Post-review changes:

- To be filled after user review, if needed.

Pause:

- Stop after this phase for final review.

## Non-Goals Unless Approved

- Large backend platform.
- Authentication.
- Database.
- Job queue.
- Admin dashboard.
- Rewriting the project into a web app before metrics improve.
- Training a large new decoder without baseline and ablation evidence.
- GAN or diffusion priors unless simpler DeepInversion/Plug-In Inversion upgrades plateau.
- Supporting every checkpoint in the deployed demo.
- Deployment before reconstruction metrics are acceptable.

## Success Criteria

The project is successful when:

- The current baseline is reproduced.
- Improved method beats baseline on approved aggregate metrics.
- Qualitative reconstructions visibly improve.
- Improvements hold across multiple classes and seeds.
- At least the approved CIFAR-10 ResNet variants are benchmarked.
- The final method is frozen and reproducible.
- Minimal API and UI serve the best method.
- Deployed demo works.
- README explains setup, evaluation, inference, and deployment.
- Final report/resume notes can defend the metric improvement and model-inversion design.

## Final Review Section

Plan creation review:

- Re-grounded the plan in the actual IE 643 repository found at `D:\IE_643_Project`.
- Replaced generic image restoration assumptions with the project's actual model-inversion/class-reconstruction workflow.
- Added current reported baseline metrics from the novelty assessment report.
- Added research papers directly relevant to feature inversion, DeepInversion, Plug-In Inversion, and training-data reconstruction.
- Kept API, UI, and deployment later in the plan.
- Made deployment required while keeping backend features minimal.


### Phase 3 Cross-Model Run Notes - 2026-08-08

Implementation notes:
- Ran the selected `plug_in_aug_sigmoid` objective across local CIFAR-10 ResNet checkpoints using an incremental runner.
- Completed full 9-row comparisons for ResNet18, ResNet34, ResNet50, and ResNet101.
- Stopped ResNet152 after 2 completed rows because the user approved stopping once evidence was sufficient for comparison.
- Saved completed-model summaries to `results/phase1_baseline/phase3_completed_model_summary.csv` and `results/phase1_baseline/phase3_model_selection.json`.

Learnings:
- Increasing ResNet depth does not monotonically improve reconstruction quality.
- ResNet18 remains strong on SSIM/PSNR but weak for class-faithful semantic inversion.
- ResNet34 is the best practical checkpoint trade-off in the completed evidence, with best mean MSE and strongest embedding alignment among complete variants.
- ResNet50 and ResNet101 are useful deep-model comparisons but do not dominate all metrics.
- ResNet152 is too slow for the current CPU-only phase and has only partial evidence.

# Lessons

## Scope plans for hackathon and placement readiness

- When planning a placement-ready hackathon project, keep the required path focused on the problem behavior and the minimum defensible system around it.
- Do not include enterprise infrastructure, large dashboards, advanced retrieval, worker queues, or voice pipelines unless they directly support the requested MVP or the user approves them.
- Mark auth, async jobs, deployment, and UX expansion as conditional when the main requirement is AI/business logic correctness.

## Keep backend scope minimal when deployment is required

- Deployment being critical does not make every backend hardening feature required.
- Keep only the backend pieces needed for a working deployed demo: server-side secrets, API routes, persistence, migrations/seed data, CORS when needed, health check, simple logs, and smoke tests.
- Add JWT auth, ownership checks, job queues, readiness probes, or model-run tables only when the deployed product requirement actually needs them.

## Put model quality before app surface when the project is metric-driven

- If the user says model/reconstruction performance is the main goal, plan metrics, dataset splits, baselines, ablations, and research-backed model improvements before API, UI, or deployment.
- Treat backend and UI as serving layers for the best model, not as the main project.
- Ask for approval on repo/data paths, domain, metric, model family, and compute budget before training-heavy work.

## Re-ground when the user clarifies the target repo

- If the user clarifies that a plan belongs in another repo, locate that repo, read its local instructions, and move the plan there before presenting it as final.
- Do not carry assumptions from a different PDF or workspace once the actual repo has been found.

## Put model quality before app surface when the project is metric-driven

- If the user says model/reconstruction performance is the main goal, plan metrics, dataset splits, baselines, and research-backed model improvements before API, UI, or deployment.
- Treat backend and UI as serving layers for the best model, not as the main project.
- Ask for approval on repo/data paths, domain, metric, dataset split, model family, and compute budget before training-heavy work.

## Re-ground when the user clarifies the target project

- If the user says a plan is for the current repo/project, verify the exposed workspace paths and local files before carrying assumptions from a different PDF or project.
- Make mismatched workspace/project context an explicit approval gate instead of silently planning against the wrong target.

## Keep metric-first reconstruction work notebook-first when requested

- If the user asks for a single notebook first, implement the reproducible experiment harness in that notebook before creating extra scripts, packages, APIs, or UI files.
- Keep optional backend/deployment scaffolding out of the first phase when the immediate goal is reconstruction performance and metric evidence.
- Remove only clearly generated or obsolete artifacts automatically; keep reports, presentations, and source references unless the user approves a broader cleanup.

## Mirror existing research code before improving it

- When converting project notebook/script logic into a cleaner harness, first preserve the existing loading semantics exactly enough that old checkpoints behave the same way.
- For this IE project, use the source script's safe checkpoint loader pattern: unwrap `state_dict` or `model`, strip `module.`, filter shape-compatible layers, and load with `strict=False`.
- Add metric provenance around the preserved behavior, but do not silently replace it with a different checkpoint-loading strategy.

## Keep inversion models autograd-safe

- Model inversion optimizes the input image through the classifier, so architecture code must preserve gradients through activations.
- Avoid in-place ReLU operations in reconstruction/inversion harnesses; they can pass normal inference tests but fail during `loss.backward()` on optimized inputs.

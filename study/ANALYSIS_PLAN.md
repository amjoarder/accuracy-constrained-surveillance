# Frozen deployment-study analysis plan

## Research question
Can independent per-task inference settings reduce sequential dual-detector processing cost while retaining both AP50 and AP50:95 in each task, and do validation constraints transfer to the respective test partitions?

## Scope and novelty boundary
Configuration adaptation, mixed precision, and Pareto optimization are established methods (Chameleon, Mainstream, recent edge-detector benchmarks). This study contributes their explicitly constrained application to heterogeneous object/behavior box detectors, a measured evaluation of independent versus uniform settings, validation-to-test retention transfer, and complete output-path accounting on a 4GB laptop GPU. It does not introduce a YOLO architecture, claim first-ever adaptation, infer violent intent, or turn component AP into joint event accuracy.

## Primary selection
Six candidate settings per task: resolutions 320, 480, 640; FP32 and FP16. Validate with square aspect-preserving letterbox, batch one, conf floor .001, NMS .7, maximum300. Reference 640 FP32. Require both AP50 and AP50:95 in each task to retain at least95% of the corresponding validation reference metric. Minimize the sum of branch latencies measured on dedicated unlabeled video inputs. Enumerate independently; with separable costs/constraints this is the exact Cartesian-product optimum. Freeze primary selection before new test trials. Retention levels90%,97.5%,100% are sensitivity analyses.

## Baselines and ablations
Before policy freeze or any new test access, the configuration space is extended with640 minimum-stride rectangular FP32 and FP16 (eight candidates per task,64 pairs). Rectangular validator shapes use pad0 instead of the library's default pad.5, matching runtime minimum-stride padding. Add uniform640 rectangular FP32/FP16 complete-output baselines (seven policies, three repeats each). All completed square trials remain unchanged. The reference remains square640FP32. The amendment JSON records the timing and reason; the extension tests the observed sensitivity to evaluation/deployment padding contracts.

Uniform640FP32; uniform640FP16 (precision only); uniform480FP16 (uniform downscaling); validation-selected independent policy; original stretched640FP32 pipeline. Profile all five, even if policy settings duplicate a baseline: duplicate trials provide measured variation and are not different algorithms. All six component test settings are descriptive trade-offs, not selection data. Report AP50 and AP50:95 separately; do not average incompatible task metrics into incident accuracy. Track per-class AP and box-match P/R at confidence.5 and IoU.5. Background error is number of background images with at least one retained prediction divided by number of background images; not false alerts per hour.

## Timing
Branch microprofile:30 identical in-memory frames, five untimed warmups, three passes. Full-output profile: firstmin(50,available)frames in each of five clips (239/pass), three repeats/policy in seeded shuffled trial order. Includes decode, inference, postprocessing/extra objectNMS, CPU transfer, text logs, behavior-triggered rawJPEG snapshots, overlays, MJPG output, GPU synchronization and final flush/release. Excludes model loading, opening capture/writer, warmups and GUI. Report repeat rates, pooled throughput, median/p95frame times and memory. Different alert outputs naturally create different output-write workloads; document counts. No energy claims without a power instrument.

## Statistical interpretation
Two predeclared constraint ablations select using validation/cost only: AP50-only retention (omit AP50:95 constraint), and pooled normalized retention (average same-metric ratios across tasks instead of requiring each task). Enumerate the same 36 branch-setting pairs. These test whether loosening the constraints hides localization or task-specific degradation. Report their fixed component-test outcomes from the six-setting grid; no extra full-output timing trial or performance claim is inferred without direct measurement.

An additional class-preservation diagnostic applies95% retention to AP50 and AP50:95 for every foreground class with nonzero reference AP, using validation and branch costs only. Its selected configurations and per-class test retention are reported from the existing fixed grid, without inferring unmeasured complete-path speed. This extension is specified before any new test trial and does not change the primary task-level policy.

Paired image-resampling intervals may describe conditional sensitivity within the supplied image sets. They do not resolve correlated recordings, contaminated historical selection, or population generalization. No arbitrary p-value fishing or cross-paper numerical ranking. A retention threshold is an engineering tolerance, not a validated safety threshold.

## Dataset checks
Normalize objectvalidation rows before evaluating; exclude exact-byte/decoded-pixel overlap with objecttraining. Objecttest425 already exact-overlap-filtered and row-normalized. Behavior230validation/115test contain background images and no exact within-archive overlap. Cross-archive shared915images prevent a large independent joint test; retain the ten paired frames only as exploratory supplementary evidence.

## Resume and provenance
Save protocol before runs. Atomic complete.json after each trial; no rerun of complete trials on resume. Preserve original weights and archives. Track source/script/checkpoint hashes, data manifests, matching arrays, per-image counts, and policy freeze time.

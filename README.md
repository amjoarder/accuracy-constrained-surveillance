# Accuracy-Constrained Deployment of Object and Behavior Detectors for Video Surveillance

Research code and numerical records accompanying the manuscript by Ajmain Muhtady Joarder, Zannatul Tasmia Moyury, Jubayer Al Mahmud, Md. Alam Hossain, M. F. Mridha.

Corresponding author: Jubayer Al Mahmud (ja.mahmud@just.edu.bd). The manuscript is prepared for submission; no publication acceptance or DOI is claimed.

## Verify the numerical results on a laptop

```bash
python -m pip install -r requirements-numerical.txt
python study/verify_repository.py
python study/reproduce_supplement.py
python study/check_geometry.py
python -m unittest discover -s study -p test_policy.py
python -m unittest discover -s study -p test_portability.py
python -m unittest discover -s study -p test_training_records.py
```

The checker independently recomputes 37 AP trials, nine complete-output throughput summaries and 1,050 external clip/policy records. Geometry is checked for 1,627 images using published native dimensions. No GPU, images, model weights or video downloads are needed. GitHub Actions runs file integrity, numerical, geometry and portability checks on Linux and Windows. See [the full repository recheck](docs/REPOSITORY_VERIFICATION.md) for coverage and limitations.

## Main observations

The 95% validation-retention choice uses 640-pixel rectangular FP32 inputs for both detectors and reaches 7.53 FPS versus 5.63 FPS for square FP32. It matches a uniform rectangular preset; it does not demonstrate heterogeneous allocation gains. The stricter 100% choice combines 480-pixel square object inputs and 640-pixel rectangular behavior inputs, reaching 6.48 versus 5.23 FPS in its separately blocked comparison. Selected behavior balanced accuracy on AIRTLab is 48.44%; high component AP does not establish transferable violence recognition.

## Contents and limits

`study/results/` contains the recorded numerical trials; `study/` contains evaluation and analysis scripts, protocols and input fingerprints. `figures/` contains aggregate plots. See [reproduction instructions](docs/REPRODUCIBILITY.md) and [publication transformations](docs/PUBLICATION_CHANGES.md).

Raw source imagery, participant footage, model weights and generated snapshots are excluded. Their redistribution rights have not been established. Full inference reruns require the identified source artifacts and environment; published records enable numerical verification without them. The code includes no new detector training. Private media names are replaced with opaque IDs and workstation prefixes with `<WORKSPACE>`; original scientific arrays and reported metrics remain unchanged.

The authors created the study datasets by preparing captured and collected images; captured images were manually annotated in Roboflow, while downloaded datasets supplied existing labels. The deployment experiments reuse the authors' previously fine-tuned object and behavior best.pt checkpoints. Original training commands, arguments and epoch logs are provided in study/training/. Read [dataset and training documentation](docs/DATASET_AND_TRAINING.md). The original datasets and both trained checkpoints are available from corresponding author Jubayer Al Mahmud (ja.mahmud@just.edu.bd) on reasonable request for research verification, including peer review.

No blanket reuse license is granted for the authors' custom code or records in this initial release. Contact the corresponding author for reuse permission. Third-party dependencies and AIRTLab retain their own terms.

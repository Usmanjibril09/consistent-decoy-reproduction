# Command-conditioned process decoys for industrial IoT honeypots

Research code for the HAI and Quanser Aero 2 evaluations and the exploratory
SimProcess reassessment. This source release excludes datasets, processed
windows, checkpoints, virtual environments, and generated results.

## Reproduction entry points

Use Python 3.11 and the stage-specific requirements in `spec/`. Obtain the data
from its providers and preserve the input layout in `configs/experiment.toml`.

| Data | Local layout |
| --- | --- |
| [HAI](https://github.com/icsdataset/hai), releases 23.05 and 22.04 | `data/raw/HAI/hai-23.05/` and `data/raw/HAI/hai-22.04/` |
| Quanser Aero 2 pick-and-place data, Zenodo DOI `10.5281/zenodo.20287835` | `data/raw/EUROCAST/` |

Use `scripts/00_inventory.py` to check inputs, then run
`scripts/01_preprocess.py` to build `data/processed/`. Train using the commands
below. The held-out stages start at `scripts/03_evaluate.py`; consult
`spec/PROTOCOL.md` and `spec/decisions.md` for stage order, inputs, and
experimental settings. Use the separate Kaggle export workflow only when you
intend to create a private training-data bundle.

Keep the Git history and tags. The full experiment scripts use them to verify
the frozen protocol and code. For an archive distribution, clone the included
Git bundle before running the experiment.

## Included methods

B1 replay; B2 ARX; B3 constrained first-order-plus-delay model; B4 warm-started
unconditioned LSTM with fitted residual noise; B5 warm-started RCGAN adaptation;
B6 latent GAN plus autoencoder adaptation; B7 TimeGAN; B8 causal Transformer;
B9 NeuPot-inspired LSTM/Huber forecaster; C1 conditional GRU; A3 no future commands.
Full, A1, A2, A4 and A5 share selected C1 weights. A3 uses its own weights with C2/C3.
The inference composition is C3 blend, then C2 correction. All layer parameters
come from training recordings, with no transitions across recording boundaries.
Exogenous inputs are replayed by every method, explicitly limiting the task.

B6 is **not a faithful NeuralPot reproduction**: the accepted manuscript compares
separate GAN and autoencoder implementations, whereas D29 requests their composition.
B4, B5 and B9 are also adaptations, not claims of exact published implementations.
B7 follows the original three-stage objectives and weights; implementation details
and deviations are recorded in spec/decisions.md (D30).

## Local checks

Use the project's existing Python environment. No additional packages are required
for Phase 2 beyond Python >=3.11, NumPy and PyTorch.

```sh
python -m unittest discover -s tests
python scripts/02_train.py --smoke --out results/stage2-smoke
```

Smoke runs all 11 trainable/fitted method entries plus the inference variants,
using 64 windows, 2 configurations, 2 seeds, 2 neural epochs and 4 GAN iterations
per phase. The protocol's full budget is unchanged. A smoke pass checks execution,
causality and resume behavior; it does not establish scientific model quality.

## Full run and resume

```sh
python scripts/02_train.py --plant all --device cuda --hours 10 --out results/phase2
```

C1, A3 and their inference layers run before the remaining baselines, so interrupted
sessions prioritize the proposed method. Full local runs require a clean git tree. Exported snapshots instead verify every
included source/config/data hash. Existing runs without identity.json, or with
changed settings/data/code/package versions, are rejected rather than overwritten.
Use a fresh output folder when the experiment changes. Identical interrupted runs
resume by repeating the command. No --resume flag is required.

An epoch checkpoint includes optimizer, best weights, sampling and CPU/CUDA RNG
state. GAN checkpoints additionally track iteration and phase and every optimizer.
The wall-time budget stops after saving a checkpoint; it is not a hard process
kill and may run past the budget while finishing the current epoch or fitting B3.
A partially completed K-dev fit restarts deterministically from its selected model.
A completed run produces COMPLETE_all.json or COMPLETE_hai/qappd.json. A paused
run says PAUSED and must not be reported as complete.

## Kaggle

1. Upload `consistent-decoy-kaggle.zip` as a **private Kaggle Dataset**. It contains
   only the processed training/validation recordings, code and specification.
2. Import `phase2_kaggle.ipynb` as a notebook. Attach that private dataset, enable
   a GPU accelerator, and use **Save Version → Save & Run All**.
3. The notebook finds the archive automatically, verifies its manifest, runs the
   regression checks and an all-method GPU smoke test, then launches full training.
   With two GPUs it runs one plant per GPU; with one GPU it runs both sequentially.
4. Results/checkpoints go under `/kaggle/working/phase2`. Check the completion
   markers and logs before calling training finished. The default budget is 10 hours,
   configurable in the notebook. This is a stop/resume budget, not an ETA.
5. To continue in a later session, save/download the output and attach it to the
   next notebook. Set RESUME_FROM to the attached previous `phase2` directory.
   The notebook copies it to writable storage, then repeats the same run. Keep
   the same code bundle and package versions. Kaggle session storage is not a backup.

The notebook uses Kaggle's preinstalled CUDA PyTorch. Do not install the old
Mac-specific requirements freeze over it. Package versions and device information
are recorded for every run. GPU numerical determinism is not guaranteed across
hardware/framework versions; exact resume equivalence is tested on CPU.

Export again after any accepted change:

```sh
python scripts/02_train.py --export-kaggle dist/consistent-decoy-kaggle.zip
```

## Outputs

Each method/configuration/seed has identity.json, ckpt.pt, model.pt and done.json.
GANs retain their final iterate as prescribed by D29. B4–B7 use K-dev for configuration
selection; other learned methods use validation FIT%. Ties prefer the earlier grid
entry. A3 inherits the C1 choice. B2/B3 are deterministic and do not acquire artificial
seed replicas. `layers/parameters.json` saves normalization, bounds, rates, motor
relations and the training-rule audit; `layers/validation.json` saves variant scores.
B3's parameters are stored separately. Neither K-dev scores nor validation FIT%
should be presented as held-out experimental results.

## Sources

- [Protocol](spec/PROTOCOL.md) and [decisions](spec/decisions.md).
- [TimeGAN authors' code](https://github.com/jsyoon0823/TimeGAN/blob/master/timegan.py).
- [RCGAN paper](https://arxiv.org/abs/1706.02633).
- [NeuralPot accepted manuscript](https://gala.gre.ac.uk/id/eprint/27976/).
- [Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks).

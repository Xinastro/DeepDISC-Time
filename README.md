# DeepDISC-Time

Amortized scene-level inference for blended variable sources in Rubin/LSST.

Extends [DeepDISC](https://github.com/lincc-frameworks/deepdisc) from static
coadds to sequences of native single-visit images, performing detection,
deblending, static/variable decomposition, classification and probabilistic
light-curve inference jointly in one forward pass. Pixels are never coadded and
no difference-imaging template is required.

## What is and is not new here

Template-free multi-epoch static/variable decomposition is **established**. The
scarlet2 time-domain extension ([Ward et al. 2025, *Astronomy and Computing*
51, 100930](https://arxiv.org/abs/2409.15427)) forward models multi-epoch,
multi-band, multi-resolution imaging without reference or difference images,
constrains source fluxes as time-varying or static, and recovers transient light
curves and host morphologies. It has been demonstrated on simulated LSST-like
supernovae, ZTF tidal disruption events, and HSC variable AGN in COSMOS out to
z = 4.

In this codebase scarlet2-TD is the **reference baseline and a supervision
source**, not a strawman. `baselines/scarlet2_td.py` is written to run it well,
because a comparison against a badly configured baseline is worthless.

What this package adds is the part per-scene optimisation structurally cannot
do:

| | scarlet2-TD | DeepDISC-Time |
|---|---|---|
| Detection | requires an input source list | inside the model |
| Inference | per-scene optimisation | amortized, one forward pass |
| Static vs. variable | imposed by the user | predicted, with a probability |
| Per-epoch flux posterior | MAP fit | calibrated flow / MDN |
| Classification | downstream | joint head |

Three experiments follow directly, and they are implemented in
`eval/experiments.py`:

- **A.** Does temporal information improve *detection*? scarlet2 cannot answer
  this because it is handed the answer.
- **B.** The accuracy/throughput frontier. The deliverable is a curve, not a
  winner; per-scene optimisation is expected to win on accuracy in some regimes.
- **C.** The epoch ablation: when does time start to help?

## Install

```bash
git clone <this repo> && cd deepdisc-time
pip install -e ".[dev]"        # core: numpy, scipy, torch, astropy
```

The core runs on CPU with no compiled dependencies, which is deliberate: the
model, the metrics and the whole test suite work without Detectron2, scarlet2,
or the Rubin stack. Those are needed only for the production paths.

```bash
pip install -e ".[deepdisc]"   # Detectron2 training path
pip install -e ".[scarlet2]"   # reference baseline (needs a matching jaxlib)
pip install -e ".[sim]"        # GalSim injection
```

On DeltaAI, install the CUDA 12 `jaxlib` wheel matching the GH200 driver stack
*before* scarlet2, and build Detectron2 from source against the same torch
build. Mixing CUDA minor versions between the JAX and torch stacks is the most
common way this environment breaks.

## Quickstart

```python
from deepdisc_time.data.injection import InjectionConfig, make_synthetic_sequence
from deepdisc_time.data.loaders import sequences_to_batch
from deepdisc_time.modeling.meta_arch import TemporalSceneModel

seqs = [make_synthetic_sequence(InjectionConfig(seed=i, nucleus_to_host=0.2))
        for i in range(4)]
model = TemporalSceneModel()

model.train()
losses = model(sequences_to_batch(seqs))     # detection, lightcurve, variability, rendering

model.eval()
solutions = model.predict_scene(sequences_to_batch(seqs), [s.scene_id for s in seqs])
```

Compare against the baselines with the same metrics:

```python
from deepdisc_time.baselines import DIAForcedPhotometry, StaticThenForcedPhotometry
from deepdisc_time.eval import evaluate

for backend in (StaticThenForcedPhotometry(), DIAForcedPhotometry()):
    print(backend.name, evaluate(backend.fit_many(seqs), seqs))
```

## Layout

```
src/deepdisc_time/
  core/schema.py        Sequence, EpochMeta, SceneSolution -- the shared contract
  data/
    drw.py              exact-update DRW sampler, structure function, Kalman likelihood
    injection.py        variable sources into real frames; synthetic fallback
    loaders.py          ragged-batch collation, epoch samplers, DeepDISC mapper
    stamps.py           on-the-fly cutouts (Butler + FITS backends)
  modeling/
    film.py             per-epoch conditioning on PSF, depth, noise, zeropoint
    temporal.py         continuous-time encoding, temporal transformer, feature pooling
    flows.py            conditional flow and MDN flux posteriors
    rendering.py        differentiable multi-epoch renderer + physics-informed loss
    encoder.py          shared epoch-conditioned encoder
    heads.py            temporal ROI heads (+ Detectron2 CascadeROIHeads subclass)
    meta_arch.py        TemporalSceneModel (pure torch) + GeneralizedRCNNTime
  baselines/            scarlet2-TD adapter, DIA, forced photometry
  eval/                 method-blind metrics; Experiments A, B, C
  training/             loop, masked-epoch SSL, warm start from static DeepDISC
  configs/              LazyConfig for the DP2 DDFs
```

Everything that models a scene emits a `SceneSolution`, and nothing in `eval/`
knows which method produced one. That is what makes Experiment B a fair
comparison rather than a rigged one.

## Design notes worth knowing before you edit

**Epochs are not channels.** Static DeepDISC feeds a `(H, W, C)` band stack.
Rubin visits are single-filter, so a sequence is `T` single-band frames whose
filter identity is metadata. Stacking epochs as channels would freeze the
sequence length into the architecture and discard the irregular cadence. The
encoder therefore takes one channel and folds the epoch axis into the batch.
This changes the backbone stem arity, so `warm_start_from_static` band-averages
pretrained stem weights rather than slicing them, and reports what loaded versus
what was skipped.

**Nothing is materialised.** Per-scene epoch stacks reach petabyte scale for the
DDFs. Visit imagery is stored once and stamps are cut at training time, which
shifts cost onto IO and CPU and is why the two-tier Delta/DeltaAI shared
filesystem matters.

**Padding is everywhere.** ELAISS1 (539 visits) and ECDFS (159 visits) will
share a batch. Every component carries a `pad_mask`; padded PSFs are unit
impulses rather than zeros, because a zero kernel NaNs out normalised paths.

**Detectron2 operates in absolute pixel units.** Anchor sizes, FPN stride
geometry and RoIAlign resolution are all pixel-scale dependent. The config
anchors assume DP2's 0.2"/pixel.

## Limitations

- `TemporalSceneModel` uses an anchor-free centre heatmap, not an RPN. It is for
  development, testing and timing; production detection quality should come from
  the Detectron2 path.
- `DIAForcedPhotometry` uses a Gaussian matching kernel, not a spatially varying
  Alard-Lupton kernel. Adequate for controlled comparison on small stamps with
  known PSFs; **not** a substitute for Rubin `ap_pipe` outputs. Compare against
  the DP2 DIA source table for anything publishable.
- The scarlet2 adapter is written against the current API but is not exercised
  in CI. Pin the scarlet2 version; the source attribute layout has moved between
  releases.
- Recovered DRW `tau` is only weakly constrained when `tau` exceeds the
  baseline. DP2 DDF baselines are 55-84 days against AGN timescales of hundreds
  of days, so treat `tau` as a lower bound in that regime, not a measurement.
- `ButlerStampService.build_index` is written against DP2 dataset type names;
  check `butler.registry` rather than assuming, since these moved between DP1
  and DP2.

## Tests

```bash
pytest tests -q
```

The suite is weighted towards things that would silently corrupt a science
result rather than raise: PSF convolution conventions (a flipped kernel shifts
sources by up to a PSF width, which is indistinguishable from a real
transient-host offset), flux conservation, density normalisation, padding
handling, and whether the calibration metrics actually detect an overconfident
posterior.

One test asserts that naive forced photometry shows a *positive* correlation
between recovered flux and seeing. That is host light leaking under a widening
PSF, i.e. the systematic this project exists to measure. If it ever stops
appearing, the baseline has been accidentally made smarter than the production
pipeline it stands in for.

## Citation

If you use this code, please cite the DeepDISC papers (Merz et al. 2023, MNRAS
526, 1122; Merz et al. 2025, OJAp 8, 40) and, when comparing against or using
the reference baseline, Ward et al. 2025 (A&C 51, 100930).

## v0.2.0: per-band static amplitude and Experiment D

### What changed and why

Until v0.1.0 the static component of a scene was a single morphology map reused
for every epoch regardless of filter. That is the same assumption `scarlet2`
makes, and it is wrong in a specific, measurable way: real galaxies have colour
gradients, so a host's apparent morphology genuinely differs between *u* and *y*.

When a band-independent morphology is fitted to such a host, the unexplained
structure has to be absorbed somewhere, and the nearest free parameter is the
**variable point-source amplitude** in the bands where the mismatch is worst. The
failure mode is therefore not a poor host model. It is a *spurious,
band-correlated light curve on a source that does not vary* — an artefact in
precisely the quantity this package exists to measure.

### How big is it

`tests/test_deepdisc_time.py::test_band_independent_fit_manufactures_variability`
asserts the effect exists. Scanning it (closed-form two-component least squares,
fixed mid-band template, 18 epochs over `ugrizy`):

| nucleus/host | grad 0.10 | 0.25 | 0.50 | 0.75 | 1.00 |
|---:|---:|---:|---:|---:|---:|
| 1.00 | 0.08 | 0.20 | 0.41 | 0.61 | 0.71 |
| 0.50 | 0.16 | 0.41 | 0.82 | 1.22 | 1.39 |
| 0.20 | 0.41 | 1.02 | 2.04 | 3.06 | 3.32 |
| 0.10 | 0.82 | 2.04 | 4.08 | 6.11 | 6.17 |
| 0.05 | 1.63 | 4.08 | 8.15 | 12.23 | 10.82 |

Entries are the peak-to-peak spread of the recovered per-band nuclear flux,
divided by its mean. A value of 1.0 means the manufactured variability is as
large as the source's entire flux. Columns are bulge-minus-disc colour in
magnitudes across `ugrizy`.

Two things to read from it. The artefact scales as the inverse of the
nucleus-to-host contrast, because the residual is a roughly fixed fraction of
*host* flux while the denominator shrinks. And it is therefore worst in exactly
the regime this project cares most about: faint nuclei in bright, structured
hosts, i.e. dwarf AGN and diffuse hosts.

**These are upper bounds on the model-form effect, not predictions for a trained
model.** The fit above uses a *fixed* mid-band template. A fitter free to choose
its own average morphology, as both this package and `scarlet2` are, will find a
better compromise and a smaller residual. How much smaller is an empirical
question and is what `experiment_d_colour_gradient` measures.

### The fix, and what it does not fix

`TemporalROIHead` now predicts a per-band log-amplitude for the static component
(`static_sed`), and `build_static_per_epoch` paints one canvas per band and
indexes it by epoch filter. Cost is independent of sequence length: six pastes,
not `T`.

This captures a pure **colour offset** — a change in the bulge-to-disc *ratio* of
brightness. It does not capture a change of *shape* with wavelength, which is
what a real colour gradient also produces. Band-conditioned morphology is the
open research question and is deliberately not attempted here.

### Backward compatibility

- `predict_sed=False` reproduces v0.1.0 behaviour exactly, and is the control arm
  of Experiment D.
- The SED decoder is zero-initialised, so an untrained model starts at unit
  amplitude in every band and reproduces v0.1.0 exactly.
- `render_scene` accepts `(B, H, W)` as before, or `(B, T, H, W)` for the
  per-epoch form. Both are tested to agree when the latter is a broadcast.
- Old checkpoints load into `predict_sed=False` models without modification.

### Why a ridge penalty rather than a normalisation

`temporal_sed_reg` is an L2 penalty on the log SED (`w["sed"]`, default 1e-2). It
breaks the exact degeneracy between a global rescaling of the morphology patch
and a uniform shift of the log SED, and it encodes the prior that the static
component is grey unless the data say otherwise.

A cross-band normalisation would have been the obvious alternative and is wrong
here: DDF coverage in `u` and `y` is far thinner than in `griz`, and normalising
over all six bands lets an unconstrained band shift the amplitude of the
constrained ones.

### New in this release

- `modeling.meta_arch.build_static_per_epoch`
- `modeling.heads.TemporalROIHead(predict_sed=...)`, `TemporalHeadOutput.static_sed`
- `modeling.meta_arch.TemporalSceneModel(predict_sed=...)`
- `data.injection.colour_gradient_host`, `make_colour_gradient_sequence`
- `eval.metrics.spurious_variability_metrics`
- `eval.experiments.experiment_d_colour_gradient`
- `SourceSolution.static_flux` is now populated at inference (previously left empty)
- `scripts/train_time.py --no-predict-sed` (the Experiment D control arm)
- `scripts/run_experiments.py --experiment d`, with `--nosed-model`, `--gradients`, `--n-scenes`

### Two bugs fixed

1. `build_static_per_epoch` originally clamped `band_idx` in place. That is the
   same tensor the encoder hands to its band embedding, and an embedding saves
   its index tensor for backward, so the version-counter bump made autograd
   refuse the backward pass. Caught by
   `test_model_trains_and_predicts_end_to_end`. Now out-of-place.

2. `scripts/run_experiments.py::load_model` read `pool_mode` back from the
   checkpoint but not `predict_sed`, so loading a `--no-predict-sed` control arm
   failed on missing keys. This is the same class of bug as the original
   Experiment A architecture mismatch, and it failed loudly for the same reason:
   the loader has no `strict=False` escape hatch. Every architecture-shaping flag
   is now read back from the checkpoint, and the arm's configuration is printed
   when it loads.

### Run count

The test suite is 55 tests, up from 47.

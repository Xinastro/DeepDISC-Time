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

## v0.3.0: external code review, acted on

An independent review of v0.2.0 as *research software* found four blocking issues
and several that would have compromised headline results. This release addresses
them. The two most serious were both cases of code whose comments described the
opposite of what it did, which is the worst kind: the tests passed and the
numbers looked plausible.

### P0, fixed and tested

**1. Masked-epoch pretraining was supervising the wrong epochs, two ways.**

The reviewer found the first: `_mask_batch` set `pad_mask = pad | held_out`, and
`RenderingLoss` zeroes the weight of masked epochs. So the held-out epochs got
*zero* loss weight while the visible ones kept theirs. The objective was
"reconstruct what you were shown". The comment claimed the opposite.

Tracing it exposed a second, deeper leak that the loss fix alone would have
hidden. `TemporalTransformer` excludes masked epochs via `src_key_padding_mask`,
which stops *other* positions attending to them, but each position's own token
still flows down the residual stream, and that token is built from the held-out
epoch's pixels. The flux head reads exactly that position. So even with the loss
mask corrected, the task was solvable by the identity.

Both are fixed. There are now three masks with three jobs, never merged:

| mask | meaning | consumer |
|---|---|---|
| `pad_mask` | epoch does not exist (ragged batch) | encoder, attention, loss |
| `ssl_mask` | epoch exists but is hidden this step | encoder, attention, token substitution |
| `render_target_mask` | epochs the loss is evaluated on | loss only |

Hidden epochs have their ROI tokens replaced by a learned
`epoch_mask_token`, as BERT and MAE do. What the model is still told about a
hidden epoch is its time, filter, PSF and depth, which is what makes the task
"predict what this scene looked like under these conditions" rather than "guess".

A subtlety worth recording: the encoder and the rendering loss need *different*
pad semantics. The encoder asks "may I look at this epoch?" and a hidden epoch is
off limits. The loss asks "are there real pixels here?" and a hidden epoch has
them. Passing the encoder mask to the loss zeroes exactly the epochs
`target_mask` selects, giving a loss of identically zero. The first version of
this fix did that, and `test_masked_pretraining_loss_responds_to_held_out_epochs_only`
caught it on the first run.

Masked pretraining now trains: 0.697 → 0.513 over 40 CPU iterations.

**2. Padded source slots were supervised as real sources.** `sequences_to_batch`
pads the source axis to the batch maximum with no mask, so the variability BCE,
the classifier and the renderer all treated empty slots as genuine non-variable
zero-flux objects at `(0, 0)`. With heterogeneous source density the resulting
bias depends on batch composition, which is the worst kind: it moves when you
reshuffle the data. There is now a `source_mask`, propagated to the variability
loss, the classifier (via `ignore_index`), the SED regulariser, and the rendering
term, where padded slots render no flux. It is kept separate from `lc_mask`
because epoch-level and source-level missingness are different mechanisms.

**3. The Detectron2 path trained a different model from the prototype.** The
mapper never attached `gt_lightcurve`, `gt_lightcurve_mask` or `gt_variable` to
the `Instances` that `TemporalCascadeROIHeads` reads, so the production head
silently received `None` and trained neither the flux likelihood nor the
variability classifier. And `GeneralizedRCNNTime` never computed a rendering
term at all, so the production model lacked the physics-informed constraint that
is the only part of the objective supervised by pixels rather than labels, and
therefore the only part that can exceed a scarlet2-derived teacher. Both are now
implemented. The mapper raises rather than guessing if annotation and truth counts
disagree, because a silent misalignment would attach one source's light curve to
another's box and no metric would reveal it.

**4. Production batching could not handle ragged sequences.**
`GeneralizedRCNNTime.preprocess` called `torch.stack` on per-scene tensors, which
raises on a batch mixing a 32-epoch and a 19-epoch scene. `EpochSampler` caps
length but does not pad. Padding now happens in `preprocess`, which is where it
belongs: the right length is the batch maximum, which a mapper working one scene
at a time cannot know. Padded PSFs are unit impulses, not zeros.

> **These Detectron2 fixes have never been executed.** No Detectron2 build was
> available. `tests/test_detectron2_integration.py` asserts the full contract
> (all eight losses on a ragged two-scene batch, finite, with gradients) and
> skips without Detectron2. Making it pass is the first production milestone. If
> it fails, fix the production path, not the test: the pure-torch model is the
> reference, and the two paths must train the same scientific model.

### P1, addressed

**The mean-pooling arm is not a coadd baseline.** Experiment A described
`pool_mode="mean"` as "roughly static/coadd". It is not: each epoch still passes
through the CNN separately and is FiLM-conditioned on its own instrumental state
before averaging, and `mean[f(I)] != f(mean[I])`. Experiment A now takes three
arms, and reports both increments separately:

| increment | isolates |
|---|---|
| `delta_static_minus_coadd` | the value of per-epoch conditioning |
| `delta_temporal_minus_static` | the value of temporal structure proper |

`CoaddDetectionBackend` does a true inverse-variance coadd. Its default peak
finder is a placeholder and says so; supply static DeepDISC for publication,
because a weak control that flatters the temporal model is worse than none.
Calling `experiment_a_detection` without the coadd arm now raises a warning.

**GPU timing was a science-result bug.** CUDA kernels are asynchronous, so
`perf_counter` around a forward pass measures enqueue time, and the error is
larger for the faster method — the direction that flatters us. New
`eval/timing.py` enforces device synchronisation (torch and JAX), discarded
warm-up, repeats with **median and IQR** rather than mean, and records the device,
batch size, epoch count and whether transfer is included alongside every number.
`--experiment b` now also writes `experiment_b_timing.csv`; quote that, not the
wall clock inside `SceneSolution`.

**Calibration claims exceeded what was exported.** The flow can represent
bimodality — the test suite demonstrates it — but `SceneSolution` carried only
mean and sigma, so PIT was computed against a Gaussian reconstruction and
measured the reconstruction. `FluxPosteriorHead.quantiles()` inverts the CDF by
bisection, inference exports a 49-level quantile grid, and `pit_values` uses it
when present. Falls back to samples, then to the Gaussian for baselines that
offer nothing else; the docstring says which path was taken.

**scarlet2 is pinned, with a test.** `requirements-scarlet2.txt` holds the pin
(placeholders, clearly marked) plus the CUDA 13 versus JAX CUDA 12 question.
`tests/test_scarlet2_integration.py` asserts that a bright isolated variable
source is recovered and that every solution carries enough metadata to reproduce
it. Also unexecuted. Reproduce a published Ward et al. (2025) result before
trusting any Experiment B number.

### P2, addressed

**Model-error floor.** `RenderingLoss(sigma_model_frac=...)` adds
`(frac * model)^2` in quadrature to the pixel variance. The variance plane
describes photon and read noise and nothing about background residuals, PSF-model
error or host-model inadequacy; without a floor the rendering term is a confident
teacher of the wrong decomposition in the brightest pixels. Default 0, so nothing
changes silently. Sweep `--sigma-model-frac` and report the sensitivity.

**Optimal-assignment matching.** `match_sources(method="hungarian")` solves the
global minimum-cost assignment. Report both rules in severe blends: a conclusion
that survives only one is a conclusion about the matcher.

### Deferred, deliberately

Real colour-gradient morphology (a learned basis `M_b = M_0 + sum_k a_bk M_k`)
and the flux parameterisation study are not in this release. The reviewer
suggested the first as a second-year problem conditional on Experiment D showing
the systematic matters, which is the right order. The second needs benchmarking
rather than assertion.

### Test count

68 tests pass, up from 55. Two further modules holding 5 tests
(`test_detectron2_integration.py`, `test_scarlet2_integration.py`) skip without
Detectron2 or scarlet2 and have never run anywhere. Run them on Delta or DeltaAI
inside the `deepdiscastro` environment; they are the gate on whether the
production path and the reference baseline are real.

## v0.4.0: second review, and one silent bug it led to

The v0.3.0 review closed every conceptual blocker in the pure-PyTorch path and
shifted the priorities from implementation correctness to **scientific validity**:
leakage, baseline hierarchy, supervision provenance, and whether objectives teach
what their names claim. This release is that work. Tracing one of the reviewer's
questions also turned up a bug that had been silently disabling half the
rendering objective.

### The bug the review surfaced

The reviewer asked whether the production rendering gradient can reach source
positions, and suggested the diagnostic: zero the other losses, backpropagate
rendering alone, see which parameter groups light up. Running it on the
**prototype** gave this:

```
roi_head.flux_head            ----    0.0000e+00   0/12
```

`LogisticMixtureFlowHead.sample` inverts the CDF by bisection, and the method
carried `@torch.no_grad()`. `mean_std` is built on sampling, and the rendering
term takes its per-epoch amplitudes from `mean_std`. So with the **default** flux
head, the rendering loss had no gradient path to the flux head at all. It shaped
the static morphology and the SED and left the light curve untouched, while the
proposal said the decomposition was constrained to reproduce every epoch. Half of
it was. The MDN head was unaffected, because its sampling is reparameterised.

Fixed with implicit differentiation: bisect under `no_grad` to find the root,
then take one Newton step at the detached root. With `F(x, θ) = q`,

    x = x̂ − (F(x̂, θ) − q) / f(x̂, θ)

The numerator is zero to bisection precision so the *value* is unchanged, while
`∂x/∂θ = −(∂F/∂θ)/f` is exactly the derivative of the true inverse. Cost is one
extra CDF and one density evaluation. `flux_head` now receives the largest
gradient of any group from the rendering term, which is what you would expect
given that the rendering residual is most directly sensitive to per-epoch flux.

The diagnostic is now `eval/gradients.py` and a test, with `must_reach` **and**
`must_not_reach` groups. Pinning what an objective does *not* train is the more
useful half: the rendering loss must not train the detector when positions come
from truth, and the contract makes that a statement rather than an assumption.

### Leakage-proof splits: `data/splits.py`

The reviewer's strongest new recommendation. Scenes here are not independent
samples: many stamps come from the same visits, and injection puts many light
curves into the same real host. A random split leaks through at least six
channels, and a leak that helps the temporal arm more than the static arm
manufactures exactly the result the project is looking for.

Splits are assigned per **group**, deterministic given a seed, serialisable, and
**auditable**. `audit_split` reports each channel rather than asserting the split
is fine, because the strategies close different channels:

| strategy | group overlap | visit overlap | notes |
|---|---|---|---|
| `by_field` | 0 | 0% | strongest; use for a headline result |
| `by_sky_block` | 0 | up to 100% | closes same-galaxy, **not** visits |
| `by_night` | 0 | 0% | does not close same-galaxy |
| `by_base_scene` | 0 | varies | minimum for injected data |

**Composition is not monotonically stronger, and this caught my own mistake.**
`compose_groupers` makes groups *finer*. A finer grouping keeps its own guarantee
that no group straddles the split, while permitting units a coarser grouper held
together to be separated. Measured on a three-field, eight-host,
three-injection set:

| grouper | group overlap | visit overlap |
|---|---|---|
| `by_field` | 0 | **0%** |
| `by_field + by_base_scene` | 0 | **33%** |

`by_field` keeps a whole field and all its visits on one side; composing it with
`by_base_scene` makes the group a `(field, host)` pair, so hosts from one field
land on both sides and the visits are shared again. My first CLI default was
`field+base` on the assumption that composition was stronger. The audit said
otherwise. The default is now `field`, the module docstring explains the trap, and
`test_composing_groupers_can_weaken_a_coarse_guarantee` pins it.

`scripts/make_split.py` builds, audits and freezes a split, printing a digest to
quote in logs and captions. `--require-clean` makes a pipeline refuse to proceed
on a leaky split. `train_time.py` and `run_experiments.py` both take `--split` and
**warn loudly without one**, because for injected data no split means the test
hosts were seen.

### B1 and B2 are now different experiments

The framework trains on scarlet2-TD solutions and then compares itself against
scarlet2-TD. Both are legitimate; reported as one number they are misleading.

- `experiment_b1_distillation` asks whether an amortized model can reproduce
  scarlet2's solutions much faster. Training on scarlet2 labels is fair here.
  The reference is the teacher's output.
- `experiment_b2_truth_accuracy` asks which method recovers **truth**. The
  reference is injection truth, and a model distilled from scarlet2 in the same
  domain cannot answer it: its agreement with scarlet2 is a property of its
  training.

`eval/provenance.py` records where a model's labels came from, and
`check_b2_eligibility` is the guard. The eligibility note is attached to every
row of the returned frame, so the caveat travels with the numbers into whatever
plot is made from them. B2 refuses outright on scenes without truth.

### Experiment A's hierarchy is explicit

Arms now carry letters, and both increments are reported with their meaning:

| arm | |
|---|---|
| A | coadded pixels → static detector (conventional) |
| B | per-epoch pixels → shared encoder → mean feature pooling (time-blind) |
| C | per-epoch pixels → shared encoder → temporal attention + variance |

`delta_B_minus_A` is the value of preserving individual exposures.
`delta_C_minus_B` is the value of modelling temporal structure. `delta_C_minus_A`
is the total, and attributing all of it to temporal modelling is the error the
letters exist to prevent.

### Experiment D now uses realistic cadence, and it matters more than expected

The reviewer argued Experiment D is no longer optional, because under irregular
filter cadence a band-dependent host morphology can be read as variability. That
is right, and the round-robin filter cycle every generator used was the cadence
*least* able to detect it.

`ddf_filter_sequence` produces runs within a filter and uneven band coverage, as
the DDFs are actually observed. Measured lag-1 autocorrelation of the spurious
light curve, gradient 0.5, contrast 0.2:

| cadence | lag-1 autocorrelation |
|---|---|
| round-robin | +0.22 |
| realistic DDF | **+0.72** |

A damped random walk is a strongly positively correlated series. Under
round-robin the artefact alternates and nobody would mistake it for variability;
under realistic cadence it becomes a slow correlated drift that a variability
search would call signal. A control built on round-robin would have
substantially understated the danger. `spurious_lag1_autocorr` is now a reported
metric.

### Also in this release

- **Peak GPU memory** in the timing record, reset after warm-up. Memory scaling
  with epoch count is half the ACCESS hardware argument and was an estimate.
- **Model-error floor** (from v0.3.0) and `--sigma-model-frac` retained; see the
  v0.3.0 notes.
- **Environment manifests**: `capture_environment` and `RunManifest` freeze
  versions, CUDA, GPU, driver and git commit next to every result.
- Synthetic sequences now get **unique visit ids** rather than `0..T` in every
  scene. Previously every synthetic split reported 100% visit overlap on data
  where visits were not shared, and a warning that always fires is one people
  learn to ignore.

### One claim removed

"The pixels are the time series" is gone from the ACCESS proposal. The model runs
pixels through a CNN and FPN into ROI tokens before any temporal inference, so
the temporal model learns from image-derived latents, not raw pixel sequences.
The accurate and still-compelling phrasing is: **inference begins from the
multi-epoch pixels rather than from pre-extracted catalogue light curves.**

### Still open, and still the top priority

Unchanged in substance from v0.3.0: the two **DeepDISC-Time integration paths**
have never run end-to-end. Detectron2 itself is compiled and imports successfully
on Delta and DeltaAI, but `tests/test_detectron2_integration.py` has never executed
the DeepDISC-Time -> Detectron2 production wiring. scarlet2-TD is not yet installed
or validated, so `tests/test_scarlet2_integration.py` has never executed either.
These are hard gates before publishing from the production path or producing the
scarlet2 accuracy-throughput comparison; they are deliberately not gates on the
pure-torch fall experiments.

### Test count

80 tests pass, up from 68. Two further modules holding 5 tests skip without
Detectron2 or scarlet2.

## v0.5.0: no new capability, by design

The third review's headline recommendation was **freeze v0.4 and run the
experiment**, with an explicit list of the only changes worth making before the
first results: bugs revealed by execution, scientific-invariance tests,
data-ingestion fixes, split and audit fixes, and instrumentation. Its closing
argument is worth quoting, because it is the thing most likely to be ignored:

> The main risk is no longer that the idea or code is underdeveloped. The risk is
> continuing to develop the machinery instead of doing the experiment.

v0.5.0 adds **no architecture, no new loss, no new model capability**. Everything
below is a test, a metric, a guard or a document. This was the intended freeze point before the final documentation/reproducibility review.

## v0.6.0: freeze reconciliation

v0.6.0 remains a **no-new-capability release**: no architecture, loss, or model
feature changes. It reconciles the executable package with the fall execution plan
and ACCESS proposal after the final review. The release makes four points explicit:
(1) the Detectron2 installation is verified but the DeepDISC-Time -> Detectron2
integration path has never executed; (2) scarlet2-TD is not yet installed/validated;
(3) the 3.1x controlled measurement establishes representation-level separability,
not trained-detector performance; and (4) the student semester and the 12-month
ACCESS project have different critical paths for scarlet2-TD. Tag this release
`v0.6.0-fall2026` and run the experiment rather than adding model machinery.

### The reviewer's highest-priority question, traced and answered

> At what stage can temporal information create a detection that static spatial
> processing did not propose?

Traced through the code. Each visit goes through the backbone **independently**,
then `TemporalFeaturePool` aggregates, *then* `CenterHeatmapHead` runs. So
detection operates on a single temporally pooled feature map. **There is no
per-epoch proposal generation and no union of per-epoch candidates**, which rules
out the mechanism most people assume: temporal evidence cannot rescue a source by
accumulating independent per-epoch detections, because per-epoch detections are
never formed.

The mechanism that does exist is the across-epoch standard-deviation channel in
`attn_var` pooling. Measured on two scenes with **exactly matched mean flux and
identical noise**, differing only in variability:

| pooling | separation of variable from constant, at the source |
|---|---:|
| `mean` | 0.042 |
| `attn_var` | 0.129 |

A factor of 3.1 in **representation-level separation at the source location**. This
establishes a plausible mechanism, not a detection-performance result; whether a
trained detector exploits it is decided by `experiment_a_sub_threshold`. The `mean`
residual is non-zero and should be: averaging
nonlinear features is not extracting features from an average, which is exactly
why the mean-pooling arm is not a coadd baseline.

So the claim is expressible, and narrower than it was being stated:

- **Mechanistically supportable:** exposing across-epoch variance gives the detector
  a stronger representation-level cue for *coherently variable* sources. A trained-model
  detection claim is supportable only if `experiment_a_sub_threshold` confirms it.
- **Not supportable:** "temporal information improves detection" unqualified.
  A single-epoch transient is diluted by `1/T` under mean pooling; `max` pooling
  preserves it and `mean` does not. That is a different problem.

`tests/scientific_invariants/test_detection_mechanism.py` pins both the mechanism
and its limit. The behavioural question — does a *trained* model use it? — is
`experiment_a_sub_threshold`, which sweeps mean nuclear flux across the detection
threshold at fixed variability amplitude. If the temporal and time-blind curves
coincide below threshold, the claim must be revised to "improves deblending and
photometry of candidate detections", and only one of those two is novel against
scarlet2-TD.

### `tests/scientific_invariants/`

17 tests in four families, with every rationale written down. A test whose
rationale cannot be stated is one nobody will know how to fix.

**Symmetry.** Permuting epochs *with* their times, filters and PSFs must leave
inference unchanged; permuting pixels *without* their times must change it.
Relabelling filters consistently must be equivariant. Adding padded epochs must
change nothing. All four pass, which also confirms padding does not leak into
attention, pooling normalisation or the rendering weights.

**Calibration and PSF.** A global flux rescaling leaves the features unchanged to
within 0.5%, because `SimpleConvBackbone` applies `GroupNorm` immediately after
its first convolution. That is scale invariance **by construction**, which is
good, and it has a flip side worth knowing: if the pixels are normalised, then
absolute brightness reaches the model *only* through the conditioner. A companion
test confirms that claiming a five-fold worse noise level does change the
representation, so the conditioning route is live. Verified for the test backbone
only; production MViTv2+FPN has never run.

**Null variability.** The controls must be exactly null, and observed the way the
survey observes. Both are asserted, including that the default null cadence
produces runs within a filter rather than alternating.

**Detection mechanism.** As above.

Three behavioural invariants need a trained model and are marked
`needs_training`: uncertainty must grow under worse seeing, variability
probability must stay calibrated near zero on colour-gradient nulls, and the
sub-threshold detection test. They are written now so the measurement is defined
before anyone is invested in a result.

### A reproducibility bug the invariance tests found

Two symmetry tests failed on first run. Diagnosis: `SceneSolution.lightcurve` is
a **Monte Carlo mean** over posterior samples, so it carries a sampling error of
`sigma/sqrt(n)` and moves between runs on identical input. Measured at 8.2 nJy
against a posterior sigma of 97, which is `97/sqrt(128)` exactly.

Not a model bug, and a real problem for a project that freezes split digests and
quotes them in captions. `lightcurve_median`, obtained by bisecting the CDF, is
now exported alongside: measured run-to-run spread 0.0. `lightcurve` keeps its
meaning as the posterior mean, which for a skewed posterior is a different and
sometimes more appropriate quantity. The invariance tests use the median, so they
test the model rather than the sampler.

### Claim guards

The reviewer's sharpest scoping point: with a peak-finder coadd arm, semester
Experiment A **cannot** support "DeepDISC-Time improves detection relative to
static DeepDISC". It supports at most "the temporal pipeline outperforms a simple
coadd peak finder".

`CoaddDetectionBackend` now warns on construction when the default detector is in
use, carries `is_placeholder_detector`, and `experiment_a_detection` writes
`coadd_arm_is_placeholder` and `supportable_claim` into **every row**. The scope
of the claim travels with the numbers instead of depending on someone remembering
a docstring.

### Conditional variable discovery

A nuclear AGN and its host are spatially coincident, so the right semantics are
**one galaxy instance carrying a variable nuclear component**, not two competing
instances. Ordinary object completeness for "the nucleus" conflates not finding
the galaxy with finding it and missing that its centre varies, and only the second
is what this project is about.

`variable_discovery_metrics` conditions on host detection and reports
`var_discovery_rate`, `var_false_positive_rate` and `nuclear_offset_rms`. Methods
that do not predict a variability probability — every baseline, including
scarlet2-TD, where static-versus-variable is imposed rather than inferred — are
**excluded from the rates rather than scored as negatives**. Counting "did not
answer" as "answered no" would flatter this framework by construction.

### Headline split eligibility

`audit_split` now reports `headline_eligible`, true only when whole fields are
held out. The reviewer's position is that train-on-one-DDF, test-on-a-held-out-DDF
is a **requirement** for a publication headline rather than a preference, because
it changes sky, galaxies, neighbours, visits, PSFs, cadence, background and
instrumental realisation at once. Clean and eligible are different bars: a spatial
split inside one field can be clean on every measured channel and still not be
eligible.

My first implementation derived field disjointness from group *keys*, which
passes spuriously for a sky-block grouper whose keys are tiles. It reads the
scenes now, and a test pins the distinction.

### Also

- `--experiment a-sub`, `b1`, `b2` in the runner, with `b1` refusing to run
  without `--with-scarlet2` and saying why.
- `pytest` marker `needs_training`.

### Test count

97 tests pass, up from 80, of which 17 are the new invariants. Five further tests
skip: two integration modules that have never run anywhere, and three
behavioural invariants awaiting a trained checkpoint.

### Unchanged, and still the gate

The Detectron2 and scarlet2 integration tests have never executed. Nothing in
this release changes that, and the reviewer is right that passing them is a hard
gate before any real training or any accuracy-throughput figure.

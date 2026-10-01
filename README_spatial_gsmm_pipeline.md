# Spatial GSMM consolidated pipeline

This package implements the current core architecture we have agreed on:

\[
G,Y \rightarrow X_{met} \rightarrow X^\star \rightarrow X^{(m)}
\rightarrow P,D,E^{out},E^{in}
\rightarrow \rho,\alpha,F
\rightarrow M^{pred}.
\]

## What is implemented

- Real Compass reaction-score ingestion.
- Compass score orientation so larger means greater predicted reaction activity.
- Calibration-fitted common reaction-wise scaling to obtain `Xstar`.
- Model A: direct `Xstar`.
- Model B: simple spatial smoothing with `lambda` and `tau_sm`.
- Model C: spatial CVAE conditioned on coordinates.
- Spatial receiver graph: kNN + maximum-distance cutoff.
- Receiver proximity predictor based on standardized `-distance^2`; no receiver-model `tau_K`.
- Direct computation of `P`, `D`, `Eout`, and `Ein` from `X^(m)`; no barred re-normalized PDE variables.
- Hierarchical Bayesian release model.
- Augmented receiver allocation with an explicit unassigned state.
- `F`, retained production, incoming exchange, and
  `Mpred = Rret + incoming`.
- Paired MSI calibration using standardized log spatial profiles.
- Held-out posterior-predictive uncertainty propagated through `rho`, `alpha`, `F`, and `Mpred`.
- Per-metabolite held-out Spearman validation.
- Hook for upstream uncertainty through multiple `PreparedTissue` variants.

## What is intentionally not implemented here

- Compass itself is not reimplemented. Run official Compass separately with Gurobi and supply the resulting reaction matrix.
- `W` is not part of the core pipeline.
- Population-level sender/receiver enrichment, permutation testing, receiver entropy, and similar downstream analyses are omitted for now.
- The current reference runner uses Pyro SVI for scalability. Full HMC for a whole Visium slide would usually be computationally impractical because `rho` and `alpha` are high-dimensional latent variables.

## Installation

```bash
pip install numpy pandas scipy scikit-learn torch pyro-ppl
```

For Model C, PyTorch is required. For Bayesian calibration, `pyro-ppl` is required.

## Input files

### `stoichiometry_long.csv`

```text
metabolite_id,reaction_id,coefficient
M_glc,R1,-1
M_pyr,R1,2
...
```

### `reaction_metadata.csv`

```text
reaction_id,is_internal
R1,1
R2,1
EX_glc,0
...
```

### `transport_map.csv`

```text
metabolite_id,reaction_id,direction
M_glc,GLC_IMPORT,import
M_glc,GLC_EXPORT,export
...
```

### Per-tissue Compass matrix

```text
spot_id,R1,R2,R3,...
AAAC...,12.4,8.1,4.2,...
...
```

For raw Compass penalties, use `"score_mode": "penalty"`. The script reverses orientation before common scaling so larger means more predicted activity.

### Coordinates

```text
spot_id,x,y
AAAC...,412.3,875.1
...
```

### MSI matrix

```text
spot_id,M_glc,M_lac,...
AAAC...,1023.1,512.8,...
...
```

### Manifest

```text
tissue_id,split,compass_scores,coordinates,msi
lung_1,calibration,lung1_compass.csv,lung1_coords.csv,lung1_msi.csv
breast_1,calibration,breast1_compass.csv,breast1_coords.csv,breast1_msi.csv
lung_heldout,heldout,lungH_compass.csv,lungH_coords.csv,lungH_msi.csv
```

All spots from a slide/sample must be assigned entirely to calibration or entirely to held-out validation.

## Example run

```bash
python spatial_gsmm_pipeline.py --config example_spatial_gsmm_config.json
```

The default example config uses Model A. Change `upstream_model` to `B` or `C` to perform the corresponding architecture run. The same split, metabolites, stoichiometry, graph definition, Bayesian model, and held-out evaluation should be used across A/B/C.

## Architecture comparison

Run the full pipeline separately for:

- `"upstream_model": "A"`
- `"upstream_model": "B"`
- `"upstream_model": "C"`

Then compare the generated `heldout_msi_spearman.csv` files. The decision about whether to retain the CVAE should be made from held-out MSI prediction, not reaction reconstruction error.

## Main outputs

- `Mpred_mean.csv`
- `Mpred_median.csv`
- `Mpred_ci025.csv`
- `Mpred_ci975.csv`
- `posterior_predictive_draws.npz`
- `heldout_msi_spearman.csv`
- `svi_loss.csv`
- `run_config.json`

The `.npz` archive contains draws for `Mpred`, `Rret`, incoming exchange, `rho`, unassigned release, and edge-level `F`.

## Important statistical interpretation

MSI is used during paired calibration to learn the global/hierarchical release and receiver relationships. `Mpred` and MSI are not assumed to share physical units; standardized log spatial profiles provide a comparable scale for the likelihood.

For final held-out validation, the principal metric is per-metabolite spatial Spearman correlation between `Mpred` and observed MSI, so common absolute units are not required.

## Upstream uncertainty

The reference runner uses one deterministic `X^(m)` by default. The file also includes `prepare_upstream_variants()` and `predict_heldout_from_population()` so bootstrap Compass realizations or CVAE posterior draws can be passed as multiple prepared tissue variants. Posterior draw `b` then uses one upstream realization and propagates it through

\[
P,D,E \rightarrow \rho,\alpha,F \rightarrow M^{pred}.
\]

For final production analysis, upstream uncertainty should also be represented during calibration, e.g. by multiple-imputation fits across upstream realizations or a fully joint model. That is computationally heavier and is deliberately kept separate from the first real-data implementation.

## One important implementation choice

The common reaction-wise scaler is fit on calibration tissues only and then frozen before transforming held-out tissues. This prevents leakage from held-out MSI or held-out reaction distributions into the learned scale.

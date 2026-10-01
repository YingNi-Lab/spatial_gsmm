#!/usr/bin/env python3
"""
Spatial GSMM — consolidated reference pipeline
==============================================

Current core target
-------------------
    G, Y -> X_met -> X* -> X^(m) -> P,D,Eout,Ein
         -> rho, alpha, F -> M_pred

This implementation reflects the current simplified architecture:
- W is NOT required.
- Three upstream reaction representations are supported:
    A) direct scaled metabolic-model output
    B) simple spatial smoothing
    C) spatial CVAE
- P,D,Eout,Ein are computed directly from X^(m); they are not re-normalized.
- The receiver model uses a kNN + distance-cap graph and standardized proximity;
  it does not use a second Gaussian bandwidth.
- A Gaussian bandwidth is used only by Model B spatial smoothing.
- Release is Beta-distributed and receiver allocation is Dirichlet-distributed
  with an explicit unassigned state.
- Paired MSI is used for calibration on a comparable transformed scale.
- Held-out validation uses per-metabolite spatial Spearman correlation.

Important
---------
1) This script does not re-implement Compass. Supply a real Compass reaction
   score matrix (spots x reactions). The Compass/Gurobi run should be performed
   separately using the official Compass installation and an appropriate
   Gurobi license.
2) Compass raw penalties must be re-oriented so that larger values mean greater
   predicted activity. This script supports score_mode='penalty' or 'activity'.
3) The default reaction scaling is calibration-fitted robust min-max scaling.
4) Static MSI validates M_pred, not the direction of an individual F_ijq edge.
5) This is a research reference implementation. For large Visium datasets,
   Pyro SVI is used instead of full HMC because the latent allocation tensor can
   be very large.

Expected global files
---------------------
stoichiometry_long.csv:
    metabolite_id,reaction_id,coefficient

reaction_metadata.csv:
    reaction_id,is_internal
    is_internal should be 1 for reactions included in P/D accounting.

transport_map.csv:
    metabolite_id,reaction_id,direction
    direction in {'import','export'}

Expected per-tissue files
-------------------------
compass_scores.csv:
    spot_id,<reaction_1>,<reaction_2>,...

coordinates.csv:
    spot_id,x,y

optional msi.csv for calibration/validation:
    spot_id,<metabolite_1>,<metabolite_2>,...

The same metabolite ordering is required across calibration tissues in the
first consolidated implementation. Restrict all tissues to a common MSI/model
metabolite set before fitting.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.special import logsumexp, expit
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

EPS = 1e-8


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

def set_seed(seed: int = 20261001) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Basic data containers
# -----------------------------------------------------------------------------

@dataclass
class ReactionModel:
    reaction_ids: List[str]
    metabolite_ids: List[str]
    S: np.ndarray                  # Q x R
    internal_mask: np.ndarray      # R bool
    import_map: np.ndarray         # Q x R, 0/1
    export_map: np.ndarray         # Q x R, 0/1

    @property
    def n_reactions(self) -> int:
        return len(self.reaction_ids)

    @property
    def n_metabolites(self) -> int:
        return len(self.metabolite_ids)


@dataclass
class SpatialGraph:
    neighbors: np.ndarray          # n x k, integer receiver indices
    valid: np.ndarray              # n x k, spatial eligibility
    distances: np.ndarray          # n x k
    z_prox: np.ndarray             # n x k; larger = closer
    k: int
    d_max: float


@dataclass
class PDEState:
    P: np.ndarray                  # n x Q
    D: np.ndarray
    Eout: np.ndarray
    Ein: np.ndarray
    zD: np.ndarray
    zEout: np.ndarray
    zEin: np.ndarray


@dataclass
class TissueInput:
    tissue_id: str
    spot_ids: List[str]
    coords: np.ndarray             # n x 2
    compass_scores: np.ndarray     # n x R
    reaction_ids: List[str]
    msi: Optional[np.ndarray] = None          # n x Q
    msi_metabolite_ids: Optional[List[str]] = None


@dataclass
class PreparedTissue:
    tissue_id: str
    spot_ids: List[str]
    coords: np.ndarray
    Xstar: np.ndarray
    Xmodel: np.ndarray
    pde: PDEState
    graph: SpatialGraph
    msi: Optional[np.ndarray] = None
    msi_z: Optional[np.ndarray] = None
    metabolite_ids: List[str] = field(default_factory=list)


# -----------------------------------------------------------------------------
# I/O and alignment
# -----------------------------------------------------------------------------

def _read_indexed_matrix(path: str | Path, id_col: str) -> Tuple[List[str], List[str], np.ndarray]:
    df = pd.read_csv(path)
    if id_col not in df.columns:
        raise ValueError(f"{path}: expected an '{id_col}' column")
    row_ids = df[id_col].astype(str).tolist()
    value_cols = [c for c in df.columns if c != id_col]
    X = df[value_cols].to_numpy(dtype=float)
    return row_ids, value_cols, X


def load_tissue(
    tissue_id: str,
    compass_path: str | Path,
    coords_path: str | Path,
    msi_path: Optional[str | Path] = None,
) -> TissueInput:
    spot_ids, reaction_ids, scores = _read_indexed_matrix(compass_path, "spot_id")

    cdf = pd.read_csv(coords_path)
    required = {"spot_id", "x", "y"}
    if not required.issubset(cdf.columns):
        raise ValueError(f"{coords_path}: coordinates must contain {sorted(required)}")
    cdf["spot_id"] = cdf["spot_id"].astype(str)
    cdf = cdf.set_index("spot_id").loc[spot_ids]
    coords = cdf[["x", "y"]].to_numpy(dtype=float)

    msi = None
    msi_ids = None
    if msi_path is not None:
        m_spots, msi_ids, msi = _read_indexed_matrix(msi_path, "spot_id")
        mdf = pd.DataFrame(msi, index=m_spots, columns=msi_ids)
        msi = mdf.loc[spot_ids].to_numpy(dtype=float)

    return TissueInput(
        tissue_id=tissue_id,
        spot_ids=spot_ids,
        coords=coords,
        compass_scores=scores,
        reaction_ids=reaction_ids,
        msi=msi,
        msi_metabolite_ids=msi_ids,
    )


def load_reaction_model(
    stoichiometry_path: str | Path,
    reaction_metadata_path: str | Path,
    transport_map_path: str | Path,
    reaction_ids: Sequence[str],
    metabolite_ids: Sequence[str],
) -> ReactionModel:
    reaction_ids = list(map(str, reaction_ids))
    metabolite_ids = list(map(str, metabolite_ids))
    r_index = {r: i for i, r in enumerate(reaction_ids)}
    q_index = {q: i for i, q in enumerate(metabolite_ids)}

    S = np.zeros((len(metabolite_ids), len(reaction_ids)), dtype=float)
    sdf = pd.read_csv(stoichiometry_path)
    for row in sdf.itertuples(index=False):
        q = str(row.metabolite_id)
        r = str(row.reaction_id)
        if q in q_index and r in r_index:
            S[q_index[q], r_index[r]] = float(row.coefficient)

    internal_mask = np.zeros(len(reaction_ids), dtype=bool)
    rdf = pd.read_csv(reaction_metadata_path)
    for row in rdf.itertuples(index=False):
        r = str(row.reaction_id)
        if r in r_index:
            internal_mask[r_index[r]] = bool(int(row.is_internal))

    import_map = np.zeros_like(S)
    export_map = np.zeros_like(S)
    tdf = pd.read_csv(transport_map_path)
    for row in tdf.itertuples(index=False):
        q = str(row.metabolite_id)
        r = str(row.reaction_id)
        direction = str(row.direction).strip().lower()
        if q not in q_index or r not in r_index:
            continue
        if direction == "import":
            import_map[q_index[q], r_index[r]] = 1.0
        elif direction == "export":
            export_map[q_index[q], r_index[r]] = 1.0
        else:
            raise ValueError(f"Unknown transport direction '{direction}'")

    return ReactionModel(
        reaction_ids=reaction_ids,
        metabolite_ids=metabolite_ids,
        S=S,
        internal_mask=internal_mask,
        import_map=import_map,
        export_map=export_map,
    )


# -----------------------------------------------------------------------------
# Compass orientation and common reaction-wise scaling
# -----------------------------------------------------------------------------

def orient_compass_scores(
    scores: np.ndarray,
    score_mode: Literal["penalty", "activity"] = "penalty",
) -> np.ndarray:
    """Return a representation where larger always means greater activity."""
    X = np.asarray(scores, dtype=float)
    if score_mode == "penalty":
        return -X
    if score_mode == "activity":
        return X.copy()
    raise ValueError("score_mode must be 'penalty' or 'activity'")


class RobustReactionScaler:
    """
    Calibration-fitted reaction-wise scaling to [0, 1].

    For each reaction, q_low and q_high are learned from calibration tissues
    only. This scaling is then frozen and applied to held-out tissues.
    """

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lo_: Optional[np.ndarray] = None
        self.hi_: Optional[np.ndarray] = None

    def fit(self, arrays: Sequence[np.ndarray]) -> "RobustReactionScaler":
        X = np.concatenate([np.asarray(a, dtype=float) for a in arrays], axis=0)
        self.lo_ = np.nanquantile(X, self.lower_quantile, axis=0)
        self.hi_ = np.nanquantile(X, self.upper_quantile, axis=0)
        bad = ~np.isfinite(self.lo_) | ~np.isfinite(self.hi_) | ((self.hi_ - self.lo_) < EPS)
        self.lo_[bad] = np.nanmin(X[:, bad], axis=0) if bad.any() else self.lo_[bad]
        self.hi_[bad] = np.nanmax(X[:, bad], axis=0) if bad.any() else self.hi_[bad]
        still_bad = (self.hi_ - self.lo_) < EPS
        self.hi_[still_bad] = self.lo_[still_bad] + 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.lo_ is None or self.hi_ is None:
            raise RuntimeError("Scaler has not been fit")
        Z = (np.asarray(X, dtype=float) - self.lo_) / (self.hi_ - self.lo_ + EPS)
        Z = np.clip(Z, 0.0, 1.0)
        return np.nan_to_num(Z, nan=0.0, posinf=1.0, neginf=0.0)

    def fit_transform(self, arrays: Sequence[np.ndarray]) -> List[np.ndarray]:
        self.fit(arrays)
        return [self.transform(a) for a in arrays]


# -----------------------------------------------------------------------------
# Spatial graph and proximity predictor
# -----------------------------------------------------------------------------

def _masked_standardize(values: np.ndarray, mask: np.ndarray, axis: int = 0) -> np.ndarray:
    """Z-score values using only mask==True entries; masked entries become 0."""
    x = np.asarray(values, dtype=float)
    m = np.asarray(mask, dtype=bool)
    out = np.zeros_like(x, dtype=float)

    if axis != 0:
        raise NotImplementedError("Only axis=0 is used in this pipeline")

    if x.ndim == 1:
        vals = x[m]
        if vals.size == 0:
            return out
        mu = vals.mean()
        sd = vals.std()
        if sd < EPS:
            sd = 1.0
        out[m] = (x[m] - mu) / sd
        return out

    # Column-wise masked standardization
    for j in range(x.shape[1]):
        mj = m[:, j] if m.ndim == x.ndim else m
        vals = x[mj, j]
        if vals.size == 0:
            continue
        mu = vals.mean()
        sd = vals.std()
        if sd < EPS:
            sd = 1.0
        out[mj, j] = (x[mj, j] - mu) / sd
    return out


def build_spatial_graph(coords: np.ndarray, k: int = 6, d_max: Optional[float] = None) -> SpatialGraph:
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]
    if n <= 1:
        raise ValueError("Need at least two spatial units")
    k_eff = min(k, n - 1)

    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean")
    nn.fit(coords)
    distances, indices = nn.kneighbors(coords)
    # Remove self neighbor in first column.
    distances = distances[:, 1:]
    neighbors = indices[:, 1:]

    if d_max is None:
        # Data-adaptive default only for convenience; pre-specify in the paper.
        d_max = float(np.quantile(distances, 0.95))

    valid = distances <= float(d_max)
    # Larger Z_prox means closer. Standardize -d^2 over eligible edges.
    raw = -(distances ** 2)
    vals = raw[valid]
    if vals.size == 0:
        raise ValueError("No eligible spatial edges; increase d_max")
    mu = vals.mean()
    sd = vals.std()
    if sd < EPS:
        sd = 1.0
    z_prox = np.zeros_like(raw)
    z_prox[valid] = (raw[valid] - mu) / sd

    return SpatialGraph(
        neighbors=neighbors.astype(int),
        valid=valid,
        distances=distances,
        z_prox=z_prox,
        k=k_eff,
        d_max=float(d_max),
    )


# -----------------------------------------------------------------------------
# Upstream candidate models A/B/C
# -----------------------------------------------------------------------------

def model_A_direct(Xstar: np.ndarray) -> np.ndarray:
    return np.asarray(Xstar, dtype=float).copy()


def model_B_smooth(
    Xstar: np.ndarray,
    graph: SpatialGraph,
    lam: float = 0.35,
    tau_sm: Optional[float] = None,
) -> np.ndarray:
    if not (0.0 <= lam <= 1.0):
        raise ValueError("lam must be in [0,1]")
    Xstar = np.asarray(Xstar, dtype=float)

    eligible_dist = graph.distances[graph.valid]
    if tau_sm is None:
        tau_sm = float(np.median(eligible_dist))
    tau_sm = max(float(tau_sm), EPS)

    K = np.exp(-(graph.distances ** 2) / (2.0 * tau_sm ** 2)) * graph.valid
    nbr = Xstar[graph.neighbors]                      # n x k x R
    weighted = (K[:, :, None] * nbr).sum(axis=1)
    denom = K.sum(axis=1, keepdims=True)
    nbr_mean = weighted / (denom + EPS)
    no_nbr = denom[:, 0] <= EPS
    nbr_mean[no_nbr] = Xstar[no_nbr]

    Xb = (1.0 - lam) * Xstar + lam * nbr_mean
    return np.clip(Xb, 0.0, None)


class SpatialCVAE:
    """Small spatial CVAE used only as candidate upstream Model C."""

    def __init__(
        self,
        n_reactions: int,
        latent_dim: int = 16,
        hidden_dim: int = 128,
        beta_kl: float = 1e-3,
        device: Optional[str] = None,
    ):
        import torch
        import torch.nn as nn

        self.torch = torch
        self.nn = nn
        self.n_reactions = n_reactions
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.beta_kl = beta_kl
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        class Net(nn.Module):
            def __init__(self, R: int, L: int, H: int):
                super().__init__()
                self.enc = nn.Sequential(
                    nn.Linear(R + 2, H), nn.ReLU(),
                    nn.Linear(H, H), nn.ReLU(),
                )
                self.mu = nn.Linear(H, L)
                self.logvar = nn.Linear(H, L)
                self.dec = nn.Sequential(
                    nn.Linear(L + 2, H), nn.ReLU(),
                    nn.Linear(H, H), nn.ReLU(),
                    nn.Linear(H, R), nn.Sigmoid(),
                )

            def encode(self, x, y):
                h = self.enc(torch.cat([x, y], dim=1))
                return self.mu(h), self.logvar(h)

            def decode(self, z, y):
                return self.dec(torch.cat([z, y], dim=1))

            def forward(self, x, y):
                mu, logvar = self.encode(x, y)
                eps = torch.randn_like(mu)
                z = mu + torch.exp(0.5 * logvar) * eps
                return self.decode(z, y), mu, logvar

        self.net = Net(n_reactions, latent_dim, hidden_dim).to(self.device)
        self.coord_mu_: Optional[np.ndarray] = None
        self.coord_sd_: Optional[np.ndarray] = None

    def _coords(self, coords: np.ndarray, fit: bool = False):
        c = np.asarray(coords, dtype=np.float32)
        if fit:
            self.coord_mu_ = c.mean(axis=0)
            self.coord_sd_ = c.std(axis=0)
            self.coord_sd_[self.coord_sd_ < EPS] = 1.0
        if self.coord_mu_ is None or self.coord_sd_ is None:
            raise RuntimeError("Coordinate scaler is not fit")
        c = (c - self.coord_mu_) / self.coord_sd_
        return self.torch.tensor(c, dtype=self.torch.float32, device=self.device)

    def fit(
        self,
        Xstar: np.ndarray,
        coords: np.ndarray,
        epochs: int = 1200,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        verbose_every: int = 200,
    ) -> "SpatialCVAE":
        torch = self.torch
        x = torch.tensor(np.asarray(Xstar), dtype=torch.float32, device=self.device)
        y = self._coords(coords, fit=True)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)

        self.net.train()
        for epoch in range(1, epochs + 1):
            opt.zero_grad()
            recon, mu, logvar = self.net(x, y)
            rec = torch.mean((recon - x) ** 2)
            kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
            loss = rec + self.beta_kl * kl
            loss.backward()
            opt.step()
            if verbose_every and epoch % verbose_every == 0:
                print(f"CVAE epoch {epoch:5d} loss={float(loss):.6f} rec={float(rec):.6f} kl={float(kl):.6f}")
        return self

    def transform(self, Xstar: np.ndarray, coords: np.ndarray, posterior_mean: bool = True) -> np.ndarray:
        torch = self.torch
        x = torch.tensor(np.asarray(Xstar), dtype=torch.float32, device=self.device)
        y = self._coords(coords, fit=False)
        self.net.eval()
        with torch.no_grad():
            mu, logvar = self.net.encode(x, y)
            if posterior_mean:
                z = mu
            else:
                z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
            out = self.net.decode(z, y)
        return out.cpu().numpy().astype(float)

    def sample_transforms(self, Xstar: np.ndarray, coords: np.ndarray, n_draws: int = 20) -> List[np.ndarray]:
        return [self.transform(Xstar, coords, posterior_mean=False) for _ in range(n_draws)]


# -----------------------------------------------------------------------------
# P, D, Eout, Ein
# -----------------------------------------------------------------------------

def _zscore_log_columns(X: np.ndarray) -> np.ndarray:
    Z = np.log(np.clip(np.asarray(X, dtype=float), 0.0, None) + EPS)
    mu = Z.mean(axis=0, keepdims=True)
    sd = Z.std(axis=0, keepdims=True)
    sd[sd < EPS] = 1.0
    return (Z - mu) / sd


def compute_pde(Xmodel: np.ndarray, reaction_model: ReactionModel) -> PDEState:
    X = np.clip(np.asarray(Xmodel, dtype=float), 0.0, None)
    if X.shape[1] != reaction_model.n_reactions:
        raise ValueError("Xmodel reaction dimension does not match reaction model")

    Splus = np.maximum(reaction_model.S, 0.0)
    Sminus = np.maximum(-reaction_model.S, 0.0)
    mask = reaction_model.internal_mask.astype(float)[None, :]

    # X (n x R), S (Q x R)
    P = (X * mask) @ Splus.T
    D = (X * mask) @ Sminus.T
    Eout = X @ reaction_model.export_map.T
    Ein = X @ reaction_model.import_map.T

    return PDEState(
        P=P,
        D=D,
        Eout=Eout,
        Ein=Ein,
        zD=_zscore_log_columns(D),
        zEout=_zscore_log_columns(Eout),
        zEin=_zscore_log_columns(Ein),
    )


# -----------------------------------------------------------------------------
# MSI transformation and validation
# -----------------------------------------------------------------------------

def standardized_log_profile(M: np.ndarray) -> np.ndarray:
    return _zscore_log_columns(np.clip(np.asarray(M, dtype=float), 0.0, None))


def evaluate_mpred_spearman(
    Mpred: np.ndarray,
    Mobs: np.ndarray,
    metabolite_ids: Sequence[str],
) -> pd.DataFrame:
    if Mpred.shape != Mobs.shape:
        raise ValueError("Mpred and Mobs must have the same shape")
    rows = []
    for q, mid in enumerate(metabolite_ids):
        a = Mpred[:, q]
        b = Mobs[:, q]
        ok = np.isfinite(a) & np.isfinite(b)
        r = np.nan
        if ok.sum() >= 3 and np.nanstd(a[ok]) > 0 and np.nanstd(b[ok]) > 0:
            r = float(spearmanr(a[ok], b[ok]).statistic)
        rows.append({"metabolite_id": mid, "spearman": r, "n_spots": int(ok.sum())})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Tissue preparation
# -----------------------------------------------------------------------------

def align_tissue_reactions(tissue: TissueInput, target_reactions: Sequence[str]) -> np.ndarray:
    idx = {r: i for i, r in enumerate(tissue.reaction_ids)}
    missing = [r for r in target_reactions if r not in idx]
    if missing:
        raise ValueError(f"{tissue.tissue_id}: missing {len(missing)} model reactions, e.g. {missing[:5]}")
    return tissue.compass_scores[:, [idx[r] for r in target_reactions]]


def align_tissue_msi(tissue: TissueInput, target_metabolites: Sequence[str]) -> Optional[np.ndarray]:
    if tissue.msi is None or tissue.msi_metabolite_ids is None:
        return None
    idx = {q: i for i, q in enumerate(tissue.msi_metabolite_ids)}
    missing = [q for q in target_metabolites if q not in idx]
    if missing:
        raise ValueError(f"{tissue.tissue_id}: MSI missing {len(missing)} metabolites, e.g. {missing[:5]}")
    return tissue.msi[:, [idx[q] for q in target_metabolites]]


def prepare_tissue(
    tissue: TissueInput,
    reaction_model: ReactionModel,
    scaler: RobustReactionScaler,
    score_mode: Literal["penalty", "activity"],
    upstream_model: Literal["A", "B", "C"] = "A",
    graph_k: int = 6,
    d_max: Optional[float] = None,
    smooth_lambda: float = 0.35,
    tau_sm: Optional[float] = None,
    cvae: Optional[SpatialCVAE] = None,
) -> PreparedTissue:
    raw_aligned = align_tissue_reactions(tissue, reaction_model.reaction_ids)
    oriented = orient_compass_scores(raw_aligned, score_mode=score_mode)
    Xstar = scaler.transform(oriented)
    graph = build_spatial_graph(tissue.coords, k=graph_k, d_max=d_max)

    if upstream_model == "A":
        Xmodel = model_A_direct(Xstar)
    elif upstream_model == "B":
        Xmodel = model_B_smooth(Xstar, graph, lam=smooth_lambda, tau_sm=tau_sm)
    elif upstream_model == "C":
        if cvae is None:
            raise ValueError("Model C requires a fitted SpatialCVAE")
        Xmodel = cvae.transform(Xstar, tissue.coords, posterior_mean=True)
    else:
        raise ValueError("upstream_model must be A, B, or C")

    pde = compute_pde(Xmodel, reaction_model)
    msi = align_tissue_msi(tissue, reaction_model.metabolite_ids)
    msi_z = standardized_log_profile(msi) if msi is not None else None

    return PreparedTissue(
        tissue_id=tissue.tissue_id,
        spot_ids=tissue.spot_ids,
        coords=tissue.coords,
        Xstar=Xstar,
        Xmodel=Xmodel,
        pde=pde,
        graph=graph,
        msi=msi,
        msi_z=msi_z,
        metabolite_ids=reaction_model.metabolite_ids,
    )


# -----------------------------------------------------------------------------
# Core forward model in NumPy (used for held-out posterior prediction)
# -----------------------------------------------------------------------------

def _masked_zscore_per_metabolite(x: np.ndarray, valid_source_q: np.ndarray) -> np.ndarray:
    """x and valid_source_q are n x Q."""
    out = np.zeros_like(x, dtype=float)
    for q in range(x.shape[1]):
        m = valid_source_q[:, q]
        vals = x[m, q]
        if vals.size == 0:
            continue
        mu = vals.mean()
        sd = vals.std()
        if sd < EPS:
            sd = 1.0
        out[m, q] = (x[m, q] - mu) / sd
    return out


def compute_receiver_pi_numpy(
    tissue: PreparedTissue,
    theta_prox: float,
    theta_D: float,
    theta_I: float,
    gamma0: float,
    gammaS: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return augmented expected allocation pi (n x Q x [k+1]) and q-specific
    receiver mask (n x Q x k). Last category is unassigned.
    """
    pde = tissue.pde
    g = tissue.graph
    n, Q = pde.P.shape
    k = g.k

    # Receiver predictors gathered at j.
    zD_j = pde.zD[g.neighbors]       # n x k x Q
    zEin_j = pde.zEin[g.neighbors]   # n x k x Q
    Ein_j = pde.Ein[g.neighbors]     # n x k x Q

    receiver_mask = g.valid[:, :, None] & (Ein_j > EPS)   # n x k x Q
    receiver_mask_nqk = np.transpose(receiver_mask, (0, 2, 1))

    scores = (
        theta_prox * g.z_prox[:, None, :]
        + theta_D * np.transpose(zD_j, (0, 2, 1))
        + theta_I * np.transpose(zEin_j, (0, 2, 1))
    )
    scores = np.where(receiver_mask_nqk, scores, -1e9)

    any_recv = receiver_mask_nqk.any(axis=2)
    C_raw = logsumexp(scores, axis=2)
    C_raw[~any_recv] = 0.0
    C = _masked_zscore_per_metabolite(C_raw, any_recv)

    s0 = gamma0 - gammaS * C
    aug = np.concatenate([scores, s0[:, :, None]], axis=2)
    aug -= np.max(aug, axis=2, keepdims=True)
    exp_aug = np.exp(np.clip(aug, -700, 50))
    pi = exp_aug / (exp_aug.sum(axis=2, keepdims=True) + EPS)

    # Exact no-receiver rule.
    pi[~any_recv, :-1] = 0.0
    pi[~any_recv, -1] = 1.0
    pi = np.clip(pi, 1e-10, 1.0)
    pi /= pi.sum(axis=2, keepdims=True)
    return pi, receiver_mask_nqk


def forward_draw_numpy(
    tissue: PreparedTissue,
    beta0: float,
    betaE: float,
    betaD: float,
    u_q: np.ndarray,
    kappa_rho: float,
    theta_prox: float,
    thetaD_recv: float,
    thetaI_recv: float,
    gamma0: float,
    gammaS: float,
    tau_alpha: float,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    pde = tissue.pde
    eta = beta0 + betaE * pde.zEout - betaD * pde.zD + u_q[None, :]
    mu_rho = expit(eta)
    mu_rho = np.clip(mu_rho, 1e-6, 1 - 1e-6)

    a = np.maximum(kappa_rho * mu_rho, 1e-4)
    b = np.maximum(kappa_rho * (1.0 - mu_rho), 1e-4)
    rho = rng.beta(a, b)

    pi, recv_mask = compute_receiver_pi_numpy(
        tissue=tissue,
        theta_prox=theta_prox,
        theta_D=thetaD_recv,
        theta_I=thetaI_recv,
        gamma0=gamma0,
        gammaS=gammaS,
    )

    n, Q, kp1 = pi.shape
    alpha = np.empty_like(pi)
    for i in range(n):
        for q in range(Q):
            conc = np.maximum(tau_alpha * pi[i, q], 1e-5)
            alpha[i, q] = rng.dirichlet(conc)

    # Enforce exact structural zero on ineligible cellular receivers by moving
    # their tiny sampled mass to the unassigned category.
    invalid_mass = (alpha[:, :, :-1] * (~recv_mask)).sum(axis=2)
    alpha[:, :, :-1] *= recv_mask
    alpha[:, :, -1] += invalid_mass
    alpha /= alpha.sum(axis=2, keepdims=True)

    F = pde.P[:, :, None] * rho[:, :, None] * alpha[:, :, :-1]
    Rret = pde.P * (1.0 - rho)
    Lunassigned = pde.P * rho * alpha[:, :, -1]

    incoming = np.zeros_like(pde.P)
    for s in range(tissue.graph.k):
        receivers = tissue.graph.neighbors[:, s]
        np.add.at(incoming, receivers, F[:, :, s])

    Mpred = Rret + incoming
    return {
        "rho": rho,
        "mu_rho": mu_rho,
        "pi": pi,
        "alpha": alpha,
        "F": F,
        "Rret": Rret,
        "Lunassigned": Lunassigned,
        "incoming": incoming,
        "Mpred": Mpred,
    }


# -----------------------------------------------------------------------------
# Hierarchical Bayesian calibration using Pyro SVI
# -----------------------------------------------------------------------------

def _require_pyro():
    try:
        import torch
        import pyro
        import pyro.distributions as dist
        from pyro.infer import SVI, Trace_ELBO, Predictive
        from pyro.infer.autoguide import AutoNormal
        from pyro.optim import ClippedAdam
    except ImportError as e:
        raise ImportError(
            "Bayesian calibration requires torch and pyro-ppl. "
            "Install with: pip install torch pyro-ppl"
        ) from e
    return torch, pyro, dist, SVI, Trace_ELBO, Predictive, AutoNormal, ClippedAdam


def _torch_zscore_log(torch, M):
    z = torch.log(torch.clamp(M, min=0.0) + EPS)
    mu = z.mean(dim=0, keepdim=True)
    sd = z.std(dim=0, keepdim=True, unbiased=False)
    sd = torch.where(sd < EPS, torch.ones_like(sd), sd)
    return (z - mu) / sd


def _torch_masked_zscore_cols(torch, x, mask):
    # x and mask: n x Q. Loop Q; Q is typically far smaller than n*k latent edges.
    cols = []
    for q in range(x.shape[1]):
        mq = mask[:, q]
        vals = x[:, q]
        count = mq.sum()
        safe_count = torch.clamp(count, min=1)
        mu = (vals * mq).sum() / safe_count
        var = (((vals - mu) ** 2) * mq).sum() / safe_count
        sd = torch.sqrt(torch.clamp(var, min=EPS))
        z = (vals - mu) / sd
        z = torch.where(mq, z, torch.zeros_like(z))
        cols.append(z)
    return torch.stack(cols, dim=1)


def make_pyro_calibration_model(calibration_tissues: Sequence[PreparedTissue]):
    """Create a Pyro model closure for paired MSI calibration."""
    torch, pyro, dist, *_ = _require_pyro()

    if not calibration_tissues:
        raise ValueError("Need at least one calibration tissue")
    Q = calibration_tissues[0].pde.P.shape[1]
    for t in calibration_tissues:
        if t.msi_z is None:
            raise ValueError(f"Calibration tissue {t.tissue_id} has no MSI")
        if t.pde.P.shape[1] != Q:
            raise ValueError("All calibration tissues must use the same metabolite set")

    def model():
        # Global release intercept and metabolite random intercept.
        beta0 = pyro.sample("beta0", dist.Normal(0.0, 1.0))
        sigma_met = pyro.sample("sigma_met", dist.HalfNormal(0.75))
        u_q = pyro.sample("u_q", dist.Normal(torch.zeros(Q), sigma_met).to_event(1))

        # Hierarchical positive coefficient populations.
        coeff_names = ["betaE", "betaD", "thetaProx", "thetaD", "thetaI"]
        pop = {}
        for name in coeff_names:
            mu = pyro.sample(f"mu_log_{name}", dist.Normal(0.0, 1.0))
            sig = pyro.sample(f"sigma_log_{name}", dist.HalfNormal(0.60))
            pop[name] = (mu, sig)

        # Unassigned-state parameters.
        gamma0 = pyro.sample("gamma0", dist.Normal(0.0, 1.0))
        gammaS = pyro.sample("gammaS", dist.HalfNormal(1.0))

        # Concentration parameters.
        kappa_rho = pyro.sample("kappa_rho", dist.LogNormal(math.log(20.0), 0.7))
        tau_alpha = pyro.sample("tau_alpha", dist.LogNormal(math.log(20.0), 0.7))

        # One observation-noise parameter per metabolite on transformed MSI scale.
        sigma_q = pyro.sample("sigma_q", dist.HalfNormal(torch.ones(Q)).to_event(1))

        for t in calibration_tissues:
            tid = str(t.tissue_id).replace("/", "_").replace(" ", "_")
            n = t.pde.P.shape[0]
            k = t.graph.k

            # Tissue-specific coefficients from hierarchical populations.
            vals = {}
            for name in coeff_names:
                mu, sig = pop[name]
                logv = pyro.sample(f"log_{name}__{tid}", dist.Normal(mu, sig))
                vals[name] = torch.exp(logv)

            P = torch.as_tensor(t.pde.P, dtype=torch.float32)
            zD = torch.as_tensor(t.pde.zD, dtype=torch.float32)
            zEout = torch.as_tensor(t.pde.zEout, dtype=torch.float32)
            zEin = torch.as_tensor(t.pde.zEin, dtype=torch.float32)
            Ein = torch.as_tensor(t.pde.Ein, dtype=torch.float32)
            neighbors = torch.as_tensor(t.graph.neighbors, dtype=torch.long)
            valid = torch.as_tensor(t.graph.valid, dtype=torch.bool)
            zprox = torch.as_tensor(t.graph.z_prox, dtype=torch.float32)
            zobs = torch.as_tensor(t.msi_z, dtype=torch.float32)

            eta = beta0 + vals["betaE"] * zEout - vals["betaD"] * zD + u_q[None, :]
            mu_rho = torch.sigmoid(eta).clamp(1e-5, 1 - 1e-5)
            rho = pyro.sample(
                f"rho__{tid}",
                dist.Beta(kappa_rho * mu_rho, kappa_rho * (1.0 - mu_rho)).to_event(2),
            )

            zD_j = zD[neighbors]       # n x k x Q
            zEin_j = zEin[neighbors]
            Ein_j = Ein[neighbors]
            recv_mask = valid[:, :, None] & (Ein_j > EPS)
            recv_mask_nqk = recv_mask.permute(0, 2, 1)

            scores = (
                vals["thetaProx"] * zprox[:, None, :]
                + vals["thetaD"] * zD_j.permute(0, 2, 1)
                + vals["thetaI"] * zEin_j.permute(0, 2, 1)
            )
            scores = torch.where(recv_mask_nqk, scores, torch.full_like(scores, -1e6))
            any_recv = recv_mask_nqk.any(dim=2)
            Craw = torch.logsumexp(scores, dim=2)
            Craw = torch.where(any_recv, Craw, torch.zeros_like(Craw))
            C = _torch_masked_zscore_cols(torch, Craw, any_recv)

            s0 = gamma0 - gammaS * C
            aug = torch.cat([scores, s0[:, :, None]], dim=2)
            pi = torch.softmax(aug, dim=2)

            # Exact no-receiver expectation: all mass to unassigned.
            no_recv = ~any_recv
            if no_recv.any():
                cell_pi = pi[:, :, :-1] * any_recv[:, :, None]
                un_pi = torch.where(no_recv, torch.ones_like(pi[:, :, -1]), pi[:, :, -1])
                pi = torch.cat([cell_pi, un_pi[:, :, None]], dim=2)
                pi = pi / pi.sum(dim=2, keepdim=True)

            conc = torch.clamp(tau_alpha * pi, min=1e-5)
            alpha = pyro.sample(f"alpha__{tid}", dist.Dirichlet(conc).to_event(2))

            # Move any numerically sampled mass on ineligible receivers to unassigned.
            cell_alpha = alpha[:, :, :-1] * recv_mask_nqk
            invalid_mass = (alpha[:, :, :-1] * (~recv_mask_nqk)).sum(dim=2)
            un_alpha = alpha[:, :, -1] + invalid_mass
            total = cell_alpha.sum(dim=2) + un_alpha
            cell_alpha = cell_alpha / total[:, :, None]
            un_alpha = un_alpha / total

            F = P[:, :, None] * rho[:, :, None] * cell_alpha
            Rret = P * (1.0 - rho)
            incoming = torch.zeros_like(P)
            for s in range(k):
                incoming = incoming.index_add(0, neighbors[:, s], F[:, :, s])
            Mpred = Rret + incoming
            zpred = _torch_zscore_log(torch, Mpred)

            pyro.deterministic(f"Mpred__{tid}", Mpred)
            pyro.deterministic(f"Rret__{tid}", Rret)
            pyro.deterministic(f"incoming__{tid}", incoming)

            pyro.sample(
                f"msi__{tid}",
                dist.Normal(zpred, sigma_q[None, :]).to_event(2),
                obs=zobs,
            )

    return model


@dataclass
class CalibrationFit:
    guide: object
    model: object
    losses: List[float]
    calibration_tissues: Sequence[PreparedTissue]


def fit_bayesian_calibration(
    calibration_tissues: Sequence[PreparedTissue],
    steps: int = 4000,
    lr: float = 0.01,
    seed: int = 20261001,
    verbose_every: int = 250,
) -> CalibrationFit:
    torch, pyro, dist, SVI, Trace_ELBO, Predictive, AutoNormal, ClippedAdam = _require_pyro()
    pyro.clear_param_store()
    pyro.set_rng_seed(seed)
    model = make_pyro_calibration_model(calibration_tissues)
    guide = AutoNormal(model)
    optimizer = ClippedAdam({"lr": lr, "clip_norm": 10.0})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    losses = []
    for step in range(1, steps + 1):
        loss = float(svi.step())
        losses.append(loss)
        if verbose_every and step % verbose_every == 0:
            print(f"SVI step {step:5d} loss={loss:.3f}")
    return CalibrationFit(guide=guide, model=model, losses=losses, calibration_tissues=calibration_tissues)


def sample_global_posterior(fit: CalibrationFit, n_draws: int = 500) -> Dict[str, np.ndarray]:
    torch, pyro, dist, SVI, Trace_ELBO, Predictive, AutoNormal, ClippedAdam = _require_pyro()
    names = [
        "beta0", "sigma_met", "u_q",
        "gamma0", "gammaS", "kappa_rho", "tau_alpha", "sigma_q",
    ]
    for c in ["betaE", "betaD", "thetaProx", "thetaD", "thetaI"]:
        names.extend([f"mu_log_{c}", f"sigma_log_{c}"])
    pred = Predictive(fit.model, guide=fit.guide, num_samples=n_draws, return_sites=names)
    draws = pred()
    return {k: v.detach().cpu().numpy() for k, v in draws.items()}


# -----------------------------------------------------------------------------
# Held-out posterior prediction with uncertainty propagated to M_pred
# -----------------------------------------------------------------------------

def predict_heldout_from_population(
    tissue_variants: Sequence[PreparedTissue],
    posterior: Dict[str, np.ndarray],
    seed: int = 20261001,
) -> Dict[str, np.ndarray]:
    """
    Generate held-out posterior-predictive draws.

    tissue_variants can contain one prepared tissue or multiple upstream
    realizations (e.g. bootstrap Compass/CVAE draws). Posterior draw b uses
    tissue_variants[b % len(tissue_variants)], which propagates upstream
    uncertainty through P,D,E,rho,alpha,F to M_pred.
    """
    if not tissue_variants:
        raise ValueError("Need at least one held-out tissue variant")
    rng = np.random.default_rng(seed)
    B = int(posterior["beta0"].shape[0])
    n, Q = tissue_variants[0].pde.P.shape
    k = tissue_variants[0].graph.k

    Mdraws = np.empty((B, n, Q), dtype=float)
    Rdraws = np.empty_like(Mdraws)
    incoming_draws = np.empty_like(Mdraws)
    rho_draws = np.empty_like(Mdraws)
    Lunassigned_draws = np.empty_like(Mdraws)
    Fdraws = np.empty((B, n, Q, k), dtype=float)

    for b in range(B):
        t = tissue_variants[b % len(tissue_variants)]
        # Draw new-tissue coefficients from learned population distributions.
        coeff = {}
        for name in ["betaE", "betaD", "thetaProx", "thetaD", "thetaI"]:
            mu = float(np.asarray(posterior[f"mu_log_{name}"][b]).squeeze())
            sig = float(np.asarray(posterior[f"sigma_log_{name}"][b]).squeeze())
            coeff[name] = math.exp(rng.normal(mu, max(sig, 1e-8)))

        out = forward_draw_numpy(
            tissue=t,
            beta0=float(np.asarray(posterior["beta0"][b]).squeeze()),
            betaE=coeff["betaE"],
            betaD=coeff["betaD"],
            u_q=np.asarray(posterior["u_q"][b]).reshape(-1),
            kappa_rho=float(np.asarray(posterior["kappa_rho"][b]).squeeze()),
            theta_prox=coeff["thetaProx"],
            thetaD_recv=coeff["thetaD"],
            thetaI_recv=coeff["thetaI"],
            gamma0=float(np.asarray(posterior["gamma0"][b]).squeeze()),
            gammaS=float(np.asarray(posterior["gammaS"][b]).squeeze()),
            tau_alpha=float(np.asarray(posterior["tau_alpha"][b]).squeeze()),
            rng=rng,
        )
        Mdraws[b] = out["Mpred"]
        Rdraws[b] = out["Rret"]
        incoming_draws[b] = out["incoming"]
        rho_draws[b] = out["rho"]
        Lunassigned_draws[b] = out["Lunassigned"]
        Fdraws[b] = out["F"]

    return {
        "Mpred_draws": Mdraws,
        "Mpred_mean": Mdraws.mean(axis=0),
        "Mpred_median": np.median(Mdraws, axis=0),
        "Mpred_lo": np.quantile(Mdraws, 0.025, axis=0),
        "Mpred_hi": np.quantile(Mdraws, 0.975, axis=0),
        "Rret_draws": Rdraws,
        "incoming_draws": incoming_draws,
        "rho_draws": rho_draws,
        "Lunassigned_draws": Lunassigned_draws,
        "F_draws": Fdraws,
    }


# -----------------------------------------------------------------------------
# Upstream uncertainty helper
# -----------------------------------------------------------------------------

def prepare_upstream_variants(
    tissue: TissueInput,
    Xmodel_draws: Sequence[np.ndarray],
    Xstar: np.ndarray,
    reaction_model: ReactionModel,
    graph: SpatialGraph,
) -> List[PreparedTissue]:
    """Convert reaction-representation draws into downstream prepared states."""
    msi = align_tissue_msi(tissue, reaction_model.metabolite_ids)
    msi_z = standardized_log_profile(msi) if msi is not None else None
    out = []
    for b, Xb in enumerate(Xmodel_draws):
        out.append(
            PreparedTissue(
                tissue_id=tissue.tissue_id,
                spot_ids=tissue.spot_ids,
                coords=tissue.coords,
                Xstar=Xstar,
                Xmodel=np.asarray(Xb, dtype=float),
                pde=compute_pde(Xb, reaction_model),
                graph=graph,
                msi=msi,
                msi_z=msi_z,
                metabolite_ids=reaction_model.metabolite_ids,
            )
        )
    return out


# -----------------------------------------------------------------------------
# Saving outputs
# -----------------------------------------------------------------------------

def save_prediction_outputs(
    outdir: str | Path,
    tissue: PreparedTissue,
    pred: Dict[str, np.ndarray],
) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    mids = tissue.metabolite_ids

    def save_matrix(name: str, arr: np.ndarray):
        df = pd.DataFrame(arr, index=tissue.spot_ids, columns=mids)
        df.index.name = "spot_id"
        df.to_csv(outdir / f"{name}.csv")

    save_matrix("Mpred_mean", pred["Mpred_mean"])
    save_matrix("Mpred_median", pred["Mpred_median"])
    save_matrix("Mpred_ci025", pred["Mpred_lo"])
    save_matrix("Mpred_ci975", pred["Mpred_hi"])

    np.savez_compressed(
        outdir / "posterior_predictive_draws.npz",
        Mpred=pred["Mpred_draws"],
        Rret=pred["Rret_draws"],
        incoming=pred["incoming_draws"],
        rho=pred["rho_draws"],
        Lunassigned=pred["Lunassigned_draws"],
        F=pred["F_draws"],
    )

    if tissue.msi is not None:
        metrics = evaluate_mpred_spearman(pred["Mpred_mean"], tissue.msi, mids)
        metrics.to_csv(outdir / "heldout_msi_spearman.csv", index=False)


# -----------------------------------------------------------------------------
# Minimal manifest-based example runner
# -----------------------------------------------------------------------------

def read_manifest(path: str | Path) -> pd.DataFrame:
    """
    Manifest columns:
      tissue_id,split,compass_scores,coordinates,msi
    split in {'calibration','heldout'}; msi may be blank for tissues without MSI.
    """
    df = pd.read_csv(path)
    need = {"tissue_id", "split", "compass_scores", "coordinates"}
    if not need.issubset(df.columns):
        raise ValueError(f"Manifest must contain {sorted(need)}")
    if "msi" not in df.columns:
        df["msi"] = np.nan
    return df


def _nonnull_path(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    return s if s else None


def run_reference_pipeline(config: Dict) -> None:
    """End-to-end reference runner for one held-out tissue."""
    set_seed(int(config.get("seed", 20261001)))
    manifest = read_manifest(config["manifest"])

    tissues: Dict[str, TissueInput] = {}
    for row in manifest.itertuples(index=False):
        tissues[str(row.tissue_id)] = load_tissue(
            tissue_id=str(row.tissue_id),
            compass_path=row.compass_scores,
            coords_path=row.coordinates,
            msi_path=_nonnull_path(row.msi),
        )

    calibration_ids = manifest.loc[manifest["split"] == "calibration", "tissue_id"].astype(str).tolist()
    heldout_ids = manifest.loc[manifest["split"] == "heldout", "tissue_id"].astype(str).tolist()
    if len(heldout_ids) != 1:
        raise ValueError("Reference runner currently expects exactly one held-out tissue")
    heldout_id = heldout_ids[0]

    # Common reactions are defined from the first calibration Compass matrix.
    first = tissues[calibration_ids[0]]
    reaction_ids = first.reaction_ids

    # Define the common metabolite set from the requested list or calibration MSI.
    if "metabolites" in config:
        metabolite_ids = [str(x) for x in config["metabolites"]]
    else:
        if first.msi_metabolite_ids is None:
            raise ValueError("Provide config['metabolites'] when calibration MSI is unavailable")
        metabolite_ids = first.msi_metabolite_ids

    rm = load_reaction_model(
        stoichiometry_path=config["stoichiometry"],
        reaction_metadata_path=config["reaction_metadata"],
        transport_map_path=config["transport_map"],
        reaction_ids=reaction_ids,
        metabolite_ids=metabolite_ids,
    )

    score_mode = config.get("score_mode", "penalty")
    oriented_cal = [
        orient_compass_scores(align_tissue_reactions(tissues[tid], reaction_ids), score_mode)
        for tid in calibration_ids
    ]
    scaler = RobustReactionScaler(
        lower_quantile=float(config.get("scale_q_low", 0.01)),
        upper_quantile=float(config.get("scale_q_high", 0.99)),
    ).fit(oriented_cal)

    upstream_model = str(config.get("upstream_model", "A")).upper()
    graph_k = int(config.get("graph_k", 6))
    d_max = config.get("d_max", None)
    if d_max is not None:
        d_max = float(d_max)

    cvaes: Dict[str, SpatialCVAE] = {}
    if upstream_model == "C":
        # Fit per tissue using ST-derived X* and coordinates only. Hyperparameters
        # must be chosen from calibration experiments before final held-out use.
        for tid, t in tissues.items():
            Xraw = align_tissue_reactions(t, reaction_ids)
            Xstar = scaler.transform(orient_compass_scores(Xraw, score_mode))
            model = SpatialCVAE(
                n_reactions=len(reaction_ids),
                latent_dim=int(config.get("cvae_latent_dim", 16)),
                hidden_dim=int(config.get("cvae_hidden_dim", 128)),
                beta_kl=float(config.get("cvae_beta_kl", 1e-3)),
            )
            model.fit(
                Xstar,
                t.coords,
                epochs=int(config.get("cvae_epochs", 1200)),
                lr=float(config.get("cvae_lr", 1e-3)),
            )
            cvaes[tid] = model

    prepared_cal = []
    for tid in calibration_ids:
        prepared_cal.append(
            prepare_tissue(
                tissues[tid], rm, scaler, score_mode,
                upstream_model=upstream_model,
                graph_k=graph_k,
                d_max=d_max,
                smooth_lambda=float(config.get("smooth_lambda", 0.35)),
                tau_sm=config.get("tau_sm", None),
                cvae=cvaes.get(tid),
            )
        )

    held = prepare_tissue(
        tissues[heldout_id], rm, scaler, score_mode,
        upstream_model=upstream_model,
        graph_k=graph_k,
        d_max=d_max,
        smooth_lambda=float(config.get("smooth_lambda", 0.35)),
        tau_sm=config.get("tau_sm", None),
        cvae=cvaes.get(heldout_id),
    )

    fit = fit_bayesian_calibration(
        prepared_cal,
        steps=int(config.get("svi_steps", 4000)),
        lr=float(config.get("svi_lr", 0.01)),
        seed=int(config.get("seed", 20261001)),
    )
    posterior = sample_global_posterior(fit, n_draws=int(config.get("posterior_draws", 500)))

    # Default: one deterministic upstream state. To propagate upstream bootstrap
    # or CVAE uncertainty, create multiple PreparedTissue variants and pass them
    # here instead.
    pred = predict_heldout_from_population([held], posterior, seed=int(config.get("seed", 20261001)))

    outdir = Path(config.get("outdir", f"spatial_gsmm_{heldout_id}_{upstream_model}"))
    save_prediction_outputs(outdir, held, pred)

    pd.DataFrame({"loss": fit.losses}).to_csv(outdir / "svi_loss.csv", index=False)
    with open(outdir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved results to {outdir}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Spatial GSMM consolidated reference pipeline")
    p.add_argument("--config", required=True, help="JSON configuration file")
    args = p.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    run_reference_pipeline(config)


if __name__ == "__main__":
    main()

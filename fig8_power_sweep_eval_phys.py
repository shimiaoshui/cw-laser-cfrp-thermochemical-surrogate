# -*- coding: utf-8 -*-
"""
Fig8 power sweep evaluator (Problem-6) — stable(ours) + suite(baselines)
========================================================================

Adds physics-consistency metrics (autograd-based) for Fig8(c):
- BC flux residual proxy: r_bc = (-n·k∇T) - (q_laser + q_conv + q_rad + q_sub)
  evaluated on front/back surface ROI points.

Notes:
- q_conv, q_rad follow COMSOL sign convention used in your models: inward flux into domain.
- n is outward unit normal: front n_z=+1, back n_z=-1.
- We approximate -n·k∇T by using ∂T/∂z and kz (normal direction is z on flat plate).

Outputs (in addition to your originals):
- fig8_records.csv includes:
    front_rbc_mae_roi, front_rbc_rel_roi, back_rbc_mae_roi, back_rbc_rel_roi
- fig8_by_power.csv includes their power-aggregated means
- Fig8c_front_BCFluxResidual_vs_P.png (+ back version)

Usage:
    python fig8_power_sweep_eval_phys.py --base_dir D:\\COMSOL\\data ... --model "..."
"""

from __future__ import annotations
import os, sys, re, json, math, time, argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import torch


# ---------------------------
# Utilities
# ---------------------------

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def _running_in_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore
        ip = get_ipython()
        if ip is None:
            return False
        return "IPKernelApp" in ip.config
    except Exception:
        return "ipykernel" in sys.modules

def _as_float_list(s: str) -> List[float]:
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    out: List[float] = []
    for tok in re.split(r"[,\s;]+", s):
        tok = tok.strip()
        if not tok:
            continue
        tok = tok.replace("_", ".")
        out.append(float(tok))
    return out

def _to_device(dev: str) -> torch.device:
    dev = str(dev or "cpu").strip().lower()
    if dev in ("cpu",):
        return torch.device("cpu")
    if dev in ("cuda", "gpu"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)

def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not isinstance(sd, dict) or not sd:
        return sd
    if any(k.startswith("module.") for k in sd.keys()):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd

def _remap_legacy_sine_keys(sd: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], bool]:
    """
    Compatibility remap for older checkpoints where SineLayer was saved as a plain Linear:
      first.weight -> first.linear.weight
      hidden.i.weight -> hidden.i.linear.weight
    """
    out: Dict[str, torch.Tensor] = {}
    changed = False
    pat = re.compile(r"^(first|hidden\.\d+)\.(weight|bias)$")
    for k, v in sd.items():
        m = pat.match(k)
        if m:
            k2 = f"{m.group(1)}.linear.{m.group(2)}"
            out[k2] = v
            changed = True
        else:
            out[k] = v
    return out, changed

def _inv_softplus(y: float) -> float:
    # inverse of softplus(y)=log(1+exp(x))
    return float(math.log(math.expm1(float(y))))

def _ensure_missing_a_raw(sd: Dict[str, torch.Tensor], model: torch.nn.Module) -> int:
    """
    Newer SineLayer uses a_raw (LAAF) for hidden layers; older ckpts don't have it.
    We inject a_raw so the effective multiplier a≈1 (matching legacy fixed-a behavior).
    """
    n = 0
    # a = softplus(a_raw) + 1e-3 -> set a=1.0
    a_raw = _inv_softplus(1.0 - 1e-3)
    for k in model.state_dict().keys():
        if not k.endswith(".a_raw"):
            continue
        if k in sd:
            continue
        if k.startswith("first."):
            sd[k] = torch.zeros(1, dtype=torch.float32)
        else:
            sd[k] = torch.tensor([a_raw], dtype=torch.float32)
        n += 1
    return n

def _is_num(x: Any) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False

def _find_scales_json(ckpt_path: str) -> Optional[str]:
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
    cands = [
        os.path.join(ckpt_dir, "scales.json"),
        os.path.join(os.path.dirname(ckpt_dir), "scales.json"),
        os.path.join(os.path.dirname(os.path.dirname(ckpt_dir)), "scales.json"),
    ]
    for p in cands:
        if os.path.isfile(p):
            return p
    return None


# ---------------------------
# Import helpers
# ---------------------------

def import_module_smart(tag: str):
    import importlib
    tag0 = str(tag).strip()
    low = tag0.lower()

    candidates: List[str] = []
    if low in ("stable", "ours"):
        candidates = ["stable_pinn_problem6_fixed_v4_ckpt", "stable", tag0]
    elif low in ("suite", "baselines"):
        candidates = ["problem6_baselines_suite", "suite", tag0]
    else:
        candidates = [tag0]

    last_err = None
    for name in candidates:
        try:
            return importlib.import_module(name)
        except Exception as e:
            last_err = e
    raise ImportError(f"Cannot import module for tag='{tag0}'. Tried: {candidates}. Last error: {last_err}")


# ---------------------------
# Model load (CRITICAL FIX)
# ---------------------------

@dataclass
class ModelEntry:
    name: str
    module_tag: str
    preset: str
    ckpt_path: str
    mod: Any
    cfg: Any
    model: torch.nn.Module
    scales: Dict[str, float]
    device: torch.device

def _load_ckpt_and_state(ckpt_path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = None
    if isinstance(ckpt, dict):
        for k in ("model", "model_state_dict", "state_dict", "net", "weights"):
            if k in ckpt and isinstance(ckpt[k], dict):
                state = ckpt[k]
                break
    if state is None:
        if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state = ckpt
        else:
            raise ValueError(f"Unrecognized checkpoint format: {ckpt_path}")
    state = _strip_module_prefix(state)
    return state, ckpt

def _apply_cfg_dict(cfg: Any, cfg_dict: Dict[str, Any]) -> None:
    if not isinstance(cfg_dict, dict):
        return
    for k, v in cfg_dict.items():
        if hasattr(cfg, k):
            try:
                setattr(cfg, k, v)
            except Exception:
                pass

def load_model_entry(entry: str, device: torch.device, strict: bool = True) -> ModelEntry:
    parts = [p.strip() for p in str(entry).split(",")]
    if len(parts) < 4:
        raise ValueError(f"Bad --model entry: {entry}. Need 4 comma-separated fields.")
    name, module_tag, preset, ckpt_path = parts[0], parts[1], parts[2], ",".join(parts[3:]).strip()
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    mod = import_module_smart(module_tag)
    if not hasattr(mod, "CFG"):
        raise AttributeError(f"Module '{module_tag}' has no CFG class.")
    cfg = mod.CFG()

    # ---- load ckpt first, apply ckpt cfg BEFORE building model ----
    sd, ckpt = _load_ckpt_and_state(ckpt_path)
    if isinstance(ckpt, dict):
        if "cfg" in ckpt and isinstance(ckpt["cfg"], dict):
            _apply_cfg_dict(cfg, ckpt["cfg"])
        if "cfg_dict" in ckpt and isinstance(ckpt["cfg_dict"], dict):
            _apply_cfg_dict(cfg, ckpt["cfg_dict"])

    # runtime fields
    try:
        cfg.device = str(device)
    except Exception:
        pass
    try:
        cfg.out_dir = os.path.dirname(os.path.abspath(ckpt_path))
    except Exception:
        pass

    # preset (suite baselines)
    preset0 = str(preset or "_").strip()
    if preset0 and preset0 != "_" and hasattr(mod, "apply_model_preset"):
        mod.apply_model_preset(cfg, preset0)
    else:
        if preset0 and preset0 != "_" and hasattr(cfg, "model_kind"):
            try:
                cfg.model_kind = preset0
            except Exception:
                pass

    # instantiate model with corrected cfg
    if hasattr(mod, "AblationSIRENFiLM"):
        model = mod.AblationSIRENFiLM(cfg).to(device)
    elif hasattr(mod, "make_model"):
        model = mod.make_model(cfg).to(device)
    else:
        raise AttributeError(f"Module '{module_tag}' doesn't expose AblationSIRENFiLM or make_model().")

    # ---- robust load: handle legacy key naming and missing a_raw ----
    try:
        _ensure_missing_a_raw(sd, model)
        res = model.load_state_dict(sd, strict=strict)
    except RuntimeError:
        sd2, changed = _remap_legacy_sine_keys(sd)
        if changed:
            sd = sd2
            injected = _ensure_missing_a_raw(sd, model)
            try:
                res = model.load_state_dict(sd, strict=strict)
            except RuntimeError:
                res = model.load_state_dict(sd, strict=False)
            print(f"[Compat] Remapped legacy keys for {name}; injected a_raw={injected}; strict={strict}")
        else:
            raise

    if hasattr(res, "missing_keys") and (res.missing_keys or res.unexpected_keys):
        if res.unexpected_keys:
            raise RuntimeError(f"[LoadError] {name}: unexpected keys remain: {res.unexpected_keys[:10]} ...")
        print(f"[Warn] load_state_dict {name}: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")

    scales: Dict[str, float] = {}
    if isinstance(ckpt, dict) and "scales" in ckpt and isinstance(ckpt["scales"], dict):
        scales = {k: float(v) for k, v in ckpt["scales"].items() if _is_num(v)}
    else:
        sj = _find_scales_json(ckpt_path)
        if sj:
            try:
                with open(sj, "r", encoding="utf-8") as f:
                    js = json.load(f)
                if isinstance(js, dict):
                    scales = {k: float(v) for k, v in js.items() if _is_num(v)}
            except Exception:
                scales = {}

    if not scales:
        print(f"[Warn] scales not found for model={name}. If outputs look wrong, check ckpt/scales.json.")

    model.eval()
    return ModelEntry(name, module_tag, preset0, ckpt_path, mod, cfg, model, scales, device)


# ---------------------------
# Dataset helpers
# ---------------------------

def surface_ids_from_coords(coords: np.ndarray, tol: float = 2e-5) -> Tuple[np.ndarray, np.ndarray]:
    z = coords[:, 2]
    zmin = float(np.min(z))
    zmax = float(np.max(z))
    back = np.where(np.abs(z - zmin) <= tol)[0]
    front = np.where(np.abs(z - zmax) <= tol)[0]
    return front.astype(np.int64), back.astype(np.int64)

def roi_mask_xy(xy: np.ndarray, x0: float, y0: float, r: float) -> np.ndarray:
    dx = xy[:, 0] - float(x0)
    dy = xy[:, 1] - float(y0)
    return ((dx*dx + dy*dy) <= (r*r))

def nearest_center_index(xy: np.ndarray, x0: float, y0: float) -> int:
    dx = xy[:, 0] - float(x0)
    dy = xy[:, 1] - float(y0)
    return int(np.argmin(dx*dx + dy*dy))

def _parse_splits(s: str) -> List[str]:
    s = (s or "").strip().lower()
    if (not s) or (s == "all"):
        return ["train", "val", "test"]
    parts = [p.strip().lower() for p in str(s).replace(";", ",").split(",") if p.strip()]
    out: List[str] = []
    for p in parts:
        if p in ("train", "tr"):
            out.append("train")
        elif p in ("val", "valid", "validation"):
            out.append("val")
        elif p in ("test", "te"):
            out.append("test")
    return out or ["train", "val", "test"]

def _filter_specs(specs: List[Any], powers: List[float], splits: List[str]) -> List[Any]:
    split_allow = set(_parse_splits(",".join(splits) if isinstance(splits, list) else str(splits)))

    def match_power(P: Any) -> bool:
        if not powers:
            return True
        try:
            Pf = float(P)
        except Exception:
            return False
        for p in powers:
            if abs(Pf - float(p)) <= 1e-3:
                return True
        return False

    out: List[Any] = []
    for sp in specs:
        s = str(getattr(sp, "split", "train")).strip().lower()
        if s not in split_allow:
            continue
        if not match_power(getattr(sp, "P_laser", float("nan"))):
            continue
        out.append(sp)

    out.sort(key=lambda sp: (float(getattr(sp, "P_laser", 0.0)),
                             float(getattr(sp, "t_ref", 0.0) or 0.0),
                             str(getattr(sp, "name", ""))))
    return out

def _summarize_specs(specs: List[Any]) -> str:
    if not specs:
        return "0 specs"
    from collections import Counter
    c_split = Counter([str(getattr(sp, "split", "train")).lower() for sp in specs])
    c_pow = Counter([round(float(getattr(sp, "P_laser", float('nan'))), 4) if _is_num(getattr(sp, "P_laser", None)) else "nan"
                     for sp in specs])
    s1 = "splits=" + ", ".join([f"{k}:{v}" for k, v in sorted(c_split.items(), key=lambda x: x[0])])
    pow_items = sorted([(k, v) for k, v in c_pow.items() if k != "nan"], key=lambda x: x[0])
    s2 = "powers=" + ", ".join([f"{k}:{v}" for k, v in pow_items[:10]]) + (" ..." if len(pow_items) > 10 else "")
    return f"N={len(specs)} | {s1} | {s2}"


# ---------------------------
# Prediction (T only)
# ---------------------------

@torch.no_grad()
def predict_T_on_nodes(me: ModelEntry,
                       spec: Any,
                       coords_subset: np.ndarray,
                       t_eval: float,
                       t_ref: float,
                       const_cpu: Dict[str, torch.Tensor],
                       batch_nodes: int = 4096) -> np.ndarray:
    mod = me.mod
    device = me.device
    const = {k: v.to(device) for k, v in const_cpu.items()}

    scen_np = mod.scenario_vec(spec).astype(np.float32, copy=False)
    M = coords_subset.shape[0]
    out_T = np.empty((M,), dtype=np.float32)

    for s in range(0, M, batch_nodes):
        e = min(M, s + batch_nodes)
        xyz = torch.from_numpy(coords_subset[s:e]).to(device=device, dtype=torch.float32)
        tb = torch.full((e - s, 1), float(t_eval), device=device, dtype=torch.float32)
        xytzn = mod.normalize_xyt(me.cfg, xyz, tb, float(t_ref))
        scen = torch.from_numpy(scen_np[None, :]).to(device=device, dtype=torch.float32).repeat(e - s, 1)
        out = me.model.forward_phys(xytzn, scen, const, me.scales)
        out_T[s:e] = out["T"].detach().to("cpu").numpy().reshape(-1).astype(np.float32, copy=False)
    return out_T


# ---------------------------
# Physics consistency (BC flux residual)
# ---------------------------

def _safe_get(out: Dict[str, torch.Tensor], key: str) -> Optional[torch.Tensor]:
    v = out.get(key, None) if isinstance(out, dict) else None
    if v is None:
        return None
    if not torch.is_tensor(v):
        return None
    return v

def bc_flux_residual_on_nodes(me: ModelEntry,
                              spec: Any,
                              coords_subset: np.ndarray,
                              t_eval: float,
                              t_ref: float,
                              const_cpu: Dict[str, torch.Tensor],
                              n_z: float,
                              batch_nodes: int = 1024,
                              eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute BC flux residual on given nodes:
      r = (-n·k∇T) - (q_laser + q_conv + q_rad + q_sub)
    Returns:
      r_abs: |r|  (W/m^2)
      r_rel: |r| / (|q_laser|+|q_conv|+|q_rad|+|q_sub|+eps)
    """
    mod = me.mod
    device = me.device
    const = {k: v.to(device) for k, v in const_cpu.items()}

    scen_np = mod.scenario_vec(spec).astype(np.float32, copy=False)
    M = coords_subset.shape[0]
    r_abs = np.empty((M,), dtype=np.float32)
    r_rel = np.empty((M,), dtype=np.float32)

    # normalization scale: because zn = z/(Lz/2) => dT/dz = dT/dzn * (2/Lz)
    Lz = float(getattr(me.cfg, "Lz", 1.0))
    dz_scale = 2.0 / max(Lz, 1e-12)

    for s in range(0, M, batch_nodes):
        e = min(M, s + batch_nodes)
        xyz = torch.from_numpy(coords_subset[s:e]).to(device=device, dtype=torch.float32)
        tb = torch.full((e - s, 1), float(t_eval), device=device, dtype=torch.float32)

        xytzn = mod.normalize_xyt(me.cfg, xyz, tb, float(t_ref))
        xytzn = xytzn.detach().clone().requires_grad_(True)

        scen = torch.from_numpy(scen_np[None, :]).to(device=device, dtype=torch.float32).repeat(e - s, 1)

        out = me.model.forward_phys(xytzn, scen, const, me.scales)

        T = _safe_get(out, "T")
        kz = _safe_get(out, "kz")
        q_laser = _safe_get(out, "q_laser")
        q_conv = _safe_get(out, "q_conv")
        q_rad = _safe_get(out, "q_rad")
        q_sub = _safe_get(out, "q_sub")

        if any(v is None for v in (T, kz, q_laser, q_conv, q_rad, q_sub)):
            # cannot compute; fill NaN
            r_abs[s:e] = np.nan
            r_rel[s:e] = np.nan
            continue

        # gradient w.r.t normalized coordinates
        g = torch.autograd.grad(T.sum(), xytzn, create_graph=False, retain_graph=False, allow_unused=False)[0]
        dT_dzn = g[:, 2:3]
        dT_dz = dT_dzn * dz_scale

        # outward normal has n_z; inward conductive flux into domain is -n·k∇T
        q_cond_in = -(float(n_z)) * kz * dT_dz

        q_rhs = q_laser + q_conv + q_rad + q_sub

        r = (q_cond_in - q_rhs)
        rabs = torch.abs(r)
        denom = (torch.abs(q_laser) + torch.abs(q_conv) + torch.abs(q_rad) + torch.abs(q_sub) + eps)
        rrel = rabs / denom

        r_abs[s:e] = rabs.detach().to("cpu").numpy().reshape(-1).astype(np.float32, copy=False)
        r_rel[s:e] = rrel.detach().to("cpu").numpy().reshape(-1).astype(np.float32, copy=False)

    return r_abs, r_rel


# ---------------------------
# Metrics + plotting
# ---------------------------

def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))

def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))

def nanmean(x: np.ndarray) -> float:
    return float(np.nanmean(x)) if np.any(np.isfinite(x)) else float("nan")

def nanp95(x: np.ndarray) -> float:
    x2 = x[np.isfinite(x)]
    if x2.size == 0:
        return float("nan")
    return float(np.percentile(x2, 95))

def plot_metric_vs_power(df_plot, metric_col: str, out_png: str, title: str, ylog: bool = False):
    models = list(dict.fromkeys(df_plot["model"].tolist()))
    plt.figure()
    for m in models:
        d = df_plot[df_plot["model"] == m].sort_values("P")
        if d.empty:
            continue
        plt.plot(d["P"].values, d[metric_col].values, marker="o", label=m)
    plt.xlabel("Power P (W)")
    plt.ylabel(metric_col)
    plt.title(title)
    if ylog:
        plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

def plot_center_vs_power(df_plot, surface: str, out_png: str, title: str):
    models = list(dict.fromkeys(df_plot["model"].tolist()))
    plt.figure()
    dtrue = (df_plot.groupby("P", as_index=False)[f"{surface}_Tcenter_true"]
                  .mean(numeric_only=True).sort_values("P"))
    plt.plot(dtrue["P"].values, dtrue[f"{surface}_Tcenter_true"].values, marker="s", label="True")

    for m in models:
        d = df_plot[df_plot["model"] == m].sort_values("P")
        if d.empty:
            continue
        plt.plot(d["P"].values, d[f"{surface}_Tcenter_pred"].values, marker="o", label=m)
    plt.xlabel("Power P (W)")
    plt.ylabel("Center temperature (K)")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


# ---------------------------
# T-only dataset fallback (avoids suite.load_dataset crashing when alpha/v_sub/v_mesh exports are missing)
# ---------------------------

from types import SimpleNamespace

def load_dataset_T_only(suite_mod, spec, cache_dir):
    """Load only temperature wide-table (coords, times, T). Works even if alpha/v_sub/v_mesh exports are absent."""
    wtT = suite_mod.load_wide_table(spec.T_path, cache_dir)
    T_key = suite_mod.pick_var(wtT.data, ["T", "temperature"])
    T = wtT.data[T_key]
    return SimpleNamespace(coords=wtT.coords, times=wtT.times, T=T)

def safe_load_dataset(suite_mod, spec, cache_dir):
    """Try suite.load_dataset; if it fails due to missing alpha/velocity files, fall back to T-only."""
    try:
        return suite_mod.load_dataset(spec, cache_dir)
    except FileNotFoundError as e:
        print(f"[T-only fallback] {getattr(spec,'name','(noname)')}: {e}")
        return load_dataset_T_only(suite_mod, spec, cache_dir)
    except KeyError as e:
        msg = str(e)
        if any(k in msg for k in ("alphap01", "alphap", "alpha_p", "alphac01", "alphaf01", "alpha_c", "alpha_f")):
            print(f"[T-only fallback] {getattr(spec,'name','(noname)')}: {msg}")
            return load_dataset_T_only(suite_mod, spec, cache_dir)
        raise


# ---------------------------
# Main
# ---------------------------

def main(argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--java_path", type=str, default="")
    ap.add_argument("--default_power", type=float, default=None)
    ap.add_argument("--powers", type=str, default="")
    ap.add_argument("--splits", type=str, default="all")
    ap.add_argument("--include_all", action="store_true")
    ap.add_argument("--t_abs", type=float, default=None)
    ap.add_argument("--t_frac", type=float, default=1.0)
    ap.add_argument("--roi_r_mm", type=float, default=0.5)
    ap.add_argument("--surface_tol", type=float, default=2e-5)
    ap.add_argument("--batch_nodes", type=int, default=4096)
    ap.add_argument("--group_mode", type=str, default="mean")
    ap.add_argument("--no_ckpt_strict", action="store_true")
    ap.add_argument("--also_mse", action="store_true")
    ap.add_argument("--model", type=str, action="append", default=[])
    ap.add_argument("--extra_sys_path", type=str, default="")

    # NEW: physics-consistency controls
    ap.add_argument("--no_phys", action="store_true", help="Disable physics-consistency (BC flux residual) computation.")
    ap.add_argument("--phys_max_points", type=int, default=1024, help="Max ROI points per surface for BC residual.")
    ap.add_argument("--phys_batch_nodes", type=int, default=1024, help="Batch size for BC residual (autograd).")
    ap.add_argument("--phys_pmax", type=float, default=20.0, help="Only compute BC residual when P <= phys_pmax.")
    ap.add_argument("--plot_pmax", type=float, default=None, help="Only plot curves for P <= plot_pmax (e.g., 20).")
    ap.add_argument("--phys_eps", type=float, default=1e-8, help="Epsilon for relative residual normalization.")

    args = ap.parse_args(argv)

    # make imports work in notebook
    if args.extra_sys_path.strip():
        for p in re.split(r"[;\n]+", args.extra_sys_path.strip()):
            p = p.strip().strip('"').strip("'")
            if p and os.path.isdir(p) and (p not in sys.path):
                sys.path.insert(0, p)

    if not args.model:
        raise ValueError("No --model entries provided.")

    out_dir = args.out_dir.strip() or os.path.join(os.getcwd(), f"fig8_out_{_now_tag()}")
    out_dir = ensure_dir(out_dir)

    device = _to_device(args.device)
    roi_r = float(args.roi_r_mm) * 1e-3

    # suite for discovery + dataset loading + const
    suite = import_module_smart("suite")
    cfgD = suite.CFG()
    cfgD.base_dir = args.base_dir
    if args.java_path:
        cfgD.java_path = args.java_path
    if args.default_power is not None:
        try:
            cfgD.default_power = float(args.default_power)
        except Exception:
            pass
    if args.include_all:
        try:
            cfgD.keep_only_listed_pairs = False
        except Exception:
            pass

    # discover + params
    specs = suite.discover_datasets(cfgD)
    if not specs:
        raise ValueError(f"discover_datasets returned 0 specs. Check --base_dir='{args.base_dir}' and filenames.")
    print("[Discovery]", _summarize_specs(specs))

    params = {}
    if args.java_path and os.path.isfile(args.java_path) and hasattr(suite, "parse_comsol_java_params"):
        params = suite.parse_comsol_java_params(args.java_path)

    powers = _as_float_list(args.powers) if args.powers else []
    split_list = _parse_splits(args.splits)

    specs_f = _filter_specs(specs, powers=powers, splits=split_list)

    # ---- ONE-SHOT AUTO FIX: if empty, fall back to all splits ----
    if not specs_f and (args.splits.strip().lower() not in ("all", "")):
        print(f"[Warn] No datasets matched splits='{args.splits}'. Falling back to splits='all'.")
        split_list = ["train", "val", "test"]
        specs_f = _filter_specs(specs, powers=powers, splits=split_list)

    if not specs_f:
        raise ValueError(f"No datasets selected after filtering. powers='{args.powers}' splits='{args.splits}'. Discovery: {_summarize_specs(specs)}")

    # ---- Deduplicate datasets (some COMSOL exports may appear twice) ----
    uniq = []
    seen = set()
    for sp in specs_f:
        key = (
            str(getattr(sp, "T_path", "")),
            float(getattr(sp, "P_laser", float("nan"))) if getattr(sp, "P_laser", None) is not None else float("nan"),
            float(getattr(sp, "t_ref", float("nan"))) if getattr(sp, "t_ref", None) is not None else float("nan"),
        )
        if key in seen:
            continue
        seen.add(key)
        uniq.append(sp)
    if len(uniq) != len(specs_f):
        print(f"[Dedup] {len(specs_f)} -> {len(uniq)} after removing duplicates by (T_path,P,t_ref).")
    specs_f = uniq

    print("[Selected]", _summarize_specs(specs_f))

    cache_dir = ensure_dir(os.path.join(out_dir, "_cache_wt"))

    # pre-build const by dataset name (CPU)
    const_by_name: Dict[str, Dict[str, torch.Tensor]] = {}
    for sp in specs_f:
        const_cpu = suite.build_const(cfgD, sp, params, device="cpu")
        const_by_name[str(getattr(sp, "name", ""))] = {k: v.to("cpu") for k, v in const_cpu.items()}

    # load models
    strict = (not bool(args.no_ckpt_strict))
    models = [load_model_entry(m, device=device, strict=strict) for m in args.model]

    import pandas as pd
    rows = []

    rng = np.random.default_rng(12345)

    for sp in specs_f:
        name = str(getattr(sp, "name", ""))
        ds = safe_load_dataset(suite, sp, cache_dir)
        coords = ds.coords
        times = ds.times
        T = ds.T

        t_ref = float(getattr(sp, "t_ref", float(np.max(times))))
        t_req = float(args.t_abs) if (args.t_abs is not None) else float(args.t_frac) * t_ref
        tidx = int(np.argmin(np.abs(times - t_req)))
        t_eval = float(times[tidx])

        front_ids, back_ids = surface_ids_from_coords(coords, tol=float(args.surface_tol))
        front_xy = coords[front_ids, 0:2]
        back_xy = coords[back_ids, 0:2]

        x0 = float(getattr(sp, "x0", 0.0))
        y0 = float(getattr(sp, "y0", 0.0))
        front_roi = roi_mask_xy(front_xy, x0, y0, roi_r)
        back_roi = roi_mask_xy(back_xy, x0, y0, roi_r)

        Tt_front = T[front_ids, tidx].reshape(-1).astype(np.float32, copy=False)
        Tt_back = T[back_ids, tidx].reshape(-1).astype(np.float32, copy=False)

        cfi = nearest_center_index(front_xy, x0, y0)
        cbi = nearest_center_index(back_xy, x0, y0)

        P = float(getattr(sp, "P_laser", float("nan")))
        split = str(getattr(sp, "split", ""))

        const_cpu = const_by_name[name]

        # prepare ROI samples for BC residual (indices in surface-local arrays)
        def sample_roi_ids(mask: np.ndarray, max_n: int) -> np.ndarray:
            ids = np.where(mask)[0]
            if ids.size <= max_n:
                return ids.astype(np.int64)
            take = rng.choice(ids, size=max_n, replace=False)
            take.sort()
            return take.astype(np.int64)

        front_roi_ids = sample_roi_ids(front_roi, int(args.phys_max_points))
        back_roi_ids = sample_roi_ids(back_roi, int(args.phys_max_points))

        for me in models:
            Tp_front = predict_T_on_nodes(me, sp, coords[front_ids], t_eval=t_eval, t_ref=t_ref,
                                          const_cpu=const_cpu, batch_nodes=int(args.batch_nodes))
            Tp_back = predict_T_on_nodes(me, sp, coords[back_ids], t_eval=t_eval, t_ref=t_ref,
                                         const_cpu=const_cpu, batch_nodes=int(args.batch_nodes))

            # physics-consistency metrics (optional; default ON)
            front_rbc_mae = float("nan")
            front_rbc_rel = float("nan")
            front_rbc_p95 = float("nan")
            back_rbc_mae = float("nan")
            back_rbc_rel = float("nan")
            back_rbc_p95 = float("nan")

            if (not args.no_phys) and np.isfinite(P) and (P <= float(args.phys_pmax)):
                # front surface: outward normal +z
                if front_roi_ids.size > 0:
                    rabs, rrel = bc_flux_residual_on_nodes(
                        me, sp,
                        coords_subset=coords[front_ids[front_roi_ids]],
                        t_eval=t_eval, t_ref=t_ref, const_cpu=const_cpu,
                        n_z=+1.0,
                        batch_nodes=int(args.phys_batch_nodes),
                        eps=float(args.phys_eps),
                    )
                    front_rbc_mae = nanmean(rabs)
                    front_rbc_rel = nanmean(rrel)
                    front_rbc_p95 = nanp95(rabs)

                # back surface: outward normal -z
                if back_roi_ids.size > 0:
                    rabs, rrel = bc_flux_residual_on_nodes(
                        me, sp,
                        coords_subset=coords[back_ids[back_roi_ids]],
                        t_eval=t_eval, t_ref=t_ref, const_cpu=const_cpu,
                        n_z=-1.0,
                        batch_nodes=int(args.phys_batch_nodes),
                        eps=float(args.phys_eps),
                    )
                    back_rbc_mae = nanmean(rabs)
                    back_rbc_rel = nanmean(rrel)
                    back_rbc_p95 = nanp95(rabs)

            rec = {
                "dataset": name, "split": split, "P": P, "t_ref": float(t_ref),
                "t_req": float(t_req), "t_eval": float(t_eval),
                "model": me.name, "module": me.module_tag, "preset": me.preset, "ckpt": me.ckpt_path,
                "front_roi_mae": mae(Tp_front[front_roi], Tt_front[front_roi]) if front_roi.any() else float("nan"),
                "back_roi_mae": mae(Tp_back[back_roi], Tt_back[back_roi]) if back_roi.any() else float("nan"),
                "front_roi_mse": mse(Tp_front[front_roi], Tt_front[front_roi]) if front_roi.any() else float("nan"),
                "back_roi_mse": mse(Tp_back[back_roi], Tt_back[back_roi]) if back_roi.any() else float("nan"),
                "front_Tcenter_true": float(Tt_front[cfi]),
                "front_Tcenter_pred": float(Tp_front[cfi]),
                "back_Tcenter_true": float(Tt_back[cbi]),
                "back_Tcenter_pred": float(Tp_back[cbi]),
                # NEW: physics-consistency (BC flux residual on ROI sample)
                "front_rbc_mae_roi": float(front_rbc_mae),
                "front_rbc_p95_roi": float(front_rbc_p95),
                "front_rbc_rel_roi": float(front_rbc_rel),
                "back_rbc_mae_roi": float(back_rbc_mae),
                "back_rbc_p95_roi": float(back_rbc_p95),
                "back_rbc_rel_roi": float(back_rbc_rel),
            }
            rows.append(rec)

            msg_phys = ""
            if not args.no_phys and np.isfinite(P) and (P <= float(args.phys_pmax)):
                msg_phys = f" | rBC(front)={rec['front_rbc_rel_roi']:.3g} (rel)"
            print(f"[OK] {me.name:16s} | {name:20s} | P={P:7.3f}W | t={t_eval:7.3f}s | frontROI_MAE={rec['front_roi_mae']:.6g}{msg_phys}")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No evaluation records produced.")

    csv_records = os.path.join(out_dir, "fig8_records.csv")
    df.to_csv(csv_records, index=False, encoding="utf-8-sig")
    print("[Save]", csv_records)

    gm = str(args.group_mode or "mean").strip().lower()
    if gm != "mean":
        gm = "mean"
    df_plot = (df.groupby(["model", "P"], as_index=False).mean(numeric_only=True))

    csv_plot = os.path.join(out_dir, "fig8_by_power.csv")
    df_plot.to_csv(csv_plot, index=False, encoding="utf-8-sig")
    print("[Save]", csv_plot)

    # plotting filter (e.g., only show 0-20W)
    df_plot_vis = df_plot.copy()
    if args.plot_pmax is not None and np.isfinite(float(args.plot_pmax)):
        df_plot_vis = df_plot_vis[df_plot_vis["P"] <= float(args.plot_pmax)]

    t_tag = f"abs={args.t_abs}" if args.t_abs is not None else f"frac={args.t_frac}"

    plot_metric_vs_power(df_plot_vis, "front_roi_mae",
                         os.path.join(out_dir, "Fig8a_front_ROI_MAE_vs_P.png"),
                         f"Front ROI MAE vs Power (t_{t_tag})")
    plot_metric_vs_power(df_plot_vis, "back_roi_mae",
                         os.path.join(out_dir, "Fig8a_back_ROI_MAE_vs_P.png"),
                         f"Back ROI MAE vs Power (t_{t_tag})")

    if args.also_mse:
        plot_metric_vs_power(df_plot_vis, "front_roi_mse",
                             os.path.join(out_dir, "Fig8a_front_ROI_MSE_vs_P.png"),
                             f"Front ROI MSE vs Power (t_{t_tag})")
        plot_metric_vs_power(df_plot_vis, "back_roi_mse",
                             os.path.join(out_dir, "Fig8a_back_ROI_MSE_vs_P.png"),
                             f"Back ROI MSE vs Power (t_{t_tag})")

    plot_center_vs_power(df_plot_vis, "front",
                         os.path.join(out_dir, "Fig8b_front_Tcenter_vs_P.png"),
                         f"Front center temperature vs Power (t_{t_tag})")
    plot_center_vs_power(df_plot_vis, "back",
                         os.path.join(out_dir, "Fig8b_back_Tcenter_vs_P.png"),
                         f"Back center temperature vs Power (t_{t_tag})")

    # NEW: physics-consistency curves (recommend log scale for absolute residual)
    if not args.no_phys:
        plot_metric_vs_power(df_plot_vis, "front_rbc_mae_roi",
                             os.path.join(out_dir, "Fig8c_front_BCFluxResidual_MAE_vs_P.png"),
                             f"Front ROI BC-flux residual MAE vs Power (t_{t_tag})", ylog=True)
        plot_metric_vs_power(df_plot_vis, "back_rbc_mae_roi",
                             os.path.join(out_dir, "Fig8c_back_BCFluxResidual_MAE_vs_P.png"),
                             f"Back ROI BC-flux residual MAE vs Power (t_{t_tag})", ylog=True)
        plot_metric_vs_power(df_plot_vis, "front_rbc_rel_roi",
                             os.path.join(out_dir, "Fig8c_front_BCFluxResidual_REL_vs_P.png"),
                             f"Front ROI BC-flux residual (relative) vs Power (t_{t_tag})", ylog=False)

    print("[Done] Fig8 power sweep completed. Outputs:", out_dir)


# ---------------------------
# Notebook convenience
# ---------------------------

DEFAULT_NOTEBOOK_ARGS: List[str] = [
    "--base_dir", r"D:\COMSOL\data",
    "--java_path", r"C:\Users\28739\Laser\横向1\最终烧蚀3.java",
    "--default_power", "10.503",
    "--splits", "all",
    "--t_abs", "1.0",
    "--out_dir", r"C:\Users\28739\Laser\横向1\paper_fig88",
    "--extra_sys_path", r"C:\Users\28739\Laser\横向1",
    # NEW: only plot up to 20W in Fig8 curves (still evaluates all datasets unless you filter --powers)
    "--plot_pmax", "20",
    "--phys_pmax", "20",
    "--phys_max_points", "1024",
    "--phys_batch_nodes", "1024",
    "--model", r"ours,stable,_,C:\Users\28739\Laser\out_pinn_v10\checkpoints\ckpt_final.pt",
    "--model", r"data_only_mlp,suite,data_only_mlp,C:\Users\28739\Laser\横向1\out_baseline_data_only_mlp\checkpoints\ckpt_final.pt",
    "--model", r"mlp_pinn,suite,mlp_pinn,C:\Users\28739\Laser\横向1\out_baseline_mlp_pinn\checkpoints\ckpt_final.pt",
    "--model", r"ff_mlp_pinn,suite,ff_mlp_pinn,C:\Users\28739\Laser\横向1\out_baseline_ff_mlp_pinn\checkpoints\ckpt_final.pt",
    "--model", r"mlp_film_pinn,suite,mlp_film_pinn,C:\Users\28739\Laser\横向1\out_baseline_mlp_film_pinn\checkpoints\ckpt_final.pt",
    "--model", r"siren_nofilm_pinn,suite,siren_nofilm_pinn,C:\Users\28739\Laser\横向1\out_baseline_siren_nofilm_pinn\checkpoints\ckpt_final.pt",
]

def run_notebook():
    main(DEFAULT_NOTEBOOK_ARGS)


if __name__ == "__main__":
    if _running_in_notebook() and ("--base_dir" not in sys.argv):
        print("[Info] Notebook detected and no --base_dir provided; using DEFAULT_NOTEBOOK_ARGS.")
        main(DEFAULT_NOTEBOOK_ARGS)
    else:
        main()

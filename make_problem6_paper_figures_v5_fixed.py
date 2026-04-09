# -*- coding: utf-8 -*-
"""
make_problem6_paper_figures_v5.py
=================================

v5 additions (requested)
------------------------
- Dataset coverage: process *all discovered powers/durations* by default (no longer limited to a small representative subset).
  You can still request the old behavior via --representative_only or "representative_only": true in models.json.
- Power filtering: optionally restrict to specific laser powers via --powers "1,5,10.503,15,18,20,25,30"
  (or models.json: "powers": [1,5,10.503,...]).
- Jupyter-friendly: main(argv=None) so you can call main([...]) directly inside notebooks.
- Besides the default time fractions (e.g., 0.5, 0.8, 1.0 of t_ref),
  you can now specify *absolute* times in seconds (e.g., 1.0 s) and the script
  will generate ALL comparison figures at those times as well.

How to specify extra absolute times
-----------------------------------
Option A) in models.json:
  "times_abs_s": [1.0, 5.0]

Option B) via CLI:
  --times_abs "1,5"

Jupyter usage
-------------
In JupyterLab, you can run:
  %run make_problem6_paper_figures_v4.py --models_json models.json --times_abs "1"

Notes
-----
- If a requested absolute time is outside dataset time range, it is skipped with a warning.
- If the time is inside range but not exactly on ds.times grid, nearest time is used, and
  the *actual* plotted time is recorded in filenames/CSVs.

Outputs
-------
out_dir/
  figs/      PNG figures
  data/      CSV/NPZ numeric dumps
  manifest.json

"""

from __future__ import annotations

import os
import json
import argparse
from types import SimpleNamespace
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch

import warnings
warnings.filterwarnings(
    "ignore",
    message=r"You are using `torch\.load` with `weights_only=False`.*"
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


DEFAULT_BASE_DIR = r"D:\COMSOL\data"
DEFAULT_OUT_DIR  = r"C:\Users\28739\Laser\横向1\paper_figs_problem6"
DEFAULT_DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_INCLUDE_ALL = True
DEFAULT_MODELS: List[str] = []


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def _now_tag() -> str:
    import datetime
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def _safe_name(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(s))

def _infer_power_from_path(path: str):
    """Try to infer power in W from folder/file names like '10.503W' or '10_503W'."""
    if not path:
        return None
    s = str(path)
    m = re.search(r"(?P<p>[0-9]+(?:[._][0-9]+)?)\s*W", s, flags=re.IGNORECASE)
    if m:
        try:
            return float(m.group("p").replace("_", "."))
        except Exception:
            return None
    return None

def _fix_missing_power_specs(specs, default_power: float | None = None):
    """If a DatasetSpec has P_laser <= 0 and its path contains '<P>W', patch P_laser and name."""
    for sp in specs:
        try:
            P = float(getattr(sp, "P_laser", 0.0))
        except Exception:
            P = 0.0
        if P > 1e-9:
            continue
        # try infer from folder path
        p1 = _infer_power_from_path(getattr(sp, "T_path", "")) or _infer_power_from_path(os.path.dirname(getattr(sp, "T_path", "")))
        if p1 is None:
            if default_power is None:
                continue
            p1 = float(default_power)
        try:
            sp.P_laser = float(p1)
        except Exception:
            continue
        # improve name readability if it's a '*_base' placeholder
        nm = str(getattr(sp, "name", ""))
        if ("base" in nm) or nm.endswith("_base"):
            try:
                dur = float(getattr(sp, "t_ref", 0.0) or 0.0)
                sp.name = f"{dur:g}s_{float(p1):g}W_inferred"
            except Exception:
                sp.name = f"{nm}_P{float(p1):g}W"
    return specs

def _to_device(device: str) -> str:
    if device == "cuda" and torch.cuda.is_available():
        return "cuda:0"
    return device

def _as_float_list(s: str) -> List[float]:
    if not s:
        return []
    out: List[float] = []
    for part in str(s).replace(";", ",").split(","):
        part = part.strip().replace("_", ".")
        if not part:
            continue
        out.append(float(part))
    return out

def _script_dir() -> str:
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except Exception:
        return os.getcwd()

def _write_template_models_json(path: str):
    template = {
        "base_dir": DEFAULT_BASE_DIR,
        "out_dir": DEFAULT_OUT_DIR,
        "device": DEFAULT_DEVICE,
        "include_all": True,
        "times_frac": [0.5, 0.8, 1.0],
        "times_abs_s": [1.0],
        "roi_r_mm": 0.5,
        "surface_tol": 1e-6,
        "models": [
            "ours,stable,_,C:\\Users\\28739\\Laser\\out_pinn_v10\\checkpoints\\ckpt_final.pt",
            "data_only_mlp,suite,data_only_mlp,C:\\Users\\28739\\Laser\\横向1\\out_baseline_data_only_mlp\\checkpoints\\ckpt_final.pt",
            "mlp_pinn,suite,mlp_pinn,C:\\Users\\28739\\Laser\\横向1\\out_baseline_mlp_pinn\\checkpoints\\ckpt_final.pt",
            "ff_mlp_pinn,suite,ff_mlp_pinn,C:\\Users\\28739\\Laser\\横向1\\out_baseline_ff_mlp_pinn\\checkpoints\\ckpt_final.pt",
            "mlp_film_pinn,suite,mlp_film_pinn,C:\\Users\\28739\\Laser\\横向1\\out_baseline_mlp_film_pinn\\checkpoints\\ckpt_final.pt",
            "siren_nofilm_pinn,suite,siren_nofilm_pinn,C:\\Users\\28739\\Laser\\横向1\\out_baseline_siren_nofilm_pinn\\checkpoints\\ckpt_final.pt"
        ]
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)

def _maybe_load_models(entries: List[str], models_json_path: str) -> Tuple[List[str], Dict[str, Any]]:
    cfgj: Dict[str, Any] = {}
    if entries:
        return entries, cfgj

    if models_json_path and os.path.isfile(models_json_path):
        with open(models_json_path, "r", encoding="utf-8") as f:
            cfgj = json.load(f)
        return list(cfgj.get("models", [])), cfgj

    mj = os.path.join(_script_dir(), "models.json")
    if os.path.isfile(mj):
        with open(mj, "r", encoding="utf-8") as f:
            cfgj = json.load(f)
        return list(cfgj.get("models", [])), cfgj

    if DEFAULT_MODELS:
        return list(DEFAULT_MODELS), cfgj

    tpl = os.path.join(_script_dir(), "models.template.json")
    if not os.path.isfile(tpl):
        _write_template_models_json(tpl)
    print("\n[Config missing] No models specified.\n"
          f"- I created a template: {tpl}\n"
          "- Copy it to models.json, edit the ckpt paths, then run again.\n")
    raise SystemExit(2)


def import_module(which: str):
    which = which.lower().strip()
    if which == "suite":
        import problem6_baselines_suite as mod
        return mod
    if which == "stable":
        import stable_pinn_problem6_fixed_v4_ckpt as mod
        return mod
    raise ValueError(f"Unknown module={which}. Use 'suite' or 'stable'.")


def load_dataset_Tonly(spec, cache_dir: str, suite_mod):
    wtT = suite_mod.load_wide_table(spec.T_path, cache_dir)
    T = wtT.data[suite_mod.pick_var(wtT.data, ["T", "temperature"])]
    ds = SimpleNamespace()
    ds.coords = wtT.coords
    ds.times = wtT.times
    ds.T = T
    ds.spec = spec
    return ds


def load_dataset_safe(spec, cache_dir: str, suite_mod, prefer_full: bool=False):
    try:
        return suite_mod.load_dataset(spec, cache_dir)
    except Exception as e:
        if prefer_full:
            print(f"[Skip] full dataset load failed for {getattr(spec,'name','?')}: {e}")
            return None
        print(f"[Warn] full dataset load failed for {getattr(spec,'name','?')}: {e}\n"
              f"       -> fallback to T-only loader.")
        try:
            return load_dataset_Tonly(spec, cache_dir, suite_mod)
        except Exception as e2:
            print(f"[Skip] T-only load also failed for {getattr(spec,'name','?')}: {e2}")
            return None


def surface_masks_from_coords(coords: np.ndarray, tol: float=2e-5):
    z = coords[:, 2]
    zmax = float(np.nanmax(z))
    zmin = float(np.nanmin(z))
    front = (z >= (zmax - tol))
    back  = (z <= (zmin + tol))
    if front.sum() == 0:
        front = np.ones_like(z, dtype=bool)
    if back.sum() == 0:
        back = np.ones_like(z, dtype=bool)
    return front, back, zmax, zmin

def roi_mask_xy(xy: np.ndarray, x0: float, y0: float, r: float) -> np.ndarray:
    dx = xy[:, 0] - x0
    dy = xy[:, 1] - y0
    rr = np.sqrt(dx*dx + dy*dy)
    return (rr <= r)

def radial_profile(xy: np.ndarray, values: np.ndarray, x0: float, y0: float, r_max: float, nbins: int=64) -> pd.DataFrame:
    dx = xy[:, 0] - x0
    dy = xy[:, 1] - y0
    rr = np.sqrt(dx*dx + dy*dy)
    edges = np.linspace(0.0, r_max, nbins+1)
    centers = 0.5*(edges[:-1] + edges[1:])
    out = {"r_center_m": [], "mean": [], "std": [], "count": []}
    for i in range(nbins):
        m = (rr >= edges[i]) & (rr < edges[i+1])
        vv = values[m]
        out["r_center_m"].append(float(centers[i]))
        out["mean"].append(float(np.nanmean(vv)) if vv.size else float("nan"))
        out["std"].append(float(np.nanstd(vv)) if vv.size else float("nan"))
        out["count"].append(int(vv.size))
    return pd.DataFrame(out)


def interp_to_grid(xy: np.ndarray, v: np.ndarray, grid_n: int=256):
    x = xy[:, 0]
    y = xy[:, 1]
    tri = mtri.Triangulation(x, y)
    interp = mtri.LinearTriInterpolator(tri, v)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    gx = np.linspace(xmin, xmax, grid_n)
    gy = np.linspace(ymin, ymax, grid_n)
    X, Y = np.meshgrid(gx, gy)
    Vg = interp(X, Y)
    Vg = np.array(Vg, dtype=np.float64)
    return X, Y, Vg

def radial_spectrum_1d(Vgrid: np.ndarray, dx: float, dy: float, nbins: int=80):
    V = np.array(Vgrid, dtype=np.float64)
    mask = np.isfinite(V)
    if mask.sum() < 10:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    mean_val = np.nanmean(V)
    V = np.where(mask, V, mean_val)
    V = V - mean_val
    V = V * mask.astype(np.float64)
    F = np.fft.fft2(V)
    P = np.abs(F) ** 2
    P = np.fft.fftshift(P)
    ny, nx = V.shape
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx)) * 2*np.pi
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=dy)) * 2*np.pi
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX*KX + KY*KY)
    kmax = float(np.nanmax(K))
    edges = np.linspace(0.0, kmax, nbins+1)
    centers = 0.5*(edges[:-1] + edges[1:])
    Ek = np.zeros((nbins,), dtype=np.float64)
    for i in range(nbins):
        m = (K >= edges[i]) & (K < edges[i+1])
        vv = P[m]
        Ek[i] = float(np.nanmean(vv)) if vv.size else float("nan")
    return centers, Ek


class LoadedModel:
    def __init__(self, name: str, module_name: str, preset: str, ckpt_path: str, mod, cfg, model, scales: Dict[str, float]):
        self.name = name
        self.module_name = module_name
        self.preset = preset
        self.ckpt_path = ckpt_path
        self.mod = mod
        self.cfg = cfg
        self.model = model
        self.scales = scales


def load_model_entry(entry: str, device: str, strict: bool=True) -> LoadedModel:
    parts = [p.strip() for p in entry.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Model entry must be 'name,module,preset,ckpt'. Got: {entry}")
    name, module_name, preset, ckpt_path = parts
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    mod = import_module(module_name)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg_dict = ckpt.get("cfg", {})

    cfg = mod.CFG()
    for k, v in cfg_dict.items():
        if hasattr(cfg, k):
            try:
                setattr(cfg, k, v)
            except Exception:
                pass

    cfg.device = _to_device(device)
    if hasattr(cfg, "out_dir"):
        cfg.out_dir = os.path.dirname(os.path.dirname(ckpt_path))

    if module_name.lower() == "suite":
        if preset:
            mod.apply_model_preset(cfg, preset)
        model = mod.make_model(cfg)
    else:
        model = mod.AblationSIRENFiLM(cfg)

    model.load_state_dict(ckpt["model"], strict=bool(strict))
    model.to(cfg.device)
    model.eval()

    scales = ckpt.get("scales", {})
    scales = {str(k): float(v) for k, v in dict(scales).items()}
    return LoadedModel(name=name, module_name=module_name, preset=preset, ckpt_path=ckpt_path,
                       mod=mod, cfg=cfg, model=model, scales=scales)


@torch.no_grad()
def predict_T_on_mask(lm: LoadedModel, ds, const: Dict[str, torch.Tensor], mask_ids: np.ndarray, tval: float) -> np.ndarray:
    mod = lm.mod
    cfg = lm.cfg
    device = cfg.device

    xyz = torch.from_numpy(ds.coords[mask_ids]).to(device)
    t_ref = float(getattr(ds.spec, "t_ref", float(np.nanmax(ds.times))))
    t = torch.full((mask_ids.size, 1), float(tval), device=device)

    xytzn = mod.normalize_xyt(cfg, xyz, t, t_ref)
    scen = torch.from_numpy(mod.scenario_vec(ds.spec)[None, :]).to(device)
    out = lm.model.forward_phys(xytzn, scen.repeat(mask_ids.size, 1), const, lm.scales)
    Tp = out["T"].detach().float().cpu().numpy().reshape(-1)
    return Tp


def save_tricontour(xy: np.ndarray, values: np.ndarray, title: str, out_png: str, levels: int=60):
    tri = mtri.Triangulation(xy[:, 0], xy[:, 1])
    plt.figure()
    plt.tricontourf(tri, values, levels=levels)
    plt.gca().set_aspect("equal")
    plt.title(title)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def fig_spectrum_true_front_vs_back(ds, front_ids: np.ndarray, back_ids: np.ndarray,
                                    tval: float, out_dir_fig: str, out_dir_data: str,
                                    grid_n: int=256, nbins: int=90, tag: str=""):
    xy_f = ds.coords[front_ids][:, :2]
    Tf = ds.T[front_ids, int(np.argmin(np.abs(ds.times - tval)))]
    Xf, Yf, Gf = interp_to_grid(xy_f, Tf, grid_n=grid_n)
    dxf = float(Xf[0, 1] - Xf[0, 0])
    dyf = float(Yf[1, 0] - Yf[0, 0])
    kf, Ef = radial_spectrum_1d(Gf, dxf, dyf, nbins=nbins)

    xy_b = ds.coords[back_ids][:, :2]
    Tb = ds.T[back_ids, int(np.argmin(np.abs(ds.times - tval)))]
    Xb, Yb, Gb = interp_to_grid(xy_b, Tb, grid_n=grid_n)
    dxb = float(Xb[0, 1] - Xb[0, 0])
    dyb = float(Yb[1, 0] - Yb[0, 0])
    kb, Eb = radial_spectrum_1d(Gb, dxb, dyb, nbins=nbins)

    data_path = os.path.join(out_dir_data, f"spectrum_true_front_vs_back{tag}.npz")
    np.savez_compressed(
        data_path,
        tval=np.array([tval], dtype=np.float64),
        k_front=kf, E_front=Ef,
        k_back=kb, E_back=Eb,
        dataset=np.array([str(getattr(ds.spec,'name',''))], dtype=object),
    )

    out_png = os.path.join(out_dir_fig, f"Fig_Spectrum_True_FrontVsBack{tag}.png")
    plt.figure()
    if kf.size:
        plt.semilogy(kf, Ef, label="Front (True)")
    if kb.size:
        plt.semilogy(kb, Eb, label="Back (True)")
    plt.title(f"True radial spectrum E(k) @ t={tval:.3f}s")
    plt.xlabel("k (rad/m)")
    plt.ylabel("E(k) (a.u.)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=240)
    plt.close()
    return out_png, data_path


def fig_surface_compare(models: List[LoadedModel], ds, surface: str,
                        mask_ids: np.ndarray, tval: float,
                        const_cpu: Dict[str, torch.Tensor],
                        out_dir_fig: str, out_dir_data: str, tag: str="",
                        levels: int=60, roi_r: float=0.5e-3, r_bins: int=60,
                        time_label: str=""):
    surface = surface.lower().strip()
    idx_t = int(np.argmin(np.abs(ds.times - tval)))
    t_plot = float(ds.times[idx_t])

    T_true = ds.T[mask_ids, idx_t].reshape(-1)
    xy = ds.coords[mask_ids][:, :2]

    x0 = float(getattr(ds.spec, "x0", 0.0))
    y0 = float(getattr(ds.spec, "y0", 0.0))
    roi = roi_mask_xy(xy, x0, y0, roi_r)

    true_png = os.path.join(out_dir_fig, f"Fig_{surface}_TrueMap_{time_label}t{t_plot:.3f}{tag}.png")
    save_tricontour(xy, T_true, f"{ds.spec.name} {surface} TRUE T @ t={t_plot:.3f}s", true_png, levels=levels)

    rows = []
    df_true = radial_profile(xy, T_true, x0, y0, r_max=max(roi_r*2.5, 2e-3), nbins=r_bins)
    df_true["model"] = "TRUE"
    df_true["surface"] = surface
    df_true["t_s"] = float(t_plot)
    df_true["dataset"] = str(ds.spec.name)
    prof_rows = [df_true]

    for lm in models:
        const = {k: v.to(lm.cfg.device) for k, v in const_cpu.items()}
        Tp = predict_T_on_mask(lm, ds, const, mask_ids, t_plot)

        err = Tp - T_true
        rows.append({
            "dataset": str(ds.spec.name),
            "surface": surface,
            "t_s": t_plot,
            "P_W": float(getattr(ds.spec, "P_laser", float("nan"))),
            "t_ref_s": float(getattr(ds.spec, "t_ref", float("nan"))),
            "model": lm.name,
            "mse": float(np.nanmean(err**2)),
            "mae": float(np.nanmean(np.abs(err))),
            "max_abs_err": float(np.nanmax(np.abs(err))),
            "roi_r_m": float(roi_r),
            "roi_mse": float(np.nanmean((err[roi])**2)) if roi.any() else float("nan"),
            "roi_mae": float(np.nanmean(np.abs(err[roi]))) if roi.any() else float("nan"),
            "time_request_s": float(tval),
            "time_used_s": float(t_plot),
        })

        pred_png = os.path.join(out_dir_fig, f"Fig_{surface}_{lm.name}_Pred_{time_label}t{t_plot:.3f}{tag}.png")
        err_png  = os.path.join(out_dir_fig, f"Fig_{surface}_{lm.name}_Err_{time_label}t{t_plot:.3f}{tag}.png")
        save_tricontour(xy, Tp, f"{ds.spec.name} {surface} {lm.name} PRED T @ t={t_plot:.3f}s", pred_png, levels=levels)
        save_tricontour(xy, err, f"{ds.spec.name} {surface} {lm.name} ERROR (Pred-True) @ t={t_plot:.3f}s", err_png, levels=levels)

        dfp = radial_profile(xy, Tp, x0, y0, r_max=max(roi_r*2.5, 2e-3), nbins=r_bins)
        dfp["model"] = lm.name
        dfp["surface"] = surface
        dfp["t_s"] = float(t_plot)
        dfp["dataset"] = str(ds.spec.name)
        prof_rows.append(dfp)

        npz_path = os.path.join(out_dir_data, f"surface_{surface}_{lm.name}_{time_label}t{t_plot:.3f}{tag}.npz")
        np.savez_compressed(
            npz_path,
            xy=xy.astype(np.float64),
            T_true=T_true.astype(np.float64),
            T_pred=Tp.astype(np.float64),
            err=err.astype(np.float64),
            roi_mask=roi.astype(np.int8),
            t_s=np.array([t_plot], dtype=np.float64),
            t_req=np.array([tval], dtype=np.float64),
            dataset=np.array([str(ds.spec.name)], dtype=object),
            model=np.array([str(lm.name)], dtype=object),
            surface=np.array([str(surface)], dtype=object),
        )

    dfm = pd.DataFrame(rows)
    metrics_csv = os.path.join(out_dir_data, f"metrics_surface_{surface}_{time_label}t{t_plot:.3f}{tag}.csv")
    dfm.to_csv(metrics_csv, index=False, encoding="utf-8-sig")

    dfprof = pd.concat(prof_rows, axis=0, ignore_index=True)
    prof_csv = os.path.join(out_dir_data, f"radial_profiles_{surface}_{time_label}t{t_plot:.3f}{tag}.csv")
    dfprof.to_csv(prof_csv, index=False, encoding="utf-8-sig")

    out_png = os.path.join(out_dir_fig, f"Fig_{surface}_RadialProfile_{time_label}t{t_plot:.3f}{tag}.png")
    plt.figure()
    for mname in ["TRUE"] + [lm.name for lm in models]:
        dfi = dfprof[dfprof["model"] == mname]
        if dfi.empty:
            continue
        plt.plot(dfi["r_center_m"].values, dfi["mean"].values, label=mname)
    plt.title(f"{ds.spec.name} {surface} radial profile @ t={t_plot:.3f}s")
    plt.xlabel("r (m)")
    plt.ylabel("T mean (K)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=240)
    plt.close()

    return {
        "true_map_png": true_png,
        "metrics_csv": metrics_csv,
        "profiles_csv": prof_csv,
        "radial_png": out_png,
    }


def _requested_times(ds, times_frac: List[float], times_abs: List[float]) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    t_ref = float(getattr(ds.spec, "t_ref", float(np.nanmax(ds.times))))
    for frac in times_frac:
        out.append((f"tfrac{frac:.2f}_", float(frac * t_ref)))
    for t in times_abs:
        out.append((f"tabs{t:.3f}_", float(t)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, default="", help="COMSOL data root folder")
    ap.add_argument("--java_path", type=str, default="", help="COMSOL exported java path (optional)")
    ap.add_argument("--out_dir", type=str, default="", help="output folder")
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--include_all", action="store_true", help="include all discovered datasets")
    ap.add_argument("--prefer_full_exports", action="store_true", help="skip datasets that cannot be fully loaded (no T-only fallback)")
    ap.add_argument("--models_json", type=str, default="", help="JSON file with a list of model entries")
    ap.add_argument("--model", action="append", default=[], help="Model entry: name,module,preset,ckpt_path (repeatable)")

    ap.add_argument("--powers", type=str, default="", help="filter by laser powers in W, e.g. '1,5,10.503,15,18,20,25,30' (optional)")
    ap.add_argument("--splits", type=str, default="all", help="which dataset splits to include: 'all' or comma-separated among train,val,test")
    ap.add_argument("--max_specs", type=int, default=0, help="optional cap on number of datasets to process (0=no cap)")
    ap.add_argument("--default_power", type=float, default=None, help="if a dataset filename lacks a ...W power tag, use this default power (W) for discovery")
    ap.add_argument("--representative_only", action="store_true", help="use old behavior: only a small representative subset of datasets")
    ap.add_argument("--no_ckpt_strict", action="store_true", help="disable strict state_dict loading")

    ap.add_argument("--roi_r_mm", type=float, default=0.5, help="ROI radius in mm")
    ap.add_argument("--grid_n", type=int, default=256, help="FFT/interp grid resolution")
    ap.add_argument("--spec_nbins", type=int, default=90, help="Radial spectrum bins")
    ap.add_argument("--times_frac", type=str, default="0.5,0.8,1.0", help="fractions of t_ref (comma-separated)")
    ap.add_argument("--times_abs", type=str, default="", help="absolute times in seconds (comma-separated), e.g. '1,5,10'")
    ap.add_argument("--surface_tol", type=float, default=2e-5, help="surface z tolerance")
    args = ap.parse_args(argv)

    entries = list(args.model)
    entries, cfgj = _maybe_load_models(entries, args.models_json)

    base_dir = args.base_dir or cfgj.get("base_dir") or DEFAULT_BASE_DIR
    out_dir  = args.out_dir  or cfgj.get("out_dir")  or (DEFAULT_OUT_DIR + "_" + _now_tag())
    device   = args.device   or cfgj.get("device")   or DEFAULT_DEVICE
    include_all = bool(args.include_all or cfgj.get("include_all", DEFAULT_INCLUDE_ALL))

    roi_r = float(cfgj.get("roi_r_mm", args.roi_r_mm)) * 1e-3
    grid_n = int(cfgj.get("grid_n", args.grid_n))
    spec_nbins = int(cfgj.get("spec_nbins", args.spec_nbins))

    times_frac = _as_float_list(args.times_frac) if args.times_frac else list(cfgj.get("times_frac", [0.5, 0.8, 1.0]))
    times_abs = _as_float_list(args.times_abs) if args.times_abs else list(cfgj.get("times_abs_s", []))

    surface_tol = float(cfgj.get("surface_tol", args.surface_tol))

    out_dir = ensure_dir(out_dir)
    out_fig = ensure_dir(os.path.join(out_dir, "figs"))
    out_data = ensure_dir(os.path.join(out_dir, "data"))
    cache_dir = ensure_dir(os.path.join(out_dir, "_cache_wt"))

    device = _to_device(device)

    ckpt_strict = (not bool(args.no_ckpt_strict))
    models = [load_model_entry(e, device=device, strict=ckpt_strict) for e in entries]

    suite = import_module("suite")
    cfgD = suite.CFG()
    cfgD.base_dir = base_dir
    if args.java_path:
        cfgD.java_path = args.java_path
    cfgD.device = "cpu"
    if include_all:
        cfgD.keep_only_listed_pairs = False

    # Optional default power for datasets whose filenames omit the '...W' tag (your 10.503W special case).
    default_power = cfgj.get("default_power", None)
    if args.default_power is not None:
        default_power = float(args.default_power)
    if default_power is not None:
        try:
            setattr(cfgD, "default_power", float(default_power))
        except Exception:
            pass

    specs = suite.discover_datasets(cfgD)
    specs = _fix_missing_power_specs(specs, default_power=default_power)
    params = suite.parse_comsol_java_params(cfgD.java_path) if (cfgD.java_path and os.path.isfile(cfgD.java_path)) else {}

        # ---- dataset selection ----
    def _parse_splits(s: str) -> List[str]:
        s = (s or "").strip().lower()
        if (not s) or (s == "all"):
            return ["train", "val", "test"]
        parts = [p.strip().lower() for p in str(s).replace(";", ",").split(",") if p.strip()]
        out: List[str] = []
        for p in parts:
            if p in ("train", "val", "test"):
                out.append(p)
        return out or ["train", "val", "test"]

    # powers filter from CLI or models.json (optional)
    powers: List[float] = []
    if args.powers:
        powers = _as_float_list(args.powers)
    else:
        pj = cfgj.get("powers", [])
        if isinstance(pj, (list, tuple)):
            try:
                powers = [float(str(x).replace("_", ".")) for x in pj]
            except Exception:
                powers = []

    split_allow = set(_parse_splits(args.splits or cfgj.get("splits", "all")))

    def _match_power(P: float) -> bool:
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

    # apply filters
    specs_f: List[Any] = []
    for sp in specs:
        s = str(getattr(sp, "split", "train")).lower()
        if s not in split_allow:
            continue
        if not _match_power(getattr(sp, "P_laser", float("nan"))):
            continue
        specs_f.append(sp)

    # stable ordering: by power -> t_ref -> name
    def _sort_key(sp):
        P = float(getattr(sp, "P_laser", 0.0))
        tr = getattr(sp, "t_ref", None)
        tr = float(tr) if (tr is not None) else 0.0
        return (P, tr, str(getattr(sp, "name", "")))

    specs_f.sort(key=_sort_key)

    representative_only = bool(args.representative_only or cfgj.get("representative_only", False))

    fig_specs: List[Any] = []
    if representative_only:
        specs_by_split: Dict[str, List[Any]] = {"train": [], "val": [], "test": []}
        for sp in specs_f:
            s = str(getattr(sp, "split", "train")).lower()
            specs_by_split.setdefault(s, []).append(sp)

        # old behavior: pick a small subset (val first)
        if specs_by_split.get("val"):
            fig_specs += specs_by_split["val"][:2]
        if not fig_specs and specs_by_split.get("train"):
            fig_specs += specs_by_split["train"][:2]
        if include_all and len(specs_f) > 3:
            fig_specs += specs_f[2:3]
    else:
        # default: process ALL filtered datasets (covers all powers)
        fig_specs = list(specs_f)

    max_specs = int(args.max_specs or cfgj.get("max_specs", 0) or 0)
    if max_specs > 0:
        fig_specs = fig_specs[:max_specs]

    if not fig_specs:
        raise ValueError("No datasets selected after filtering. Check --powers/--splits and discovery results.")

    const_by_name: Dict[str, Dict[str, torch.Tensor]] = {}
    for sp in fig_specs:
        const_cpu = suite.build_const(suite.CFG(), sp, params, device="cpu")
        const_by_name[str(sp.name)] = {k: v.to("cpu") for k, v in const_cpu.items()}

    manifest: Dict[str, Any] = {
        "base_dir": base_dir,
        "out_dir": out_dir,
        "device": device,
        "include_all": include_all,
        "prefer_full_exports": bool(args.prefer_full_exports),
        "times_frac": times_frac,
        "times_abs_s": times_abs,
        "models": [{"name": m.name, "module": m.module_name, "preset": m.preset, "ckpt": m.ckpt_path} for m in models],
        "generated": {},
    }

    for sp in fig_specs:
        ds = load_dataset_safe(sp, cache_dir, suite, prefer_full=bool(args.prefer_full_exports))
        if ds is None:
            continue

        front_mask, back_mask, *_ = surface_masks_from_coords(ds.coords, tol=surface_tol)
        front_ids = np.where(front_mask)[0]
        back_ids  = np.where(back_mask)[0]

        tag = f"_{_safe_name(ds.spec.name)}"

        t_ref = float(getattr(ds.spec, "t_ref", float(np.nanmax(ds.times))))
        t_spec = float((times_frac[-1] if times_frac else 1.0) * t_ref)
        idx_ts = int(np.argmin(np.abs(ds.times - t_spec)))
        t_plot = float(ds.times[idx_ts])

        png, npz = fig_spectrum_true_front_vs_back(ds, front_ids, back_ids, t_plot,
                                                   out_dir_fig=out_fig, out_dir_data=out_data,
                                                   grid_n=grid_n, nbins=spec_nbins, tag=tag)

        manifest["generated"].setdefault(ds.spec.name, {})
        manifest["generated"][ds.spec.name]["true_spectrum_front_vs_back"] = {"png": png, "npz": npz}

        const_cpu = const_by_name.get(ds.spec.name, suite.build_const(suite.CFG(), ds.spec, params, device="cpu"))

        tmin = float(np.min(ds.times))
        tmax = float(np.max(ds.times))
        for time_label, t_req in _requested_times(ds, times_frac=times_frac, times_abs=times_abs):
            if (t_req < tmin - 1e-9) or (t_req > tmax + 1e-9):
                print(f"[Warn] Requested time {t_req:.6f}s is outside ds.times range [{tmin:.6f},{tmax:.6f}] for {ds.spec.name}. Skip.")
                continue

            out = fig_surface_compare(models, ds, "front", front_ids, t_req, const_cpu,
                                      out_dir_fig=out_fig, out_dir_data=out_data, tag=tag, roi_r=roi_r,
                                      time_label=time_label)
            manifest["generated"][ds.spec.name][f"front_compare_{time_label}{t_req:.6f}"] = out

            out = fig_surface_compare(models, ds, "back", back_ids, t_req, const_cpu,
                                      out_dir_fig=out_fig, out_dir_data=out_data, tag=tag, roi_r=roi_r,
                                      time_label=time_label)
            manifest["generated"][ds.spec.name][f"back_compare_{time_label}{t_req:.6f}"] = out

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("[Done] Figures saved to:", out_fig)
    print("[Done] Data saved to:", out_data)
    print("[Done] Manifest:", manifest_path)



def run_in_notebook(models_json: str = "models.json",
                    base_dir: str = "",
                    out_dir: str = "",
                    device: str = "",
                    times_abs: str = "",
                    times_frac: str = "",
                    powers: str = "",
                    splits: str = "all",
                    include_all: bool = True,
                    representative_only: bool = False,
                    max_specs: int = 0,
                    prefer_full_exports: bool = False,
                    java_path: str = "",
                    default_power: float | None = None,
                    no_ckpt_strict: bool = False):
    """Convenience wrapper for JupyterLab.

    Example:
        from make_problem6_paper_figures_v5 import run_in_notebook
        run_in_notebook(models_json="models.json", times_abs="1", powers="1,5,10.503,15,18,20,25,30", default_power=10.503)
    """
    argv = []
    if default_power is not None: argv += ["--default_power", str(default_power)]
    if base_dir: argv += ["--base_dir", base_dir]
    if java_path: argv += ["--java_path", java_path]
    if out_dir:  argv += ["--out_dir", out_dir]
    if device:   argv += ["--device", device]
    if models_json: argv += ["--models_json", models_json]
    if times_abs: argv += ["--times_abs", times_abs]
    if times_frac: argv += ["--times_frac", times_frac]
    if powers: argv += ["--powers", powers]
    if splits: argv += ["--splits", splits]
    if max_specs and int(max_specs) > 0: argv += ["--max_specs", str(int(max_specs))]
    if include_all: argv += ["--include_all"]
    if representative_only: argv += ["--representative_only"]
    if prefer_full_exports: argv += ["--prefer_full_exports"]
    if no_ckpt_strict: argv += ["--no_ckpt_strict"]
    return main(argv)


if __name__ == "__main__":
    main()

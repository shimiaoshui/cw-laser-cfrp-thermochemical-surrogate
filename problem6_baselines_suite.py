
# coding: utf-8
"""
Stable Parametric PINN Surrogate for COMSOL Laser Ablation (multi-dataset)  [v2]

What this fixes vs 问题1.txt:
1) Uses COMSOL-consistent closures from the exported Java:
   - Abs_eff = A_abs_v*(1-alphap01) + A_abs_c*alphap01
   - Q_pyro/Q_ox_* and Q_total closure from dalpha_dt
   - rho_eff, cp_eff, k_in, k_z as alpha-dependent material properties
2) Replaces "Q_total as a free head" with physics closure (still supervised if you have Q_total CSV).
3) Adds ODE consistency: d(alpha)/dt == dalpha_dt (autograd) and optionally supervised dalpha_dt.
4) Stops physics weights collapsing to ~0:
   - Uses LRA-style adaptive weighting with a non-trivial floor (min_phys, min_bc)
   - Physics residuals are better scaled
5) Adds causal time curriculum + simple RAR pool for collocation points.

Dependencies: numpy, pandas, torch, matplotlib
Tested: PyTorch 2.x

USAGE:
- Edit CFG.base_dir to your COMSOL data folder.
- (Optional) edit CFG.datasets or use discover_datasets().
- Run: python stable_pinn_ablation_v2.py

Outputs:
- checkpoints/*.pt
- scales.json, train_history.json
- evaluation figures in out_dir/eval_*
"""

from __future__ import annotations
import os, re, math, json, random, hashlib, time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "CFG", "DatasetSpec", "AblationSIRENFiLM", "normalize_xyt", "scenario_vec",
    "load_wide_table", "load_dataset", "discover_datasets", "train", "main"
]


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# ============================================================
# 0) Utils
# ============================================================

def set_seed(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def save_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def robust_center_scale(x: np.ndarray, q: float = 0.995, floor: float = 1e-9) -> Tuple[float, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 1.0
    c = float(np.quantile(x, 0.5))
    dev = np.abs(x - c)
    s = float(np.quantile(dev, q))
    return c, max(s, floor)

def robust_scale_abs(x: np.ndarray, q: float = 0.999, floor: float = 1e-9) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 1.0
    s = float(np.quantile(np.abs(x), q))
    return max(s, floor)

def strip_units(expr: str) -> str:
    """
    COMSOL exports literals like 1e-6[kg/m^3], 110.53[kJ/mol].
    We:
      - keep the numeric part
      - convert kJ -> J (multiply 1e3) for the two common patterns kJ/mol and kJ/kg
      - otherwise remove [...] without scaling (safe default)
    """
    s = expr.strip()
    # handle kJ -> J for simple literals
    m = re.fullmatch(r"\s*([-+0-9.eE]+)\s*\[\s*kJ\s*/\s*(mol|kg)\s*\]\s*", s)
    if m:
        val = float(m.group(1)) * 1e3
        return f"{val}"
    # remove any [ ... ] unit annotations after numeric literals
    s = re.sub(r"([-+0-9.eE]+)\s*\[[^\]]+\]", r"\1", s)
    return s

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

# ============================================================
# 1) Wide-table loader (robust time mapping)
# ============================================================

_TIME_RE = re.compile(r"^(?P<var>.+?)\s*(?:\([^)]*\))?\s*@\s*t\s*=\s*(?P<t>[-+0-9.eE]+)\s*$")

def _rt(t: float, nd: int = 8) -> float:
    return float(round(float(t), nd))


def align_by_time(arr: "np.ndarray", times_src: "np.ndarray", times_ref: "np.ndarray") -> "np.ndarray":
    """Align arr[:, :] from times_src onto times_ref by nearest match (atol=1e-6).

    COMSOL headers often contain times like 0.1, but float32 storage can shift it to 0.10000000149.
    We therefore round times and align by nearest match.
    """
    if arr.ndim != 2:
        raise ValueError(f"align_by_time expects 2D array, got shape={arr.shape}")
    src_keys = [_rt(float(t)) for t in times_src.tolist()]
    ref_keys = [_rt(float(t)) for t in times_ref.tolist()]
    src_map = {k: i for i, k in enumerate(src_keys)}
    out = np.full((arr.shape[0], len(ref_keys)), np.nan, dtype=arr.dtype)
    src_arr = np.array(src_keys, dtype=np.float64)
    for j, k in enumerate(ref_keys):
        i = src_map.get(k, None)
        if i is None:
            nearest = int(np.argmin(np.abs(src_arr - float(k))))
            if not np.isclose(src_arr[nearest], float(k), rtol=0.0, atol=1e-6):
                raise KeyError(f"Cannot align time {k}; nearest={src_arr[nearest]} (src)")
            i = nearest
        out[:, j] = arr[:, i]
    return out


def _coord_keys(coords: "np.ndarray", eps: float = 1e-9) -> "np.ndarray":
    """Quantize coordinates to integer keys for robust matching.
    eps=1e-9 m is typically safe for COMSOL exports written in mm with ~1e-6 mm precision.
    """
    c = np.asarray(coords, dtype=np.float64)
    k = np.rint(c / float(eps)).astype(np.int64)
    return k

def reindex_by_coords(arr: "np.ndarray", coords_src: "np.ndarray", coords_ref: "np.ndarray",
                      name: str, eps: float = 1e-9, max_missing_frac: float = 1e-3) -> "np.ndarray":
    """Reindex arr (Nsrc, Nt[, ...]) onto coords_ref (Nref,3) by matching quantized xyz keys.
    If too many points are missing, raise ValueError with a COMSOL-export hint.
    """
    if coords_src.shape[0] == coords_ref.shape[0]:
        # quick path: if shapes match, accept (even if order differs; order check is expensive)
        return arr

    # Build map from key->index for src
    ks = _coord_keys(coords_src, eps=eps)
    # use structured dtype for fast hashing
    ks_view = ks.view([("x", np.int64), ("y", np.int64), ("z", np.int64)])
    src_map = {tuple(row): i for i, row in enumerate(ks)}

    kr = _coord_keys(coords_ref, eps=eps)
    idx = np.full((coords_ref.shape[0],), -1, dtype=np.int64)
    missing = 0
    for i, row in enumerate(kr):
        j = src_map.get((int(row[0]), int(row[1]), int(row[2])), -1)
        idx[i] = j
        if j < 0:
            missing += 1

    missing_frac = missing / float(coords_ref.shape[0])
    if missing_frac > max_missing_frac:
        raise ValueError(
            f"[CoordsMismatch] {name}: Nsrc={coords_src.shape[0]} cannot be mapped to Nref={coords_ref.shape[0]} "
            f"(missing={missing} = {missing_frac:.3%}).\n"
            f"Likely you exported different 'Data set' / selection / resolution for this quantity in COMSOL. "
            f"Ensure ALL exported CSVs use the SAME dataset (e.g., same Solution), SAME evaluation (same mesh nodes), "
            f"and SAME selection. Then delete cache_npz and rerun."
        )

    # Build output with same trailing dims
    out_shape = (coords_ref.shape[0],) + arr.shape[1:]
    out = np.full(out_shape, np.nan, dtype=arr.dtype)
    good = idx >= 0
    out[good] = arr[idx[good]]
    return out

def choose_reference_coords(tables: Dict[str, "WideTable"]) -> Tuple[str, "np.ndarray"]:
    """Choose the most common node count (mode) across tables as reference, to avoid mixing plot-grids with mesh nodes."""
    Ns = [(k, v.coords.shape[0]) for k, v in tables.items() if v is not None and getattr(v, "coords", None) is not None]
    if not Ns:
        raise ValueError("No tables provided to choose_reference_coords")
    # mode of N; tie-break by larger N (usually the real mesh nodes vs tiny boundary subset)
    from collections import Counter
    cnt = Counter([n for _, n in Ns])
    modeN = max(cnt.items(), key=lambda kv: (kv[1], kv[0]))[0]
    # prefer v_sub if matches, else T, else first match
    for pref in ["v_sub", "T", "v_mesh", "Q", "qbnd", "rate"]:
        if pref in tables and tables[pref].coords.shape[0] == modeN:
            return pref, tables[pref].coords
    for k, n in Ns:
        if n == modeN:
            return k, tables[k].coords
    return Ns[0][0], tables[Ns[0][0]].coords

def _detect_delim(line: str) -> str:
    cands = [",", "\t", ";"]
    best, bestn = ",", 0
    for d in cands:
        n = len(line.split(d))
        if n > bestn:
            bestn, best = n, d
    return best

def _parse_length_unit(lines: List[str]) -> str:
    for ln in lines[:80]:
        if "Length unit" in ln:
            parts = re.split(r"[\t, ]+", ln.strip().lstrip("%").strip())
            if len(parts) >= 3:
                return parts[-1]
    return "m"

def _unit_scale(u: str) -> float:
    u = u.strip().lower()
    if u == "m": return 1.0
    if u == "mm": return 1e-3
    if u in ["um", "µm"]: return 1e-6
    if u == "nm": return 1e-9
    return 1.0

@dataclass
class WideTable:
    coords: np.ndarray          # (N,3) meters
    times: np.ndarray           # (Nt,)
    data: Dict[str, np.ndarray] # var -> (N,Nt)

def load_wide_table(path: str, cache_dir: str) -> WideTable:
    """
    Load a COMSOL "wide-table" export (X,Y,Z + var @ t=... columns) and cache as NPZ.

    Critical note:
    COMSOL metadata + header lines usually start with '%'. Therefore we must NOT pass
    comment='%' when we still need the header row. We explicitly parse the commented
    header, then read numeric rows starting after it.
    """
    ensure_dir(cache_dir)

    # cache key must change when file changes
    st = os.stat(path)
    key = md5(f"{os.path.abspath(path)}|{st.st_mtime_ns}|{st.st_size}")
    npz_path = os.path.join(cache_dir, f"wide_{key}.npz")

    if os.path.exists(npz_path):
        npz = np.load(npz_path, allow_pickle=True)
        coords = npz["coords"]
        times = npz["times"]
        keys = list(npz["keys"])
        arrs = npz["arrs"]
        data = {k: arrs[i] for i, k in enumerate(keys)}
        return WideTable(coords=coords, times=times, data=data)

    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        lines = f.readlines()

    length_unit = _parse_length_unit(lines)
    scale = _unit_scale(length_unit)

    # ---- find the commented header line: '% X Y Z ...'
    header_idx = None
    for i, ln in enumerate(lines[:400]):  # COMSOL headers are near the top
        s = ln.strip()
        if not s.startswith("%"):
            continue
        s2 = s.lstrip("%").strip()
        # allow comma/tab/space/semicolon separation
        if re.match(r"^X[\t ,;]+Y[\t ,;]+Z[\t ,;]+", s2):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(
            f"Header line like '% X Y Z ...' not found in: {path}. "
            "Please export as a wide table that includes columns such as 'T (K) @ t=0'."
        )

    header_line = lines[header_idx].lstrip("%").strip()
    delim = _detect_delim(header_line)
    cols = [c.strip() for c in header_line.split(delim)]
    if len(cols) < 4:
        # whitespace-delimited exports (or header contains spaces)
        cols = re.split(r"\s+", header_line.strip())
        delim = None

    # ---- parse (var, time) for each data column
    parsed: List[Tuple[str, float, int]] = []  # (var, t, column_index)
    times_set = set()
    for j in range(3, len(cols)):
        name = cols[j]
        m = _TIME_RE.match(str(name))
        if m:
            var = m.group("var").strip()
            t = _rt(float(m.group("t")))
            parsed.append((var, t, j))
            times_set.add(t)
        else:
            # If it doesn't match '@ t=', treat as a single-time column at t=0
            var = str(name).strip()
            t = _rt(0.0)
            parsed.append((var, t, j))
            times_set.add(t)

    if len(parsed) == 0:
        raise ValueError(
            "Parsed 0 data columns from header. This usually means the file is not a COMSOL wide-table export, "
            "or the header is being parsed incorrectly.\n"
            f"Header sample: {cols[:12]}\n"
            "Expected something like: 'T (K) @ t=0', 'T (K) @ t=0.1', ..."
        )

    # Build a stable time grid in Python float (rounded), THEN store as float32 for arrays.
    # Do NOT use float32 values as dict keys: 0.1 -> 0.10000000149 can cause KeyError (float precision).
    times_list = [ _rt(t) for t in sorted(times_set) ]  # stable python floats
    times = np.array(times_list, dtype=np.float32)      # stored for downstream use
    t_to_idx = {t: i for i, t in enumerate(times_list)}

    # ---- read numeric table starting after the header line
    skiprows = header_idx + 1
    if delim is None:
        df = pd.read_csv(
            path,
            skiprows=skiprows,
            header=None,
            delim_whitespace=True,
            comment="%",
            engine="c",
            dtype=np.float32,
        )
    else:
        df = pd.read_csv(
            path,
            skiprows=skiprows,
            header=None,
            sep=delim,
            comment="%",
            engine="c",
            dtype=np.float32,
        )

    arr = df.to_numpy(dtype=np.float32, copy=False)
    if arr.shape[1] < 4:
        raise ValueError(
            f"Parsed numeric table has too few columns: shape={arr.shape} from {path}. "
            "Check delimiter and export settings (comma/tab/space). "
            f"Detected header delimiter='{delim}'."
        )

    coords = (arr[:, :3].astype(np.float32, copy=False) * np.float32(scale))

    # ---- assemble var -> (N,Nt)
    data: Dict[str, np.ndarray] = {}
    N = coords.shape[0]
    Nt = times.size
    for var, t, j in parsed:
        if var not in data:
            data[var] = np.full((N, Nt), np.nan, dtype=np.float32)
        if j >= arr.shape[1]:
            # header had more columns than pandas parsed; skip silently but keep NaNs
            continue
        # stable key + nearest fallback within tolerance
        tk = _rt(t)
        idx_t = t_to_idx.get(tk, None)
        if idx_t is None:
            # fallback: nearest time index (handles rare header formatting differences)
            tt = np.array(list(t_to_idx.keys()), dtype=np.float64)
            k = float(tk)
            nearest = int(np.argmin(np.abs(tt - k)))
            if not np.isclose(tt[nearest], k, rtol=0.0, atol=1e-6):
                raise KeyError(
                    f"Time key {k} not found in time grid (nearest={tt[nearest]}). "
                    "This is likely a header parsing issue; please check your @ t=... columns."
                )
            idx_t = nearest
        data[var][:, idx_t] = arr[:, j]

    # cache (same format as existing v3 cache)
    keys = np.array(list(data.keys()), dtype=object)
    if keys.size == 0:
        raise ValueError(
            f"No variables were loaded from {path} (data dict empty). "
            "Check that your export includes at least one expression column. "
            f"Header sample: {cols[:12]}"
        )
    arrs = np.stack([data[k] for k in keys], axis=0)
    np.savez_compressed(npz_path, coords=coords, times=times, keys=keys, arrs=arrs)
    return WideTable(coords=coords, times=times, data=data)

def _canon_key(s: str) -> str:
    # aggressively normalize COMSOL-exported expression names:
    #   "q_sub (W/m^2)" -> "qsub", "ht.hf1.q0" -> "hthf1q0"
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

def pick_var(data: Dict[str, np.ndarray], keys: List[str]) -> str:
    """
    Robust variable name resolver for COMSOL wide-table exports.

    Why needed:
      - Wide tables often include units or extra suffixes, and boundary exports can rename columns
        (e.g., q_sub, qsub, ht.hf1.q0, ...). We canonicalize names before matching.

    Strategy:
      1) case-insensitive exact match
      2) canonical exact match (strip non-alnum)
      3) canonical substring match (candidate in key)
      4) canonical reverse-substring (key in candidate), for very short candidate aliases
    """
    ks = list(data.keys())
    if len(ks) == 0:
        raise KeyError("pick_var(): data dict is empty.")

    low = [k.lower() for k in ks]
    canon = [_canon_key(k) for k in ks]

    # 1) case-insensitive exact match
    for cand in keys:
        c = str(cand).lower()
        if c in low:
            return ks[low.index(c)]

    # 2) canonical exact match
    for cand in keys:
        c = _canon_key(cand)
        if c in canon:
            return ks[canon.index(c)]

    # 3) canonical substring match
    for cand in keys:
        c = _canon_key(cand)
        if not c:
            continue
        for i, ck in enumerate(canon):
            if c in ck:
                return ks[i]

    # 4) reverse-substring (helps if keys include longer tokens and candidate is short)
    for cand in keys:
        c = _canon_key(cand)
        if not c:
            continue
        for i, ck in enumerate(canon):
            if ck and ck in c:
                return ks[i]

    raise KeyError(f"Cannot find any of {keys} in keys[0:12]={ks[:12]} ... total={len(ks)}")


# ============================================================
# 2) Parse COMSOL Java export for key constants
# ============================================================

_PARAM_RE = re.compile(r'model\.param\(\)\.set\("(?P<k>[^"]+)",\s*"(?P<v>[^"]*)"\s*\)\s*;')

def parse_comsol_java_params(java_path: str) -> Dict[str, float]:
    """
    Extract model.param().set("name","value") from COMSOL java export.
    Returns numeric values with units stripped where possible.
    """
    with open(java_path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()

    out: Dict[str, float] = {}
    for m in _PARAM_RE.finditer(txt):
        k = m.group("k").strip()
        vraw = m.group("v").strip()
        v = strip_units(vraw)
        # many params are expressions; we only keep pure numeric
        if re.fullmatch(r"[-+0-9.eE]+", v.strip()):
            out[k] = float(v)
    return out

def get_param(params: Dict[str, float], name: str, default: float) -> float:
    return float(params.get(name, default))

# ============================================================
# 3) Dataset specification + loading
# ============================================================

@dataclass
class DatasetSpec:
    name: str
    enabled: bool = True
    split: str = "train"  # train/val/test

    # surface identification (top surface z=+Lz/2 by default in your model)
    # If surface_z is None, we fall back to using zmax from the loaded coordinates.
    surface_z: Optional[float] = None
    surface_tol: float = 2e-5
    surface_mode: str = "band"   # 'band' (z>=z_surf-tol) or 'isclose' (|z-z_surf|<=tol)


    # file paths (can point to same file; pick_var will extract)
    T_path: str = ""
    alpha_path: str = ""       # may equal T_path
    # Optional: alpha fields exported as separate CSVs (common in your COMSOL export)
    alpha_p_path: str = ""   # file containing alphap01
    alpha_c_path: str = ""   # file containing alphac01
    alpha_f_path: str = ""   # file containing alphaf01
    v_sub_path: str = ""
    v_mesh_path: str = ""
    v_in_path: str = ""        # optional; if missing, compute v_in=-(v_sub+v_mesh)
    Qvol_path: str = ""        # Q_total in volume
    qbnd_path: str = ""        # q_laser/q_conv/q_rad/q_sub
    dalpha_path: str = ""      # dalpha_dt (dp/dc/df)

    # scenario parameters
    P_laser: float = 10.0
    w0: float = 5e-4
    x0: float = 0.0
    y0: float = 0.0
    tauL: float = 0.2
    A_abs_v: float = 0.90
    A_abs_c: float = 0.95
    h_conv: float = 30.0
    eps_surf: float = 0.9
    T_inf: float = 300.0
    sigmaSB: float = 5.670374419e-8

    # material / kinetic constants (from Java when possible)
    Vf: float = 0.60
    Vr: float = 0.40
    rho_cf: float = 1800.0
    rho_res: float = 1200.0
    rho_char: float = 375.0
    theta: float = 0.5
    cp_cf: float = 800.0
    cp_res: float = 1000.0
    cp_char: float = 900.0
    k_cf_in: float = 10.0
    k_cf_z: float = 1.0
    k_res: float = 0.3
    k_char: float = 1.0

    H_pyro: float = 8.0e5
    H_ox: float = 3.2e7
    H_sub: float = 5.0e7

    # reaction layer thickness (m) for recession mapping
    drxn: float = 1e-4

    # optional: align time reference per dataset
    t_ref: Optional[float] = None

@dataclass
class Dataset:
    spec: DatasetSpec
    coords: np.ndarray   # (N,3)
    times: np.ndarray    # (Nt,)
    dt: float
    surf_mask: np.ndarray  # (N,)

    # fields
    T: np.ndarray
    ap: np.ndarray
    ac: np.ndarray
    af: np.ndarray
    v_sub: np.ndarray
    v_mesh: np.ndarray
    v_in: np.ndarray
    Q_total: np.ndarray
    qb_true: np.ndarray      # (N,Nt,4) [q_laser,q_conv,q_rad,q_sub]
    dr_true: np.ndarray      # (N,Nt,3) [dp,dc,df]

    # RAR pool: list of (node_idx, time_idx)
    rar_pool: List[Tuple[int,int]] = field(default_factory=list)

def load_dataset(spec: DatasetSpec, cache_dir: str) -> Dataset:
    wtT = load_wide_table(spec.T_path, cache_dir)
    coords = wtT.coords
    times = wtT.times
    dt = float(np.median(np.diff(times))) if times.size > 1 else 1.0

    # T and alphas
    T = wtT.data[pick_var(wtT.data, ["T", "temperature"])]
    # ---- alpha fields ----
    # Priority:
    #   1) spec.alpha_path : one file containing alphap01/alphac01/alphaf01 columns
    #   2) spec.alpha_{p,c,f}_path : three separate files (your current export list)
    #   3) fallback to temperature file (only works if you exported alphas together with T)
    if spec.alpha_path and os.path.exists(spec.alpha_path):
        wtA = load_wide_table(spec.alpha_path, cache_dir)
        ap = wtA.data[pick_var(wtA.data, ["alphap01", "alphap", "alpha_p"])]
        ac = wtA.data[pick_var(wtA.data, ["alphac01", "alphac", "alpha_c"])]
        af = wtA.data[pick_var(wtA.data, ["alphaf01", "alphaf", "alpha_f"])]
        if wtA.times.shape[0] != times.shape[0] or np.max(np.abs(wtA.times - times)) > 0:
            ap = align_by_time(ap, wtA.times, times)
            ac = align_by_time(ac, wtA.times, times)
            af = align_by_time(af, wtA.times, times)
    elif (spec.alpha_p_path and os.path.exists(spec.alpha_p_path)) or (spec.alpha_c_path and os.path.exists(spec.alpha_c_path)) or (spec.alpha_f_path and os.path.exists(spec.alpha_f_path)):
        Nt = times.shape[0]
        N = coords.shape[0]
        ap = np.zeros((N, Nt), dtype=np.float32)
        ac = np.zeros((N, Nt), dtype=np.float32)
        af = np.zeros((N, Nt), dtype=np.float32)

        if spec.alpha_p_path and os.path.exists(spec.alpha_p_path):
            wtp = load_wide_table(spec.alpha_p_path, cache_dir)
            tmp = wtp.data[pick_var(wtp.data, ["alphap01", "alphap", "alpha_p"])]
            ap = align_by_time(tmp, wtp.times, times) if (wtp.times.shape[0] != Nt or np.max(np.abs(wtp.times - times)) > 0) else tmp

        if spec.alpha_c_path and os.path.exists(spec.alpha_c_path):
            wtc = load_wide_table(spec.alpha_c_path, cache_dir)
            tmp = wtc.data[pick_var(wtc.data, ["alphac01", "alphac", "alpha_c"])]
            ac = align_by_time(tmp, wtc.times, times) if (wtc.times.shape[0] != Nt or np.max(np.abs(wtc.times - times)) > 0) else tmp

        if spec.alpha_f_path and os.path.exists(spec.alpha_f_path):
            wtf = load_wide_table(spec.alpha_f_path, cache_dir)
            tmp = wtf.data[pick_var(wtf.data, ["alphaf01", "alphaf", "alpha_f"])]
            af = align_by_time(tmp, wtf.times, times) if (wtf.times.shape[0] != Nt or np.max(np.abs(wtf.times - times)) > 0) else tmp
    else:
        wtA = wtT
        ap = wtA.data[pick_var(wtA.data, ["alphap01", "alphap", "alpha_p"])]
        ac = wtA.data[pick_var(wtA.data, ["alphac01", "alphac", "alpha_c"])]
        af = wtA.data[pick_var(wtA.data, ["alphaf01", "alphaf", "alpha_f"])]

    # velocities
    wts = load_wide_table(spec.v_sub_path, cache_dir)
    v_sub = wts.data[pick_var(wts.data, ["v_sub", "vsub"])]

    wtm = load_wide_table(spec.v_mesh_path, cache_dir)
    v_mesh = wtm.data[pick_var(wtm.data, ["v_mesh", "vmesh"])]

    if spec.v_in_path and os.path.exists(spec.v_in_path):
        wti = load_wide_table(spec.v_in_path, cache_dir)
        try:
            vin_key = pick_var(wti.data, ["v_in", "vin", "inward", "-((v_mesh + v_sub))", "v_mesh + v_sub"])
            v_in = wti.data[vin_key]
        except KeyError:
            v_in = -(v_mesh + v_sub)
    else:
        v_in = -(v_mesh + v_sub)

    # volume heat
    wtQ = load_wide_table(spec.Qvol_path, cache_dir)
    Q_total = wtQ.data[pick_var(wtQ.data, ["Q_total", "Qtotal", "Q"])]

    # boundary heat (q_laser/q_conv/q_rad/q_sub)
    # NOTE: COMSOL boundary exports often change expression names across models or plot groups.
    # We only *supervise* q_sub directly (see training loop), but we keep the full 4-channel array for bookkeeping.
    wtb = load_wide_table(spec.qbnd_path, cache_dir)

    def _try_get(keys: List[str]) -> Optional[np.ndarray]:
        try:
            return wtb.data[pick_var(wtb.data, keys)]
        except Exception:
            return None

    # broaden aliases to survive column name changes (underscores, dots, hf1.q0, etc.)
    q_laser = _try_get(["q_laser", "qlaser", "q_laser_in", "laser", "ht.hf1.q0", "hf1.q0", "q0"])
    q_conv  = _try_get(["q_conv", "qconv", "conv", "q_convection", "ht.qconv", "qconvective"])
    q_rad   = _try_get(["q_rad", "qrad", "rad", "q_radiation", "ht.qrad", "qradative"])
    q_sub   = _try_get(["q_sub", "qsub", "sub", "q_sublimation", "qevap", "q_evap", "q_vap", "q_vapor"])

    # fallback: if q_sub column is missing, compute it from v_sub using the SAME closure as in forward_phys
    # q_sub = -(v_sub * rho_eff) * H_sub
    if q_sub is None:
        rho_eff_fb = (spec.Vf*spec.rho_cf +
                      spec.Vr*spec.rho_res*(1.0-ap) +
                      spec.Vr*spec.rho_char*spec.theta*ap*(1.0-ac))
        rho_eff_fb = np.clip(rho_eff_fb, 1e-6, np.inf)
        q_sub = -(v_sub * rho_eff_fb) * float(spec.H_sub)

    # If other channels missing, fill zeros (they are not used for supervision).
    if q_laser is None: q_laser = np.zeros_like(q_sub, dtype=np.float32)
    if q_conv  is None: q_conv  = np.zeros_like(q_sub, dtype=np.float32)
    if q_rad   is None: q_rad   = np.zeros_like(q_sub, dtype=np.float32)

    qb_true = np.stack([q_laser, q_conv, q_rad, q_sub], axis=-1).astype(np.float32)

    # rates
    wtr = load_wide_table(spec.dalpha_path, cache_dir)
    dp = wtr.data[pick_var(wtr.data, ["dalphap_dt", "dalphap"])]
    dc = wtr.data[pick_var(wtr.data, ["dalphac_dt", "dalphac"])]
    df = wtr.data[pick_var(wtr.data, ["dalphaf_dt", "dalphaf"])]
    dr_true = np.stack([dp, dc, df], axis=-1).astype(np.float32)


    # ------------------------------------------------------------
    # Node alignment guard: COMSOL exports may differ in node count if you export from a plot grid vs mesh nodes.
    # We align everything onto a reference coordinate set (mode node count across mandatory tables).
    # ------------------------------------------------------------
    tables = {
        "T": wtT,
        "v_sub": wts,
        "v_mesh": wtm,
        "Q": wtQ,
        "qbnd": wtb,
        "rate": wtr,
    }
    ref_name, ref_coords = choose_reference_coords(tables)

    if coords.shape[0] != ref_coords.shape[0]:
        # reindex T and alphas to ref
        T = reindex_by_coords(T, coords, ref_coords, name="T")
        ap = reindex_by_coords(ap, coords, ref_coords, name="alphap01")
        ac = reindex_by_coords(ac, coords, ref_coords, name="alphac01")
        af = reindex_by_coords(af, coords, ref_coords, name="alphaf01")
        coords = ref_coords

    # align other arrays if needed
    if wts.coords.shape[0] != coords.shape[0]:
        v_sub = reindex_by_coords(v_sub, wts.coords, coords, name="v_sub")
    if wtm.coords.shape[0] != coords.shape[0]:
        v_mesh = reindex_by_coords(v_mesh, wtm.coords, coords, name="v_mesh")
    if wtQ.coords.shape[0] != coords.shape[0]:
        Q_total = reindex_by_coords(Q_total, wtQ.coords, coords, name="Q_total")
    if wtb.coords.shape[0] != coords.shape[0]:
        q_laser = reindex_by_coords(q_laser, wtb.coords, coords, name="q_laser", max_missing_frac=0.05)
        q_conv  = reindex_by_coords(q_conv,  wtb.coords, coords, name="q_conv",  max_missing_frac=0.05)
        q_rad   = reindex_by_coords(q_rad,   wtb.coords, coords, name="q_rad",   max_missing_frac=0.05)
        q_sub   = reindex_by_coords(q_sub,   wtb.coords, coords, name="q_sub",   max_missing_frac=0.05)
        qb_true = np.stack([q_laser, q_conv, q_rad, q_sub], axis=-1).astype(np.float32)
    if wtr.coords.shape[0] != coords.shape[0]:
        # reindex rates
        dp = reindex_by_coords(dp, wtr.coords, coords, name="dalphap_dt")
        dc = reindex_by_coords(dc, wtr.coords, coords, name="dalphac_dt")
        df = reindex_by_coords(df, wtr.coords, coords, name="dalphaf_dt")
        dr_true = np.stack([dp, dc, df], axis=-1).astype(np.float32)

    # v_in depends on v_sub/v_mesh; recompute after any reindex to guarantee consistent N
    v_in = -(v_mesh + v_sub)

    # surface mask: top surface (your model: z = +5e-4 = +Lz/2)
    zmax = float(np.nanmax(coords[:, 2]))
    z_surf = float(zmax if getattr(spec, "surface_z", None) is None else getattr(spec, "surface_z"))
    tol = float(getattr(spec, "surface_tol", 2e-5))
    mode = str(getattr(spec, "surface_mode", "band")).lower()
    if mode == "isclose":
        surf_mask = np.isclose(coords[:, 2], z_surf, atol=tol)
    else:
        surf_mask = (coords[:, 2] >= z_surf - tol)

    return Dataset(
        spec=spec, coords=coords, times=times, dt=dt, surf_mask=surf_mask,
        T=T.astype(np.float32), ap=ap.astype(np.float32), ac=ac.astype(np.float32), af=af.astype(np.float32),
        v_sub=v_sub.astype(np.float32), v_mesh=v_mesh.astype(np.float32), v_in=v_in.astype(np.float32),
        Q_total=Q_total.astype(np.float32), qb_true=qb_true, dr_true=dr_true,
    )

# ============================================================
# 4) Sampling
# ============================================================

def sample_indices_time(Nt: int, B: int, t_max_frac: float = 1.0) -> np.ndarray:
    t_max_frac = float(np.clip(t_max_frac, 0.0, 1.0))
    tmax = max(1, int(round((Nt-1) * t_max_frac)))
    return np.random.randint(0, tmax+1, size=B)

def sample_state(ds: Dataset, B: int, t_max_frac: float = 1.0) -> Tuple[np.ndarray, np.ndarray, Dict[str,np.ndarray]]:
    N = ds.coords.shape[0]
    Nt = ds.times.size
    ni = np.random.randint(0, N, size=B)
    ti = sample_indices_time(Nt, B, t_max_frac)
    xyz = ds.coords[ni]
    t = ds.times[ti][:, None].astype(np.float32)

    y = {
        "T": ds.T[ni, ti][:, None].astype(np.float32),
        "a": np.stack([ds.ap[ni, ti], ds.ac[ni, ti], ds.af[ni, ti]], axis=1).astype(np.float32),
        "v": np.stack([ds.v_sub[ni, ti], ds.v_mesh[ni, ti], ds.v_in[ni, ti]], axis=1).astype(np.float32),
        "Q": ds.Q_total[ni, ti][:, None].astype(np.float32),
        "dr": ds.dr_true[ni, ti, :].astype(np.float32),
        "qb": ds.qb_true[ni, ti, :].astype(np.float32),
    }
    return xyz, t, y

def sample_surface(ds: Dataset, B: int, t_max_frac: float = 1.0) -> Tuple[np.ndarray, np.ndarray, Dict[str,np.ndarray]]:
    surf_ids = np.where(ds.surf_mask)[0]
    if surf_ids.size == 0:
        surf_ids = np.arange(ds.coords.shape[0])
    ni = np.random.choice(surf_ids, size=B, replace=True)
    Nt = ds.times.size
    ti = sample_indices_time(Nt, B, t_max_frac)
    xyz = ds.coords[ni]
    t = ds.times[ti][:, None].astype(np.float32)

    y = {
        "T": ds.T[ni, ti][:, None].astype(np.float32),
        "qb": ds.qb_true[ni, ti, :].astype(np.float32),
        "dr": ds.dr_true[ni, ti, :].astype(np.float32),
        "a": np.stack([ds.ap[ni, ti], ds.ac[ni, ti], ds.af[ni, ti]], axis=1).astype(np.float32),
        "v": np.stack([ds.v_sub[ni, ti], ds.v_mesh[ni, ti], ds.v_in[ni, ti]], axis=1).astype(np.float32),
    }
    return xyz, t, y

def sample_phys(ds: Dataset, B: int, w_focus: float, t_max_frac: float = 1.0, rar_frac: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collocation points.
    - half from RAR pool if available
    - half from focused spatial sampling around (x0,y0)
    """
    N = ds.coords.shape[0]
    Nt = ds.times.size

    use_rar = (len(ds.rar_pool) > 0) and (rar_frac > 0)
    Br = int(round(B * rar_frac)) if use_rar else 0
    Bf = B - Br

    xyz_list = []
    t_list = []

    if Br > 0:
        sel = np.random.choice(np.arange(len(ds.rar_pool)), size=Br, replace=True)
        ni = np.array([ds.rar_pool[i][0] for i in sel], dtype=np.int64)
        ti = np.array([ds.rar_pool[i][1] for i in sel], dtype=np.int64)
        xyz_list.append(ds.coords[ni])
        t_list.append(ds.times[ti][:, None].astype(np.float32))

    # focused sampling
    ni = np.random.randint(0, N, size=Bf)
    xyz = ds.coords[ni].copy()
    # blend toward center via Gaussian weight
    r2 = (xyz[:,0]-ds.spec.x0)**2 + (xyz[:,1]-ds.spec.y0)**2
    w = np.exp(-r2 / max(w_focus*w_focus, 1e-20)).astype(np.float32)
    u = np.random.rand(Bf).astype(np.float32)
    mix = (u < w)
    if mix.any():
        xyz[mix,0] = (np.random.normal(ds.spec.x0, w_focus*0.4, size=mix.sum())).astype(np.float32)
        xyz[mix,1] = (np.random.normal(ds.spec.y0, w_focus*0.4, size=mix.sum())).astype(np.float32)
    xyz_list.append(xyz)

    ti = sample_indices_time(Nt, Bf, t_max_frac)
    t_list.append(ds.times[ti][:, None].astype(np.float32))

    return np.concatenate(xyz_list, axis=0).astype(np.float32), np.concatenate(t_list, axis=0).astype(np.float32)

# ============================================================
# 5) Normalization & scenario features
# ============================================================

@dataclass
class CFG:
    # geometry reference (15mm x 15mm x 1mm default)
    Lx: float = 15e-3
    Ly: float = 15e-3
    Lz: float = 1e-3

    # surface definition for BC/depth: your top surface is z=+5e-4 (=+Lz/2)
    surface_z: Optional[float] = None   # if None -> +Lz/2; set explicitly if your exports include other boundaries
    surface_tol: float = 1e-6
    surface_mode: str = "band"   # 'band' or 'isclose'

    # training
    device: str = "cuda:1" if torch.cuda.is_available() else "cpu"
    seed: int = 1234
    out_dir: str = "./out_pinn_v8"

    steps: int = 20000
    lr: float = 2e-4
    wd: float = 1e-4

    B_state: int = 2048
    B_phys: int = 2048
    B_bc: int = 1024

    warmup_steps: int = 1500
    ramp_steps: int = 3000          # physics ramp
    causal_ramp_steps: int = 6000   # time curriculum
    # sampling emphasis (helps low-power / long-duration cases)
    power_sample_alpha: float = 0.6   # dataset sampling prob ∝ (P+eps)^(-alpha) (alpha=0 -> round-robin)
    power_sample_eps: float = 0.5     # W, avoids exploding weights near P=0

    # time weighting (helps long-time horizons like 100s at low power)
    time_weight_alpha: float = 1.5    # w(t)=1+alpha*(t_norm^p); set 0 to disable
    time_weight_pow: float = 2.0
    time_weight_on_data: bool = True
    time_weight_on_phys: bool = True
    time_weight_on_bc: bool = True

    # weights (base)
    w_aux: float = 1.0
    w_ode: float = 0.2

    # adaptive weight control
    use_lra: bool = True
    lra_every: int = 50
    lra_alpha: float = 0.5
    w_phys_init: float = 1e-2
    w_bc_init: float = 1e-2
    min_phys: float = 1e-3
    min_bc: float = 1e-3
    max_phys: float = 10.0
    max_bc: float = 10.0
    ema_beta: float = 0.9

    # RAR
    rar_every: int = 500
    rar_candidates: int = 8192
    rar_keep: int = 4096
    rar_frac: float = 0.5

    # model
    width: int = 256
    depth: int = 10
    siren_w0: float = 30.0
    scen_dim: int = 7  # [P, w0, A_abs_v, A_abs_c, h, eps, Tinf]
    use_film: bool = True
    cond_t_ref: bool = True  # append log1p(t_ref) to FiLM conditioning (improves varying durations)

    # ------------------------------------------------------------
    # Baseline comparison switch
    # model_kind:
    #   - "ours"              : SIREN + FiLM (your original)
    #   - "siren_nofilm_pinn" : SIREN without FiLM (tests conditioning effect)
    #   - "mlp_pinn"          : vanilla MLP-PINN (tanh, no FiLM, no Fourier)
    #   - "ff_mlp_pinn"       : Fourier-features + MLP-PINN
    #   - "mlp_film_pinn"     : MLP + FiLM (conditioning without periodic activations)
    #   - "data_only_mlp"     : supervised-only MLP (no physics residuals)
    model_kind: str = "ours"
    mlp_act: str = "tanh"   # "tanh" | "silu" | "relu"



    # mHC-like hyperconnection mixing (deepen/complexify without exploding width)
    use_mhc: bool = True
    mhc_heads: int = 4            # width must be divisible by heads
    mhc_use_h0: bool = True       # mix with first-layer features h0

    # Fourier feature encoding (improves multi-scale spatial/temporal representation)
    use_fourier: bool = True
    fourier_bands: int = 8
    fourier_max_freq: float = 16.0

    # Mixture-of-heads specialization for low-P vs high-P (helps low-power accuracy)
    use_moe_heads: bool = True
    moe_P_ref: float = 15.0
    moe_gate_slope: float = 8.0
    moe_gate_bias: float = -0.5


    # gating / scaling for low-power stability & accuracy
    P_gate_ref: float = 10.0      # ablation/rate soft saturation: gateP = P/(P+P_gate_ref); temperature uses hard gate (P>0)
    gate_eps: float = 1e-3        # avoid division by 0 in raw-space supervision

    # low-power prior / ablation gating (helps OOD at 1W)
    gateV_pow: float = 2.0          # v/dr scale ~ gateP^pow (rates/velocities are strongly nonlinear)
    gateR_pow: float = 2.0
    P_noabl_th: float = 2.5         # below ~2.5W, ablation should be near-zero (smooth prior)
    P_noabl_width: float = 0.5      # smoothness of threshold in W (sigmoid width)
    w_noabl: float = 0.2            # weight of no-ablation prior term


    # raw-space supervision (counters vanishing gradients at low P)
    raw_target_clip: float = 50.0 # clamp raw targets to avoid extreme values

    # extra absolute-temperature supervision (helps low-ΔT regimes)
    T_abs_scale: float = 300.0
    w_T_raw: float = 1.0
    w_T_abs: float = 0.2

    w_v_raw: float = 0.5
    w_v_abs: float = 0.5

    w_dr_raw: float = 0.5
    w_dr_abs: float = 0.5

    # auto split rules (Problem-5)
    split_tolP: float = 1e-3
    split_tolT: float = 5e-2
    train_pairs: list = field(default_factory=lambda: [(1.0,100.0),(5.0,100.0),(10.503,100.0),(15.0,75.0),(20.0,40.0),(25.0,7.4437)])
    val_pairs: list = field(default_factory=lambda: [(18.0,40.0)])
    keep_only_listed_pairs: bool = True  # disable non-listed discovered datasets

    # other
    grad_clip: float = 1.0
    ckpt_every: int = 200
    eval_every: int = 2000

    # data
    java_path: str = r"D:\COMSOL\data\最终烧蚀2_nosolve20W.java"
    base_dir: str = r"D:\COMSOL\data"   # CHANGE THIS on your machine
    datasets: List[DatasetSpec] = field(default_factory=list)

def normalize_xyt(cfg: CFG, xyz: torch.Tensor, t: torch.Tensor, t_ref: float) -> torch.Tensor:
    """
    x,y,z -> [-1,1] by half-length; t -> [0,1] by t_ref
    """
    xn = xyz[:, 0:1] / (cfg.Lx/2)
    yn = xyz[:, 1:2] / (cfg.Ly/2)
    zn = xyz[:, 2:3] / (cfg.Lz/2)
    tn = t / max(t_ref, 1e-12)
    return torch.cat([xn, yn, zn, tn], dim=1)


def _time_weights(cfg: CFG, t_norm: torch.Tensor) -> torch.Tensor:
    """t_norm in [0,1]; returns per-sample weights emphasizing late times."""
    a = float(getattr(cfg, 'time_weight_alpha', 0.0))
    if a <= 0.0:
        return torch.ones_like(t_norm)
    p = float(getattr(cfg, 'time_weight_pow', 2.0))
    return 1.0 + a * torch.clamp(t_norm, 0.0, 1.0).pow(p)

def _wmse(pred: torch.Tensor, target: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Weighted MSE normalized by mean(w) to keep scale comparable."""
    if w is None:
        return F.mse_loss(pred, target)
    # broadcast w to match pred shape
    while w.dim() < pred.dim():
        w = w.unsqueeze(-1)
    num = (w * (pred - target) ** 2).mean()
    den = w.mean().clamp_min(1e-12)
    return num / den
def scenario_vec(spec: DatasetSpec) -> np.ndarray:
    """
    Raw scenario vector (not normalized): keep in physical units.
    By default this is 7D:
        [P, w0, A_abs_v, A_abs_c, h_conv, eps, T_inf]
    If spec.t_ref is available (recommended), append an 8th feature:
        log1p(t_ref)  (helps distinguish 7.44s/40s/75s/100s dynamics)
    """
    base = [
        spec.P_laser, spec.w0, spec.A_abs_v, spec.A_abs_c,
        spec.h_conv, spec.eps_surf, spec.T_inf
    ]
    if getattr(spec, "t_ref", None) is not None:
        base.append(float(math.log1p(float(spec.t_ref))))
    return np.array(base, dtype=np.float32)

# ============================================================
# 6) Model: SIREN + FiLM conditioning (parametric surrogate)
# ============================================================

class SineLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, w0: float = 30.0, is_first: bool = False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.w0 = float(w0)
        self.is_first = bool(is_first)
        # L-LAAF/N-LAAF style learnable amplitude on the sine argument (hidden layers only).
        self.use_laff = (not self.is_first)
        if self.use_laff:
            self.a_raw = nn.Parameter(torch.zeros(1))  # a=softplus(a_raw)+1e-3
        else:
            self.register_buffer('a_raw', torch.zeros(1))
        self.linear = nn.Linear(in_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            if self.is_first:
                # SIREN first-layer init
                bound = 1 / self.in_dim
            else:
                bound = math.sqrt(6 / self.in_dim) / self.w0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = 1.0
        if self.use_laff and (not self.is_first):
            a = (F.softplus(self.a_raw) + 1e-3)
        return torch.sin((self.w0 * a) * self.linear(x))

class FourierFeatures(nn.Module):
    """Deterministic Fourier feature encoding for normalized coordinates.

    This is a lightweight multi-scale positional encoding that often improves PINN accuracy,
    especially when solutions contain sharp gradients or multi-frequency content.
    """
    def __init__(self, in_dim: int = 4, num_bands: int = 8, max_freq: float = 16.0):
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_bands = int(num_bands)
        self.max_freq = float(max_freq)

        if self.num_bands <= 0:
            raise ValueError("num_bands must be positive.")
        if self.max_freq <= 0:
            raise ValueError("max_freq must be positive.")

        if self.num_bands == 1:
            freqs = torch.tensor([self.max_freq], dtype=torch.float32)
        else:
            freqs = 2.0 ** torch.linspace(0.0, math.log2(self.max_freq), steps=self.num_bands)
        self.register_buffer("freqs", freqs.view(1, 1, -1))  # (1,1,K)
        self.out_dim = 2 * self.in_dim * self.num_bands

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,in_dim) in normalized space (roughly O(1))
        x = x[:, :self.in_dim].unsqueeze(-1)                 # (B,in_dim,1)
        ang = 2.0 * math.pi * x * self.freqs                 # (B,in_dim,K)
        enc = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)  # (B,2*in_dim,K)
        return enc.reshape(x.shape[0], -1)

class FiLM(nn.Module):
    def __init__(self, scen_dim: int, width: int, depth: int):
        super().__init__()
        self.width = width
        self.depth = depth
        self.net = nn.Sequential(
            nn.Linear(scen_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, depth * 2 * width),
        )

    def forward(self, scen: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        returns gammas, betas: each list length=depth, each tensor (B,width)
        """
        h = self.net(scen)
        h = h.view(-1, self.depth, 2, self.width)
        gammas = [h[:, i, 0, :] for i in range(self.depth)]
        betas  = [h[:, i, 1, :] for i in range(self.depth)]
        return gammas, betas

class AblationSIRENFiLM(nn.Module):
    def __init__(self, cfg: CFG):
        super().__init__()
        self.cfg = cfg

        # scenario standardization constants (reasonable refs)
        self.register_buffer("scen_mu", torch.tensor([10.0, 5e-4, 0.9, 0.95, 30.0, 0.9, 300.0], dtype=torch.float32))
        # extra conditioning scalar: log1p(t_ref [s]) (duration). Kept separate to preserve scen_mu/scen_sig compatibility.
        self.register_buffer("tref_mu", torch.tensor([math.log1p(50.0)], dtype=torch.float32))
        self.register_buffer("tref_sig", torch.tensor([1.0], dtype=torch.float32))
        self.register_buffer("scen_sig", torch.tensor([10.0, 5e-4, 0.2, 0.2, 30.0, 0.2, 300.0], dtype=torch.float32))

        in_dim = 4  # x,y,z,t normalized
        width = cfg.width
        depth = cfg.depth

        # optional Fourier positional encoding in normalized coordinate space
        self.use_fourier = bool(getattr(cfg, "use_fourier", False))
        self.ff = FourierFeatures(in_dim=4,
                                  num_bands=int(getattr(cfg, "fourier_bands", 8)),
                                  max_freq=float(getattr(cfg, "fourier_max_freq", 16.0))) if self.use_fourier else None
        if self.ff is not None:
            in_dim = 4 + int(self.ff.out_dim)

        self.first = SineLayer(in_dim, width, w0=cfg.siren_w0, is_first=True)
        self.hidden = nn.ModuleList([SineLayer(width, width, w0=cfg.siren_w0, is_first=False) for _ in range(depth)])
        self.film = FiLM(scen_dim=(7 + (1 if cfg.cond_t_ref else 0)), width=width, depth=depth) if cfg.use_film else None

        self.use_mhc = bool(getattr(cfg, "use_mhc", False))
        self.mhc_heads = int(getattr(cfg, "mhc_heads", 1))
        self.mhc_use_h0 = bool(getattr(cfg, "mhc_use_h0", True))
        if self.use_mhc:
            if width % self.mhc_heads != 0:
                raise ValueError(f"width={width} must be divisible by mhc_heads={self.mhc_heads}")
            comps = 3 if self.mhc_use_h0 else 2
            self.mhc_logits = nn.Parameter(torch.zeros(depth, self.mhc_heads, comps, dtype=torch.float32))

        # heads: predict minimal independent fields
        self.h_dT = nn.Linear(width, 1)
        self.h_a  = nn.Linear(width, 3)   # ap,ac,af
        self.h_v  = nn.Linear(width, 2)   # v_sub, v_mesh
        self.h_dr = nn.Linear(width, 3)   # dp,dc,df
        self.h_qsub = nn.Linear(width, 1) # q_sub
        # Mixture-of-heads specialization: separate low-power heads blended by a gate g(P)
        self.use_moe_heads = bool(getattr(cfg, "use_moe_heads", False))
        self.moe_P_ref = float(getattr(cfg, "moe_P_ref", 15.0))
        self.moe_gate_slope = float(getattr(cfg, "moe_gate_slope", 8.0))
        self.moe_gate_bias = float(getattr(cfg, "moe_gate_bias", -0.5))
        if self.use_moe_heads:
            self.h_dT_lo = nn.Linear(width, 1)
            self.h_a_lo  = nn.Linear(width, 3)
            self.h_v_lo  = nn.Linear(width, 2)
            self.h_dr_lo = nn.Linear(width, 3)
            self.h_qsub_lo = nn.Linear(width, 1)


    def scen_norm(self, scen: torch.Tensor) -> torch.Tensor:
        """Normalize scenario features.
        Supports:
          - 7D: [P,w0,Aabs_v,Aabs_c,h,eps,Tinf]  -> normalized with scen_mu/scen_sig
          - 8D: append log1p(t_ref)             -> last dim normalized with tref_mu/tref_sig
        """
        if scen.shape[-1] == 7:
            return (scen - self.scen_mu) / (self.scen_sig + 1e-12)
        elif scen.shape[-1] == 8:
            s0 = (scen[:, :7] - self.scen_mu) / (self.scen_sig + 1e-12)
            # last dim is log1p(t_ref)
            s1 = (scen[:, 7:8] - self.tref_mu) / (self.tref_sig + 1e-12)
            return torch.cat([s0, s1], dim=1)
        else:
            raise ValueError(f"Unexpected scen dim={scen.shape[-1]}; expected 7 or 8.")

    def forward_latent(self, xytzn: torch.Tensor, scen: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_in = xytzn if getattr(self, 'ff', None) is None else torch.cat([xytzn, self.ff(xytzn)], dim=1)
        h = self.first(x_in)
        h0 = h
        if self.film is not None:
            gammas, betas = self.film(self.scen_norm(scen))
        else:
            gammas, betas = None, None

        for i, layer in enumerate(self.hidden):
            h_new = layer(h)
            if gammas is not None:
                h_new = h_new * (1.0 + gammas[i]) + betas[i]

            if getattr(self, "use_mhc", False):
                B = h_new.shape[0]
                heads = self.mhc_heads
                dh = h_new.shape[1] // heads
                h_r = h.view(B, heads, dh)
                hn_r = h_new.view(B, heads, dh)
                w = torch.softmax(self.mhc_logits[i], dim=-1)  # (heads, comps)
                w0 = w[:, 0].view(1, heads, 1)
                w1 = w[:, 1].view(1, heads, 1)
                if self.mhc_use_h0:
                    w2 = w[:, 2].view(1, heads, 1)
                    h0_r = h0.view(B, heads, dh)
                    h_r = (w0 * h_r) + (w1 * hn_r) + (w2 * h0_r)
                else:
                    h_r = (w0 * h_r) + (w1 * hn_r)
                h = h_r.view(B, heads * dh)
            else:
                h = h_new

        if getattr(self, "use_moe_heads", False):
            # gate using raw P (first scenario dim)
            P = scen[:, 0:1]
            g = torch.sigmoid(self.moe_gate_slope * (P / (self.moe_P_ref + 1e-12) - 1.0) + self.moe_gate_bias)  # (B,1)
            dT_raw = g * self.h_dT(h) + (1.0 - g) * self.h_dT_lo(h)
            a_raw  = g * self.h_a(h)  + (1.0 - g) * self.h_a_lo(h)
            v_raw  = g * self.h_v(h)  + (1.0 - g) * self.h_v_lo(h)
            dr_raw = g * self.h_dr(h) + (1.0 - g) * self.h_dr_lo(h)
            qsub_raw = g * self.h_qsub(h) + (1.0 - g) * self.h_qsub_lo(h)
        else:
            dT_raw = self.h_dT(h)
            a_raw  = self.h_a(h)
            v_raw  = self.h_v(h)
            dr_raw = self.h_dr(h)
            qsub_raw = self.h_qsub(h)

        return {
            "dT_raw": dT_raw,
            "a_raw": a_raw,
            "v_raw": v_raw,
            "dr_raw": dr_raw,
            "qsub_raw": qsub_raw,
        }
    # ---- COMSOL-consistent closures ----

    @staticmethod
    def rampL(t: torch.Tensor, tauL: torch.Tensor) -> torch.Tensor:
        # 0.5*(1+tanh((t-3*tauL)/tauL))
        return 0.5 * (1.0 + torch.tanh((t - 3.0*tauL) / (tauL + 1e-12)))

    def forward_phys(self,
                     xytzn: torch.Tensor,
                     scen: torch.Tensor,
                     const: Dict[str, torch.Tensor],
                     scales: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """
        Returns all physics-relevant quantities, with hard constraints:
          - P=0 => gateP=0 => dT=0, v=0, dr=0, q_sub=0 exactly; T=T_inf.
          - alphas in [0,1]
          - v_sub,v_mesh >=0
        """
        lat = self.forward_latent(xytzn, scen)

        # reconstruct physical x,y,z,t
        Lx, Ly, Lz = const["Lx"], const["Ly"], const["Lz"]
        t_ref = const["t_ref"]
        x = xytzn[:, 0:1] * (Lx/2)
        y = xytzn[:, 1:2] * (Ly/2)
        z = xytzn[:, 2:3] * (Lz/2)
        t = xytzn[:, 3:4] * t_ref

        P = scen[:, 0:1]
        w0 = scen[:, 1:2]
        A_abs_v = scen[:, 2:3]
        A_abs_c = scen[:, 3:4]
        h = scen[:, 4:5]
        eps = scen[:, 5:6]
        T_inf = scen[:, 6:7]

        # exact gate at P=0 (linear in P for temperature scaling)
        gateP = torch.clamp(P / (float(self.cfg.P_gate_ref) + 1e-12), min=0.0)
        gateP2 = gateP * gateP
        # additional smooth 'no-ablation' gate vs power (used for v/dr; avoids positive bias at very low P)
        P_th = float(getattr(self.cfg, "P_noabl_th", 0.0))
        P_w  = float(getattr(self.cfg, "P_noabl_width", 1.0))
        if P_th > 0.0 and P_w > 0.0:
            gateAblP = torch.sigmoid((P - P_th) / (P_w + 1e-12))
        else:
            gateAblP = torch.ones_like(P)
        gateV = (gateP ** float(getattr(self.cfg, "gateV_pow", 1.0))) * gateAblP
        gateR = (gateP ** float(getattr(self.cfg, "gateR_pow", 1.0))) * gateAblP
        # T
        T_scale = float(scales["T_scale"])
        dT = gateP * F.softplus(lat["dT_raw"]) * T_scale
        T = T_inf + dT

        # alphas
        a = torch.sigmoid(lat["a_raw"])
        ap, ac, af = a[:,0:1], a[:,1:2], a[:,2:3]

        # velocities (raw v_sub will be capped by energy limit later, consistent with COMSOL mdot_sub cap)
        v_scale = float(scales["v_scale"])
        v_sub_raw = gateV * F.softplus(lat["v_raw"][:,0:1]) * v_scale
        v_mesh = gateV * F.softplus(lat["v_raw"][:,1:2]) * v_scale
        # placeholder; will be overwritten after q_laser/q_conv/q_rad
        v_sub = v_sub_raw
        v_in = -(v_sub + v_mesh)

        # rates
        dr_scale = float(scales["dr_scale"])
        dr = gateR * F.softplus(lat["dr_raw"]) * dr_scale
        dp, dc, df = dr[:,0:1], dr[:,1:2], dr[:,2:3]

        # Abs_eff (COMSOL)
        Abs_eff = A_abs_v*(1.0-ap) + A_abs_c*ap

        # q_laser (COMSOL)
        r2 = (x-const["x0"])**2 + (y-const["y0"])**2
        q_laser = self.rampL(t, const["tauL"]) * Abs_eff * (2.0*P/(math.pi*w0*w0 + 1e-20)) * torch.exp(-2.0*r2/(w0*w0 + 1e-20))

        # material props (COMSOL-like)
        Vf = const["Vf"]; Vr = const["Vr"]
        rho_cf = const["rho_cf"]; rho_res = const["rho_res"]; rho_char = const["rho_char"]; theta = const["theta"]
        cp_cf = const["cp_cf"]; cp_res = const["cp_res"]; cp_char = const["cp_char"]
        k_cf_in = const["k_cf_in"]; k_cf_z = const["k_cf_z"]; k_res = const["k_res"]; k_char = const["k_char"]

        rho_eff = Vf*rho_cf + Vr*rho_res*(1.0-ap) + Vr*rho_char*theta*ap*(1.0-ac)
        # safe denom like COMSOL's flc2hs guard
        denom = torch.clamp(rho_eff, min=1e-6)
        cp_eff = (Vf*rho_cf*cp_cf + Vr*rho_res*(1.0-ap)*cp_res + Vr*rho_char*theta*ap*cp_char) / denom

        k_in = Vf*k_cf_in + Vr*(1.0-ap)*k_res + Vr*ap*k_char
        k_z  = Vf*k_cf_z  + Vr*(1.0-ap)*k_res + Vr*ap*k_char
        kx = k_in; ky = k_in; kz = k_z

        # heat sources (COMSOL closure)
        H_pyro = const["H_pyro"]; H_ox = const["H_ox"]
        Q_pyro = -(Vr*rho_res) * H_pyro * dp
        Q_ox_char = +(Vr*rho_res*theta*ap) * H_ox * dc
        Q_ox_fib  = +(Vf*rho_cf) * H_ox * df
        Q_total = Q_pyro + Q_ox_char + Q_ox_fib

        # q_conv, q_rad with COMSOL sign convention (inward flux into domain)
        # COMSOL: q_conv = h*(T_amb - T), q_rad = eps*sigma*(T_amb^4 - T^4)
        q_conv = h * (T_inf - T)
        sigmaSB = const["sigmaSB"]
        q_rad = eps * sigmaSB * (T_inf**4 - T**4)

        # energy-limited sublimation closure (COMSOL):
        #   q_net = max(q_laser + q_conv + q_rad, 0)
        #   mdot_cap = q_net/H_sub
        #   v_cap = mdot_cap/rho_eff
        q_net = F.relu(q_laser + q_conv + q_rad)
        rho_eff_cap = torch.clamp(rho_eff, min=1e-6)
        v_cap = q_net / (const["H_sub"] * rho_eff_cap + 1e-12)
        v_sub = torch.minimum(v_sub_raw, v_cap)
        v_in = -(v_sub + v_mesh)

        # q_sub (negative when removing heat): q_sub = -mdot_sub*H_sub = -(v_sub*rho_eff)*H_sub
        q_sub = -(v_sub * rho_eff_cap) * const["H_sub"]

        return {
            # core
            "x": x, "y": y, "z": z, "t": t,
            "T": T, "T_inf": T_inf,
            "ap": ap, "ac": ac, "af": af,
            "v_sub": v_sub, "v_mesh": v_mesh, "v_in": v_in,
            "dp": dp, "dc": dc, "df": df,
            # derived
            "Abs_eff": Abs_eff,
            "q_laser": q_laser, "q_conv": q_conv, "q_rad": q_rad, "q_sub": q_sub,
            "rho_eff": rho_eff, "cp_eff": cp_eff, "kx": kx, "ky": ky, "kz": kz,
            "Q_pyro": Q_pyro, "Q_ox_char": Q_ox_char, "Q_ox_fib": Q_ox_fib, "Q_total": Q_total,
            "gateP": gateP,  # temperature gate (linear in P_ref) for raw-space scaling
        "gateAbl": gateAblP,
            "gateV": gateV,
            "gateR": gateR,
            "dT_raw": lat["dT_raw"],
            "v_raw": lat["v_raw"],
            "dr_raw": lat["dr_raw"],
            "qsub_raw": lat["qsub_raw"],
        }
# ============================================================
# 7) Physics residuals (PDE + BC + ODE)
# ============================================================

def pde_residual(cfg: CFG, out: Dict[str, torch.Tensor], xytzn: torch.Tensor, const: Dict[str, torch.Tensor], scales: Dict[str,float]) -> torch.Tensor:
    """
    r = rho*cp*dT/dt - div(k grad T) - Q_total
    Uses full variable-coefficient divergence: div(kx dTdx, ky dTdy, kz dTdz)
    """
    T = out["T"]
    rho_cp = out["rho_eff"] * out["cp_eff"]
    Q = out["Q_total"]
    kx, ky, kz = out["kx"], out["ky"], out["kz"]

    invLx = 2.0/const["Lx"]
    invLy = 2.0/const["Ly"]
    invLz = 2.0/const["Lz"]
    invTr = 1.0/const["t_ref"]

    g1 = torch.autograd.grad(T.sum(), xytzn, create_graph=True)[0]
    dTdx = g1[:,0:1]*invLx
    dTdy = g1[:,1:2]*invLy
    dTdz = g1[:,2:3]*invLz
    dTdt = g1[:,3:4]*invTr

    fx = kx * dTdx
    fy = ky * dTdy
    fz = kz * dTdz

    dfxdx = torch.autograd.grad(fx.sum(), xytzn, create_graph=True)[0][:,0:1]*invLx
    dfydy = torch.autograd.grad(fy.sum(), xytzn, create_graph=True)[0][:,1:2]*invLy
    dfzdz = torch.autograd.grad(fz.sum(), xytzn, create_graph=True)[0][:,2:3]*invLz

    div = dfxdx + dfydy + dfzdz
    r = rho_cp * dTdt - div - Q

    # scale to stabilize
    T_scale = float(scales["T_scale"])
    # reference rho*cp at initial state (ap=0,ac=0)
    rho_ref = float(const["rho_ref"])
    cp_ref = float(const["cp_ref"])
    rhoCp_ref = rho_ref * cp_ref
    pde_scale = max(float(scales["Q_scale"]), rhoCp_ref * T_scale / float(const["t_ref"]), 1e-6)
    return r / pde_scale

def bc_residual(out: Dict[str, torch.Tensor], xytzn_b: torch.Tensor, const: Dict[str, torch.Tensor], scales: Dict[str,float]) -> torch.Tensor:
    """
    Surface energy balance on top surface (approx n=(0,0,1)):
      -kz dT/dz = q_laser - q_conv - q_rad - q_sub
    """
    T = out["T"]
    kz = out["kz"]
    q_in = out["q_laser"] + out["q_conv"] + out["q_rad"] + out["q_sub"]

    invLz = 2.0/const["Lz"]
    g1 = torch.autograd.grad(T.sum(), xytzn_b, create_graph=True)[0]
    dTdz = g1[:,2:3]*invLz

    lhs = -kz * dTdz
    r = lhs - q_in

    q_scale = max(float(scales["q_scale"]), 1e-6)
    return r / q_scale

def ode_residual(out: Dict[str, torch.Tensor], xytzn: torch.Tensor, const: Dict[str, torch.Tensor], scales: Dict[str,float]) -> torch.Tensor:
    """
    Enforce d(alpha)/dt == d alpha / dt head (dp/dc/df)
    """
    ap, ac, af = out["ap"], out["ac"], out["af"]
    dp, dc, df = out["dp"], out["dc"], out["df"]

    invTr = 1.0/const["t_ref"]
    gap = torch.autograd.grad(ap.sum(), xytzn, create_graph=True)[0][:,3:4]*invTr
    gac = torch.autograd.grad(ac.sum(), xytzn, create_graph=True)[0][:,3:4]*invTr
    gaf = torch.autograd.grad(af.sum(), xytzn, create_graph=True)[0][:,3:4]*invTr

    r = torch.cat([(gap - dp), (gac - dc), (gaf - df)], dim=1)
    dr_scale = max(float(scales["dr_scale"]), 1e-6)
    return r / dr_scale

# ============================================================
# 8) Adaptive weight (LRA-style)
# ============================================================

def grad_norm(loss: torch.Tensor, params: List[torch.Tensor]) -> torch.Tensor:
    gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    s = torch.zeros((), device=loss.device)
    for g in gs:
        if g is None:
            continue
        s = s + (g.detach()**2).sum()
    return torch.sqrt(s + 1e-12)

# ============================================================
# 9) Visualization / evaluation
# ============================================================

def surface_tricontour(coords_xy: np.ndarray, values: np.ndarray, out_png: str, title: str, vmin=None, vmax=None):
    tri = mtri.Triangulation(coords_xy[:,0], coords_xy[:,1])
    plt.figure(figsize=(7,6))
    im = plt.tricontourf(tri, values, levels=80, vmin=vmin, vmax=vmax)
    plt.colorbar(im, label="T (K)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

@torch.no_grad()

# ============================================================
# 6b) Baselines: MLP backbones (vanilla / Fourier / FiLM)
# ============================================================

def _get_act(name: str):
    name = str(name or "tanh").lower().strip()
    if name == "tanh":
        return torch.tanh
    if name in ("silu", "swish"):
        return F.silu
    if name == "relu":
        return F.relu
    raise ValueError(f"Unknown activation: {name}")

class AblationMLP(nn.Module):
    """
    Baseline model that keeps EXACTLY the same physics closures as AblationSIRENFiLM
    (forward_phys is copied verbatim), but replaces the backbone with a standard MLP.

    This isolates the effect of spectral bias / high-frequency representation.
    """
    def __init__(self, cfg: CFG, use_film: bool = False):
        super().__init__()
        self.cfg = cfg
        self.act = _get_act(getattr(cfg, "mlp_act", "tanh"))

        # same scenario standardization as SIREN class
        self.register_buffer("scen_mu", torch.tensor([10.0, 5e-4, 0.9, 0.95, 30.0, 0.9, 300.0], dtype=torch.float32))
        self.register_buffer("tref_mu", torch.tensor([math.log1p(50.0)], dtype=torch.float32))
        self.register_buffer("tref_sig", torch.tensor([1.0], dtype=torch.float32))
        self.register_buffer("scen_sig", torch.tensor([10.0, 5e-4, 0.2, 0.2, 30.0, 0.2, 300.0], dtype=torch.float32))

        in_dim = 4  # x,y,z,t normalized
        width = int(cfg.width)
        depth = int(cfg.depth)

        # optional Fourier positional encoding (reuse cfg.use_fourier)
        self.use_fourier = bool(getattr(cfg, "use_fourier", False))
        self.ff = FourierFeatures(in_dim=4,
                                  num_bands=int(getattr(cfg, "fourier_bands", 8)),
                                  max_freq=float(getattr(cfg, "fourier_max_freq", 16.0))) if self.use_fourier else None
        if self.ff is not None:
            in_dim = 4 + int(self.ff.out_dim)

        self.first = nn.Linear(in_dim, width)
        self.hidden = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        self.film = FiLM(scen_dim=(7 + (1 if cfg.cond_t_ref else 0)), width=width, depth=depth) if use_film else None

        # heads (same as original)
        self.h_dT = nn.Linear(width, 1)
        self.h_a  = nn.Linear(width, 3)
        self.h_v  = nn.Linear(width, 2)
        self.h_dr = nn.Linear(width, 3)
        self.h_qsub = nn.Linear(width, 1)

    def scen_norm(self, scen: torch.Tensor) -> torch.Tensor:
        # scen: (B,7) or (B,8 if appended log1p(t_ref))
        if scen.shape[1] == 7:
            s = (scen - self.scen_mu) / (self.scen_sig + 1e-12)
            return s
        # first 7 standardized + last one standardized separately
        s7 = (scen[:, :7] - self.scen_mu) / (self.scen_sig + 1e-12)
        st = (scen[:, 7:8] - self.tref_mu) / (self.tref_sig + 1e-12)
        return torch.cat([s7, st], dim=1)

    def forward_latent(self, xytzn: torch.Tensor, scen: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_in = xytzn if self.ff is None else torch.cat([xytzn, self.ff(xytzn)], dim=1)
        h = self.act(self.first(x_in))

        if self.film is not None:
            gammas, betas = self.film(self.scen_norm(scen))
        else:
            gammas, betas = None, None

        for i, layer in enumerate(self.hidden):
            h_new = self.act(layer(h))
            if gammas is not None:
                h_new = h_new * (1.0 + gammas[i]) + betas[i]
            h = h_new

        return {
            "dT_raw": self.h_dT(h),
            "a_raw": self.h_a(h),
            "v_raw": self.h_v(h),
            "dr_raw": self.h_dr(h),
            "qsub_raw": self.h_qsub(h),
        }

    # ---- copy the same COMSOL-consistent closures from AblationSIRENFiLM ----

    @staticmethod
    def rampL(t: torch.Tensor, tauL: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 + torch.tanh((t - 3.0*tauL) / (tauL + 1e-12)))

    def forward_phys(self,
                     xytzn: torch.Tensor,
                     scen: torch.Tensor,
                     const: Dict[str, torch.Tensor],
                     scales: Dict[str, float]) -> Dict[str, torch.Tensor]:
        # This is identical to AblationSIRENFiLM.forward_phys except lat source.
        lat = self.forward_latent(xytzn, scen)

        Lx, Ly, Lz = const["Lx"], const["Ly"], const["Lz"]
        t_ref = const["t_ref"]
        x = xytzn[:, 0:1] * (Lx/2)
        y = xytzn[:, 1:2] * (Ly/2)
        z = xytzn[:, 2:3] * (Lz/2)
        t = xytzn[:, 3:4] * t_ref

        P = scen[:, 0:1]
        w0 = scen[:, 1:2]
        A_abs_v = scen[:, 2:3]
        A_abs_c = scen[:, 3:4]
        h = scen[:, 4:5]
        eps = scen[:, 5:6]
        T_inf = scen[:, 6:7]

        gateP = torch.clamp(P / (float(self.cfg.P_gate_ref) + 1e-12), min=0.0)
        gateP2 = gateP * gateP
        P_th = float(getattr(self.cfg, "P_noabl_th", 0.0))
        P_w  = float(getattr(self.cfg, "P_noabl_width", 1.0))
        if P_th > 0.0 and P_w > 0.0:
            gateAblP = torch.sigmoid((P - P_th) / (P_w + 1e-12))
        else:
            gateAblP = torch.ones_like(P)
        gateV = (gateP ** float(getattr(self.cfg, "gateV_pow", 1.0))) * gateAblP
        gateR = (gateP ** float(getattr(self.cfg, "gateR_pow", 1.0))) * gateAblP

        T_scale = float(scales["T_scale"])
        dT = gateP * F.softplus(lat["dT_raw"]) * T_scale
        T = T_inf + dT

        a = torch.sigmoid(lat["a_raw"])
        ap, ac, af = a[:,0:1], a[:,1:2], a[:,2:3]

        v_scale = float(scales["v_scale"])
        v_sub_raw = gateV * F.softplus(lat["v_raw"][:,0:1]) * v_scale
        v_mesh = gateV * F.softplus(lat["v_raw"][:,1:2]) * v_scale
        v_sub = v_sub_raw
        v_in = -(v_sub + v_mesh)

        dr_scale = float(scales["dr_scale"])
        dr = gateR * F.softplus(lat["dr_raw"]) * dr_scale
        dp, dc, df = dr[:,0:1], dr[:,1:2], dr[:,2:3]

        Abs_eff = A_abs_v*(1.0-ap) + A_abs_c*ap

        r2 = (x-const["x0"])**2 + (y-const["y0"])**2
        q_laser = self.rampL(t, const["tauL"]) * Abs_eff * (2.0*P/(math.pi*w0*w0 + 1e-20)) * torch.exp(-2.0*r2/(w0*w0 + 1e-20))

        Vf = const["Vf"]; Vr = const["Vr"]
        rho_cf = const["rho_cf"]; rho_res = const["rho_res"]; rho_char = const["rho_char"]; theta = const["theta"]
        cp_cf = const["cp_cf"]; cp_res = const["cp_res"]; cp_char = const["cp_char"]
        k_cf_in = const["k_cf_in"]; k_cf_z = const["k_cf_z"]; k_res = const["k_res"]; k_char = const["k_char"]

        rho_eff = Vf*rho_cf + Vr*rho_res*(1.0-ap) + Vr*rho_char*theta*ap*(1.0-ac)
        denom = torch.clamp(rho_eff, min=1e-6)
        cp_eff = (Vf*rho_cf*cp_cf + Vr*rho_res*(1.0-ap)*cp_res + Vr*rho_char*theta*ap*cp_char) / denom

        k_in = Vf*k_cf_in + Vr*(1.0-ap)*k_res + Vr*ap*k_char
        k_z  = Vf*k_cf_z  + Vr*(1.0-ap)*k_res + Vr*ap*k_char
        kx = k_in; ky = k_in; kz = k_z

        H_pyro = const["H_pyro"]; H_ox = const["H_ox"]
        Q_pyro = -(Vr*rho_res) * H_pyro * dp
        Q_ox_char = +(Vr*rho_res*theta*ap) * H_ox * dc
        Q_ox_fib  = +(Vf*rho_cf) * H_ox * df
        Q_total = Q_pyro + Q_ox_char + Q_ox_fib

        q_conv = h * (T - T_inf)
        q_rad  = eps * const["sigmaSB"] * (T**4 - T_inf**4)

        # energy-limited sublimation flux cap
        H_sub = const["H_sub"]
        q_avail = torch.clamp(q_laser - q_conv - q_rad, min=0.0)
        v_cap = q_avail / (rho_cf * H_sub + 1e-12)  # m/s
        qsub_raw = gateV * F.softplus(lat["qsub_raw"])
        q_sub = qsub_raw
        v_sub = torch.minimum(v_sub_raw, v_cap)

        # keep sign convention: q_sub is "loss" positive
        q_sub = v_sub * rho_cf * H_sub

        v_in = -(v_sub + v_mesh)

        return {
            # core
            "x": x, "y": y, "z": z, "t": t,
            "T": T, "T_inf": T_inf,
            "ap": ap, "ac": ac, "af": af,
            "v_sub": v_sub, "v_mesh": v_mesh, "v_in": v_in,
            "dp": dp, "dc": dc, "df": df,
            # derived
            "Abs_eff": Abs_eff,
            "q_laser": q_laser, "q_conv": q_conv, "q_rad": q_rad, "q_sub": q_sub,
            "rho_eff": rho_eff, "cp_eff": cp_eff, "kx": kx, "ky": ky, "kz": kz,
            "Q_pyro": Q_pyro, "Q_ox_char": Q_ox_char, "Q_ox_fib": Q_ox_fib, "Q_total": Q_total,
            # gates & raws (required by the shared training loop)
            "gateP": gateP,      # temperature scaling gate (linear in P_ref)
            "gateAbl": gateAblP, # ablation gate (sigmoid around threshold)
            "gateV": gateV,
            "gateR": gateR,
            "dT_raw": lat["dT_raw"],
            "v_raw": lat["v_raw"],
            "dr_raw": lat["dr_raw"],
            "qsub_raw": lat["qsub_raw"],
        }

def make_model(cfg: CFG) -> nn.Module:
    kind = str(getattr(cfg, "model_kind", "ours")).lower().strip()
    if kind == "ours":
        cfg.use_film = True
        return AblationSIRENFiLM(cfg)
    if kind == "siren_nofilm_pinn":
        cfg.use_film = False
        return AblationSIRENFiLM(cfg)
    if kind == "mlp_pinn":
        return AblationMLP(cfg, use_film=False)
    if kind == "ff_mlp_pinn":
        cfg.use_fourier = True
        return AblationMLP(cfg, use_film=False)
    if kind == "mlp_film_pinn":
        cfg.use_film = True
        return AblationMLP(cfg, use_film=True)
    if kind == "data_only_mlp":
        return AblationMLP(cfg, use_film=False)
    raise ValueError(f"Unknown cfg.model_kind={kind}")

def apply_model_preset(cfg: CFG, preset: str):
    """
    Convenience presets that enforce 'normal' baselines:
      - normal MLP-PINN: tanh MLP, no FiLM, no Fourier, no mHC, no MoE.
      - ours: keep your defaults.
    """
    p = str(preset).lower().strip()
    if p in ("mlp_pinn", "baseline_mlp"):
        cfg.model_kind = "mlp_pinn"
        cfg.mlp_act = "tanh"
        cfg.use_film = False
        cfg.use_fourier = False
        cfg.use_mhc = False
        cfg.use_moe_heads = False
        return
    if p in ("ff_mlp_pinn", "baseline_ff"):
        cfg.model_kind = "ff_mlp_pinn"
        cfg.mlp_act = "tanh"
        cfg.use_film = False
        cfg.use_fourier = True
        cfg.use_mhc = False
        cfg.use_moe_heads = False
        return
    if p in ("mlp_film_pinn", "baseline_mlp_film"):
        cfg.model_kind = "mlp_film_pinn"
        cfg.mlp_act = "tanh"
        cfg.use_film = True
        cfg.use_fourier = False
        cfg.use_mhc = False
        cfg.use_moe_heads = False
        return
    if p in ("siren_nofilm_pinn", "baseline_siren_nofilm"):
        cfg.model_kind = "siren_nofilm_pinn"
        cfg.use_film = False
        cfg.use_fourier = True
        cfg.use_mhc = True
        cfg.use_moe_heads = False
        return
    if p in ("data_only_mlp", "baseline_data"):
        cfg.model_kind = "data_only_mlp"
        cfg.mlp_act = "tanh"
        cfg.use_film = False
        cfg.use_fourier = False
        cfg.use_mhc = False
        cfg.use_moe_heads = False
        return
    if p in ("ours", "main"):
        cfg.model_kind = "ours"
        # leave other cfg as-is
        return
    raise ValueError(f"Unknown preset: {preset}")


def eval_dataset(cfg: CFG, model: nn.Module, ds: Dataset, const: Dict[str, torch.Tensor], scales: Dict[str,float],
                 out_dir: str, times_to_plot: Tuple[float,...] = (1.0, 10.0, 50.0, 100.0)):
    ensure_dir(out_dir)
    device = cfg.device

    scen = torch.from_numpy(scenario_vec(ds.spec)[None,:]).to(device)
    t_ref = float(ds.spec.t_ref if ds.spec.t_ref is not None else ds.times.max())

    # Tmax curve
    Tmax_pred = []
    Tmax_true = []
    depth_pred = []
    depth_true = []
    depth_p = np.zeros(ds.coords.shape[0], dtype=np.float64)
    depth_t = np.zeros(ds.coords.shape[0], dtype=np.float64)

    for j, tval in enumerate(ds.times):
        xyz = torch.from_numpy(ds.coords).to(device)
        t = torch.full((ds.coords.shape[0],1), float(tval), device=device)
        xytzn = normalize_xyt(cfg, xyz, t, t_ref)
        out = model.forward_phys(xytzn, scen.repeat(ds.coords.shape[0],1), const, scales)
        Tpred = out["T"].detach().cpu().numpy().reshape(-1)
        Tmax_pred.append(float(np.nanmax(Tpred)))
        Tmax_true.append(float(np.nanmax(ds.T[:, j])))

        # ablation depth via integrating v_mesh+v_sub (surface recession speed)
        vrec_p = (out["v_mesh"] + out["v_sub"]).detach().cpu().numpy().reshape(-1)
        vrec_t = (ds.v_mesh[:, j] + ds.v_sub[:, j]).reshape(-1)
        if j > 0:
            dt = float(ds.times[j] - ds.times[j-1])
            depth_p += vrec_p * dt
            depth_t += vrec_t * dt
        depth_pred.append(float(np.nanmax(depth_p[ds.surf_mask])))
        depth_true.append(float(np.nanmax(depth_t[ds.surf_mask])))

    t_axis = ds.times
    plt.figure()
    plt.plot(t_axis, Tmax_true, label="True Tmax")
    plt.plot(t_axis, Tmax_pred, label="Pred Tmax")
    plt.xlabel("t (s)")
    plt.ylabel("Tmax (K)")
    plt.title(f"{ds.spec.name} Tmax")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{ds.spec.name}_Tmax.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(t_axis, depth_true, label="True depth")
    plt.plot(t_axis, depth_pred, label="Pred depth")
    plt.xlabel("t (s)")
    plt.ylabel("Max ablation depth (m)")
    plt.title(f"{ds.spec.name} Depth (integral of v_mesh+v_sub)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{ds.spec.name}_Depth.png"), dpi=200)
    plt.close()

    # surface maps
    surf_ids = np.where(ds.surf_mask)[0]
    xy = ds.coords[surf_ids][:, :2]
    for tv in times_to_plot:
        idx = int(np.argmin(np.abs(ds.times - tv)))
        tval = float(ds.times[idx])
        xyz = torch.from_numpy(ds.coords[surf_ids]).to(device)
        t = torch.full((surf_ids.size,1), tval, device=device)
        xytzn = normalize_xyt(cfg, xyz, t, t_ref)
        out = model.forward_phys(xytzn, scen.repeat(surf_ids.size,1), const, scales)
        Tp = out["T"].detach().cpu().numpy().reshape(-1)
        Tt = ds.T[surf_ids, idx].reshape(-1)
        surface_tricontour(xy, Tp, os.path.join(out_dir, f"{ds.spec.name}_SurfT_pred_t{tval:.2f}.png"),
                           f"{ds.spec.name} Surface T Pred @ t={tval:.2f}s")
        surface_tricontour(xy, Tt, os.path.join(out_dir, f"{ds.spec.name}_SurfT_true_t{tval:.2f}.png"),
                           f"{ds.spec.name} Surface T True @ t={tval:.2f}s")
        surface_tricontour(xy, (Tp-Tt), os.path.join(out_dir, f"{ds.spec.name}_SurfT_err_t{tval:.2f}.png"),
                           f"{ds.spec.name} Surface T Error @ t={tval:.2f}s")

# ============================================================
# 10) Dataset discovery helper
# ============================================================

def _glob_first(base_dir: str, patterns: List[str]) -> Optional[str]:
    import glob
    for p in patterns:
        hits = glob.glob(os.path.join(base_dir, p))
        if hits:
            # choose the newest by mtime
            hits.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            return hits[0]
    return None


def apply_split_rules(cfg: CFG, specs: List[DatasetSpec]) -> List[DatasetSpec]:
    """
    Assign split=train/val based on (P_laser, t_ref) pairs.
    Problem-5 default:
      train={1/100,5/100,10.503/100,15/75,20/40,25/7.4437}, val={18/40}.
    Note:
      - Some older discover_datasets() versions leave spec.t_ref=None. We infer duration from spec.name like '40s_20W'.
    """
    def _close(a: float, b: float, tol: float) -> bool:
        return abs(float(a) - float(b)) <= tol

    def _infer_tref(s: DatasetSpec) -> float:
        if s.t_ref is not None:
            return float(s.t_ref)
        name = str(getattr(s, "name", "") or "")
        md = re.search(r"(\d+(?:\.\d+)?)\s*s", name, flags=re.I)
        if md:
            return float(md.group(1))
        return 0.0

    train_pairs = [(float(p), float(t)) for (p, t) in getattr(cfg, "train_pairs", [])]
    val_pairs   = [(float(p), float(t)) for (p, t) in getattr(cfg, "val_pairs", [])]

    for s in specs:
        P = float(getattr(s, "P_laser", 0.0) or 0.0)
        T = _infer_tref(s)

        in_val = any(_close(P, p, cfg.split_tolP) and _close(T, t, cfg.split_tolT) for (p, t) in val_pairs)
        in_tr  = any(_close(P, p, cfg.split_tolP) and _close(T, t, cfg.split_tolT) for (p, t) in train_pairs)

        if in_val:
            s.split = "val"
            s.enabled = True
        elif in_tr:
            s.split = "train"
            s.enabled = True
        else:
            if getattr(cfg, "keep_only_listed_pairs", False):
                s.enabled = False
            else:
                s.split = "train"
                s.enabled = True
    return specs

def discover_datasets(cfg: CFG) -> List[DatasetSpec]:
    '''
    Recursive discovery for your naming conventions.

    Key fix vs earlier versions:
      - For the same (duration, power, folder), keep ONLY ONE temperature file.
        Prefer '*最终温度常规*.csv' over '*最终温度.csv' to avoid accidentally using plot-grid exports
        that have a different number of points.
    '''
    import glob
    from collections import defaultdict
    specs: List[DatasetSpec] = []

    base_dir = cfg.base_dir
    # candidate temperature files (recursive)
    candidates = []
    candidates += glob.glob(os.path.join(base_dir, "**", "*最终温度常规*.csv"), recursive=True)
    candidates += glob.glob(os.path.join(base_dir, "**", "*最终温度.csv"), recursive=True)
    candidates = sorted(set(candidates))

    def nonempty(p: str) -> bool:
        try:
            return os.path.isfile(p) and os.path.getsize(p) > 0
        except Exception:
            return False

    def parse_power_from_path(path: str) -> Tuple[float, bool]:
        """Parse laser power in W from filename or any parent folder segment.
        Accepts forms like '5W', '10.503W', '10_503W', '10-503W' (underscore/dash treated as decimal point).
        Returns (P, found_flag).
        """
        def _parse_one(s: str):
            # e.g. 10.503W, 10_503W, 10-503W, 25W
            m = re.search(r"(?P<p>\d+(?:[\._-]\d+)?)\s*[Ww]", s)
            if not m:
                return (float('nan'), False)
            pstr = m.group('p').replace('_', '.').replace('-', '.')
            try:
                return (float(pstr), True)
            except Exception:
                return (float('nan'), False)

        # 1) filename
        bn = os.path.basename(path)
        P, ok = _parse_one(bn)
        if ok:
            return P, True

        # 2) parent folders (nearest first)
        parts = re.split(r"[\\/]+", os.path.dirname(path))
        for seg in reversed(parts):
            P, ok = _parse_one(seg)
            if ok:
                return P, True

        return float('nan'), False

    candidates = [p for p in candidates if nonempty(p)]

    # group by (folder, dur, power-tag-present?, power)
    groups = defaultdict(list)
    for fp in candidates:
        bn = os.path.basename(fp)
        m_dur = re.search(r"(?P<dur>[0-9.]+)s", bn)
        if not m_dur:
            continue
        dur = float(m_dur.group("dur"))
        Pnom, p_found = parse_power_from_path(fp)
        if not p_found:
            # Special-case: your 10.503W datasets sometimes omit the power tag; by your convention, 100s corresponds to 10.503W.
            if abs(dur - 100.0) < 1e-9:
                Pnom, p_found = 10.503, True
            else:
                Pnom = float(getattr(cfg, "default_power", 0.0))
        
        folder = os.path.dirname(fp)
        groups[(folder, dur, Pnom)].append(fp)

    def pick_best_temp(paths: List[str]) -> str:
        # Prefer '温度常规', then smaller file size (usually fewer columns), then newest mtime.
        def score(p: str):
            bn = os.path.basename(p)
            prefer = 1 if ("温度常规" in bn) else 0
            try:
                sz = os.path.getsize(p)
            except Exception:
                sz = 1<<60
            try:
                mt = os.path.getmtime(p)
            except Exception:
                mt = 0.0
            # higher is better: prefer, then -size, then mtime
            return (prefer, -sz, mt)
        return max(paths, key=score)

    def find_one(folder: str, patterns: List[str]) -> str:
        for pat in patterns:
            hits = glob.glob(os.path.join(folder, pat))
            hits = [h for h in hits if nonempty(h)]
            if hits:
                hits.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                return hits[0]
        return ""

    for (folder, dur, Pnom), tps in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][0])):
        T_path = pick_best_temp(tps)
        bn = os.path.basename(T_path)
        # Power tag for pattern matching (accept power parsed from filename OR folder)
        Ptmp, p_found2 = parse_power_from_path(T_path)
        ptag = f"{Pnom:g}W" if p_found2 or (abs(Ptmp - Pnom) < 1e-6) else ""

        alpha_p = find_one(folder, [f"{dur:g}s*质量分数alphap01*{ptag}*.csv", f"{dur:g}s*质量分数alphap01*.csv"])
        alpha_c = find_one(folder, [f"{dur:g}s*质量分数alphac01*{ptag}*.csv", f"{dur:g}s*质量分数alphac01*.csv"])
        alpha_f = find_one(folder, [f"{dur:g}s*质量分数alphaf01*{ptag}*.csv", f"{dur:g}s*质量分数alphaf01*.csv"])

        vsub  = find_one(folder, [f"{dur:g}s*最终v_sub*{ptag}*.csv", f"{dur:g}s*最终v_sub*.csv"])
        vmesh = find_one(folder, [f"{dur:g}s*最终v_mesh*{ptag}*.csv", f"{dur:g}s*最终v_mesh*.csv"])
        vin   = find_one(folder, [f"{dur:g}s*最终向内速度*{ptag}*.csv", f"{dur:g}s*最终向内速度*.csv"])

        Qvol  = find_one(folder, [f"{dur:g}s*最终体热*{ptag}*.csv", f"{dur:g}s*最终体热*.csv"])
        qbnd  = find_one(folder, [f"{dur:g}s*最终边界热*{ptag}*.csv", f"{dur:g}s*最终边界热*.csv"])
        dalp  = find_one(folder, [f"{dur:g}s*最终反应速率*{ptag}*.csv", f"{dur:g}s*最终反应速率*.csv"])

        mandatory_ok = (T_path and nonempty(T_path) and vsub and nonempty(vsub) and vmesh and nonempty(vmesh) and Qvol and nonempty(Qvol) and qbnd and nonempty(qbnd) and dalp and nonempty(dalp))

        spec = DatasetSpec(
            name=f"{dur:g}s_{Pnom:g}W" if ptag else f"{dur:g}s_base",
            enabled=bool(mandatory_ok),
            split="train",
            T_path=T_path,
            alpha_path="",
            alpha_p_path=alpha_p,
            alpha_c_path=alpha_c,
            alpha_f_path=alpha_f,
            v_sub_path=vsub,
            v_mesh_path=vmesh,
            v_in_path=vin,
            Qvol_path=Qvol,
            qbnd_path=qbnd,
            dalpha_path=dalp,
            t_ref=dur,
            P_laser=Pnom,
            surface_z=(cfg.surface_z if cfg.surface_z is not None else 0.5*cfg.Lz),
            surface_tol=cfg.surface_tol,
            surface_mode=cfg.surface_mode,
        )
        specs.append(spec)

    if specs:
        return specs
    return cfg.datasets

# ============================================================
# 11) Training
# ============================================================

def build_const(cfg: CFG, spec: DatasetSpec, params: Dict[str,float], device: str) -> Dict[str, torch.Tensor]:
    """
    Per-dataset constants: geometry + scenario + material.
    """
    t_ref = float(spec.t_ref if spec.t_ref is not None else (spec.t_ref or 100.0))
    # derive rho_ref/cp_ref for scaling
    rho_ref = spec.Vf*spec.rho_cf + spec.Vr*spec.rho_res
    cp_ref = (spec.Vf*spec.rho_cf*spec.cp_cf + spec.Vr*spec.rho_res*spec.cp_res) / max(rho_ref, 1e-6)

    return {
        "Lx": torch.tensor(cfg.Lx, device=device),
        "Ly": torch.tensor(cfg.Ly, device=device),
        "Lz": torch.tensor(cfg.Lz, device=device),
        "t_ref": torch.tensor(t_ref, device=device),
        "x0": torch.tensor(spec.x0, device=device),
        "y0": torch.tensor(spec.y0, device=device),
        "tauL": torch.tensor(spec.tauL, device=device),
        "sigmaSB": torch.tensor(spec.sigmaSB, device=device),

        "Vf": torch.tensor(spec.Vf, device=device),
        "Vr": torch.tensor(spec.Vr, device=device),
        "rho_cf": torch.tensor(spec.rho_cf, device=device),
        "rho_res": torch.tensor(spec.rho_res, device=device),
        "rho_char": torch.tensor(spec.rho_char, device=device),
        "theta": torch.tensor(spec.theta, device=device),
        "cp_cf": torch.tensor(spec.cp_cf, device=device),
        "cp_res": torch.tensor(spec.cp_res, device=device),
        "cp_char": torch.tensor(spec.cp_char, device=device),
        "k_cf_in": torch.tensor(spec.k_cf_in, device=device),
        "k_cf_z": torch.tensor(spec.k_cf_z, device=device),
        "k_res": torch.tensor(spec.k_res, device=device),
        "k_char": torch.tensor(spec.k_char, device=device),

        "H_pyro": torch.tensor(spec.H_pyro, device=device),
        "H_ox": torch.tensor(spec.H_ox, device=device),
        "H_sub": torch.tensor(spec.H_sub, device=device),
        "drxn": torch.tensor(spec.drxn, device=device),

        "rho_ref": torch.tensor(rho_ref, device=device),
        "cp_ref": torch.tensor(cp_ref, device=device),
    }

def compute_global_scales(dss: List[Dataset]) -> Dict[str,float]:
    T_all, v_all, Q_all, dr_all, q_all = [], [], [], [], []
    for ds in dss:
        N = ds.T.shape[0]  # use actual array shape (coords may come from a different export)
        Nt = ds.times.size
        if not (ds.v_sub.shape[0]==N and ds.v_mesh.shape[0]==N and ds.v_in.shape[0]==N and ds.Q_total.shape[0]==N):
            raise ValueError(f"[ShapeMismatch] N(T)={N} but v_sub={ds.v_sub.shape}, v_mesh={ds.v_mesh.shape}, v_in={ds.v_in.shape}, Q_total={ds.Q_total.shape}. Check COMSOL exports and delete cache_npz.")
        n_s = min(50000, N)
        ni = np.random.choice(np.arange(N), size=n_s, replace=False)
        ti = np.random.randint(0, Nt, size=n_s)

        T_all.append(ds.T[ni, ti])
        v_all.append(ds.v_sub[ni, ti]); v_all.append(ds.v_mesh[ni, ti]); v_all.append(ds.v_in[ni, ti])
        Q_all.append(ds.Q_total[ni, ti])
        dr_all.append(ds.dr_true[ni, ti, :].reshape(-1))
        q_all.append(ds.qb_true[ni, ti, :].reshape(-1))

    if len(T_all) == 0:
        raise ValueError(
            "compute_global_scales(): no samples collected. This usually means no enabled datasets or empty surf_mask/time selection. "
            "Check dataset discovery, split rules, and that temperature CSVs contain valid time columns."
        )
    T_all = np.concatenate(T_all).astype(np.float32)
    v_all = np.concatenate(v_all).astype(np.float32)
    Q_all = np.concatenate(Q_all).astype(np.float32)
    dr_all = np.concatenate(dr_all).astype(np.float32)
    q_all = np.concatenate(q_all).astype(np.float32)

    _, T_scale = robust_center_scale(T_all, q=0.995, floor=1e-6)
    # delta-T (relative to T_inf) scale across datasets (for raw-space supervision)
    dT_all = []
    for ds in dss:
        try:
            Tinf = float(getattr(ds.spec, 'T_inf', 300.0) or 300.0)
        except Exception:
            Tinf = 300.0
        dT_all.append((ds.T - Tinf).reshape(-1,1))
    dT_all = np.concatenate(dT_all).astype(np.float32)
    dT_scale = robust_scale_abs(dT_all, q=0.995, floor=1e-6)
    v_scale = robust_scale_abs(v_all, q=0.999, floor=1e-12)
    Q_scale = robust_scale_abs(Q_all, q=0.999, floor=1e-12)
    dr_scale = robust_scale_abs(dr_all, q=0.999, floor=1e-12)
    q_scale = robust_scale_abs(q_all, q=0.999, floor=1e-12)

    return dict(T_scale=float(T_scale), dT_scale=float(dT_scale), v_scale=float(v_scale), Q_scale=float(Q_scale),
                dr_scale=float(dr_scale), q_scale=float(q_scale))

def train(cfg: CFG):
    # resolve output paths early so you always know where checkpoints go
    cfg.out_dir = os.path.abspath(cfg.out_dir)
    ensure_dir(cfg.out_dir)
    ckpt_dir = os.path.join(cfg.out_dir, "checkpoints")
    ensure_dir(ckpt_dir)
    cache_dir = os.path.join(cfg.out_dir, "cache_npz")
    ensure_dir(cache_dir)
    print(f"[Paths] CWD={os.getcwd()}")
    print(f"[Paths] out_dir={cfg.out_dir}")
    print(f"[Paths] ckpt_dir={ckpt_dir}")
    print(f"[Paths] cache_dir={cache_dir}")
    set_seed(cfg.seed)

    # parse Java params and inject into dataset specs (best effort)
    params = parse_comsol_java_params(cfg.java_path) if (cfg.java_path and os.path.exists(cfg.java_path)) else {}

    # auto discover if none specified
    if not cfg.datasets:
        cfg.datasets = discover_datasets(cfg)
        cfg.datasets = apply_split_rules(cfg, cfg.datasets)
        if not cfg.datasets:
            raise ValueError("No datasets discovered. Set CFG.datasets manually.")

    # load datasets
    dss: List[Dataset] = []
    for spec in cfg.datasets:
        if not spec.enabled:
            continue
        # enrich some constants from Java if present
        spec.w0 = float(get_param(params, "w0", spec.w0))
        spec.tauL = float(get_param(params, "tauL", spec.tauL))
        spec.A_abs_v = float(get_param(params, "A_abs_v", spec.A_abs_v))
        spec.A_abs_c = float(get_param(params, "A_abs_c", spec.A_abs_c))
        spec.Vf = float(get_param(params, "Vf", spec.Vf))
        spec.Vr = float(get_param(params, "Vr", 1.0 - spec.Vf))
        spec.rho_cf = float(get_param(params, "rho_cf", spec.rho_cf))
        spec.rho_res = float(get_param(params, "rho_res", spec.rho_res))
        spec.rho_char = float(get_param(params, "rho_char", spec.rho_char))
        spec.theta = float(get_param(params, "theta", spec.theta))
        spec.cp_cf = float(get_param(params, "cp_cf", spec.cp_cf))
        spec.cp_res = float(get_param(params, "cp_res", spec.cp_res))
        spec.cp_char = float(get_param(params, "cp_char", spec.cp_char))
        spec.k_cf_in = float(get_param(params, "k_cf_in", spec.k_cf_in))
        spec.k_cf_z  = float(get_param(params, "k_cf_z", spec.k_cf_z))
        spec.k_res   = float(get_param(params, "k_res", spec.k_res))
        spec.k_char  = float(get_param(params, "k_char_th", spec.k_char))
        spec.H_pyro  = float(get_param(params, "H_pyro", spec.H_pyro))
        spec.H_sub   = float(get_param(params, "H_sub", spec.H_sub))
        # H_ox often derived; if missing keep default

        dss.append(load_dataset(spec, cache_dir))

    # scales
    scales = compute_global_scales(dss)
    save_json(os.path.join(cfg.out_dir, "scales.json"), scales)

    device = cfg.device

    data_only = (str(getattr(cfg,'model_kind','')).lower().strip() == 'data_only_mlp')
    model = make_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.steps, 1))

    # adaptive weights
    w_phys = float(cfg.w_phys_init)
    w_bc = float(cfg.w_bc_init)
    w_phys_ema = w_phys
    w_bc_ema = w_bc

    hist = {
        "step": [], "lr": [],
        "L": [], "Ldata": [], "Laux": [], "Lphys": [], "Lbc": [], "Lode": [],
        "w_phys": [], "w_bc": [],
    }

    # group datasets by split for cycling
    train_dss = [ds for ds in dss if ds.spec.split == "train"]
    val_dss = [ds for ds in dss if ds.spec.split == "val"]
    test_dss = [ds for ds in dss if ds.spec.split == "test"]
    if not train_dss:
        train_dss = dss

    # dataset sampling probabilities (favor low power)
    train_ds_prob = None
    if getattr(cfg, 'power_sample_alpha', 0.0) > 0.0 and len(train_dss) > 1:
        Ps = np.array([float(getattr(ds.spec, 'P_laser', 0.0)) for ds in train_dss], dtype=np.float64)
        epsP = float(getattr(cfg, 'power_sample_eps', 0.5))
        alphaP = float(getattr(cfg, 'power_sample_alpha', 0.6))
        w = (Ps + epsP) ** (-alphaP)
        w = np.clip(w, 1e-12, None)
        train_ds_prob = (w / w.sum()).astype(np.float64)


    # constants per dataset
    const_by_name: Dict[str, Dict[str, torch.Tensor]] = {}
    for ds in dss:
        const_by_name[ds.spec.name] = build_const(cfg, ds.spec, params, device)

    # select params for grad norm (avoid heads dominating)
    trunk_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "h_" in n:
            continue
        trunk_params.append(p)

    t0 = time.time()
    last_step = -1
    interrupted = False
    crashed = False
    try:
        for step in range(cfg.steps + 1):
            last_step = step
            model.train()

            # choose dataset (power-aware sampling to emphasize low-power / long-duration)
            if getattr(cfg, 'power_sample_alpha', 0.0) > 0.0:
                # precomputed weights outside the loop if available
                if 'train_ds_prob' in locals():
                    ds = np.random.choice(train_dss, p=train_ds_prob)
                else:
                    ds = train_dss[step % len(train_dss)]
            else:
                ds = train_dss[step % len(train_dss)]
            spec = ds.spec
            t_ref = float(spec.t_ref if spec.t_ref is not None else ds.times.max())
            const = const_by_name[spec.name]

            # scenario tensor
            scen = torch.from_numpy(scenario_vec(spec)[None,:]).to(device)

            # ramp schedules
            if step < cfg.warmup_steps:
                phys_ramp = 0.0
            else:
                phys_ramp = min(1.0, (step - cfg.warmup_steps) / max(cfg.ramp_steps, 1))
            t_max_frac = min(1.0, step / max(cfg.causal_ramp_steps, 1))

            # ---------------- data batch ----------------
            xyz, t, y = sample_state(ds, cfg.B_state, t_max_frac=t_max_frac)
            xb = torch.from_numpy(xyz).to(device)
            tb = torch.from_numpy(t).to(device)
            xytzn = normalize_xyt(cfg, xb, tb, t_ref)
            w_time_state = None
            if getattr(cfg, 'time_weight_on_data', False):
                w_time_state = _time_weights(cfg, xytzn[:, 3:4].detach())

            out = model.forward_phys(xytzn, scen.repeat(cfg.B_state,1), const, scales)

            # temperature: combine raw-space supervision (better low-P gradients) + light abs-K supervision.
            T_true = torch.from_numpy(y["T"]).to(device)
            T_inf = out["T_inf"].detach()
            dT_true_K = (T_true - T_inf).clamp_min(0.0)

            gateP = out["gateP"].detach().clamp_min(0.0)
            dT_scale = float(scales.get("dT_scale", scales["T_scale"]))
            denom = (gateP + float(cfg.gate_eps)) * dT_scale + 1e-12
            raw_tgt = (dT_true_K / denom).clamp(0.0, float(cfg.raw_target_clip))

            raw_pred = F.softplus(out["dT_raw"])
            L_T_raw = _wmse(raw_pred, raw_tgt, w_time_state)
            L_T_abs = _wmse(out["T"] / float(cfg.T_abs_scale), T_true / float(cfg.T_abs_scale), w_time_state)
            L_T = float(cfg.w_T_raw) * L_T_raw + float(cfg.w_T_abs) * L_T_abs

            a_true = torch.from_numpy(y["a"]).to(device)
            L_a = _wmse(torch.cat([out["ap"], out["ac"], out["af"]], dim=1), a_true, w_time_state)

            v_true = torch.from_numpy(y["v"]).to(device)
            v_pred = torch.cat([out["v_sub"], out["v_mesh"], out["v_in"]], dim=1) / (scales["v_scale"] + 1e-12)
            v_true_n = v_true / (scales["v_scale"] + 1e-12)
            L_v_abs = _wmse(v_pred, v_true_n, w_time_state)

            raw_v_pred = F.softplus(out["v_raw"])
            raw_v_tgt = (v_true[:,0:2].clamp_min(0.0) / ((out["gateV"].detach() + float(cfg.gate_eps)) * float(scales["v_scale"]) + 1e-12)).clamp(0.0, float(cfg.raw_target_clip))
            L_v_raw = _wmse(raw_v_pred, raw_v_tgt, w_time_state)

            L_v = float(cfg.w_v_raw) * L_v_raw + float(cfg.w_v_abs) * L_v_abs

            L_data = L_T + L_a + L_v
            # auxiliary supervision (rates + Q_total + q_sub) — still helpful for fast convergence
            dr_true = torch.from_numpy(y["dr"]).to(device)
            L_dr_abs = _wmse(torch.cat([out["dp"], out["dc"], out["df"]], dim=1) / (scales["dr_scale"] + 1e-12),
                                  dr_true / (scales["dr_scale"] + 1e-12), w_time_state)
            raw_dr_pred = F.softplus(out["dr_raw"])
            raw_dr_tgt = (dr_true.clamp_min(0.0) / ((out["gateR"].detach() + float(cfg.gate_eps)) * float(scales["dr_scale"]) + 1e-12)).clamp(0.0, float(cfg.raw_target_clip))
            L_dr_raw = _wmse(raw_dr_pred, raw_dr_tgt, w_time_state)
            L_dr = float(cfg.w_dr_raw) * L_dr_raw + float(cfg.w_dr_abs) * L_dr_abs
            Q_true = torch.from_numpy(y["Q"]).to(device)
            L_Q = _wmse(out["Q_total"]/(scales["Q_scale"]+1e-12), Q_true/(scales["Q_scale"]+1e-12), w_time_state)

            qb_true = torch.from_numpy(y["qb"]).to(device)
            # only supervise q_sub directly; q_laser/conv/rad are computed
            L_qsub = _wmse(out["q_sub"]/(scales["q_scale"]+1e-12), qb_true[:,3:4]/(scales["q_scale"]+1e-12), w_time_state)

            L_aux = L_dr + L_Q + L_qsub

            # ---------------- physics batch ----------------
            xyzp, tp = sample_phys(ds, cfg.B_phys, w_focus=spec.w0*2.0, t_max_frac=t_max_frac, rar_frac=cfg.rar_frac)
            xbp = torch.from_numpy(xyzp).to(device)
            tbp = torch.from_numpy(tp).to(device)
            xytzn_p = normalize_xyt(cfg, xbp, tbp, t_ref)
            xytzn_p.requires_grad_(True)
            outp = model.forward_phys(xytzn_p, scen.repeat(cfg.B_phys,1), const, scales)
            w_time_phys = None
            if getattr(cfg, 'time_weight_on_phys', False):
                w_time_phys = _time_weights(cfg, xytzn_p[:, 3:4].detach())

            r_pde = pde_residual(cfg, outp, xytzn_p, const, scales)
            L_phys = _wmse(r_pde, torch.zeros_like(r_pde), w_time_phys)

            r_ode = ode_residual(outp, xytzn_p, const, scales)
            L_ode = _wmse(r_ode, torch.zeros_like(r_ode), w_time_phys)

            # ---------------- BC batch ----------------
            xyzb, tb, yb = sample_surface(ds, cfg.B_bc, t_max_frac=t_max_frac)
            xbb = torch.from_numpy(xyzb).to(device)
            tbb = torch.from_numpy(tb).to(device)
            xytzn_b = normalize_xyt(cfg, xbb, tbb, t_ref)
            xytzn_b.requires_grad_(True)
            outb = model.forward_phys(xytzn_b, scen.repeat(cfg.B_bc,1), const, scales)
            w_time_bc = None
            if getattr(cfg, 'time_weight_on_bc', False):
                w_time_bc = _time_weights(cfg, xytzn_b[:, 3:4].detach())
            r_bc = bc_residual(outb, xytzn_b, const, scales)
            L_bc = _wmse(r_bc, torch.zeros_like(r_bc), w_time_bc)
            # no-ablation prior at very low power (regularizer for low-P stability)
            L_noabl = torch.tensor(0.0, device=device)
            if float(getattr(cfg, "w_noabl", 0.0)) > 0.0 and float(getattr(cfg, "P_noabl_th", 0.0)) > 0.0:
                # reuse the current state batch's xytzn; keep other scenario params same; randomize P in [0, P_noabl_th]
                Pmax = float(cfg.P_noabl_th)
                scen_low = scen.detach().repeat(xytzn.shape[0], 1).clone()
                scen_low[:, 0:1] = torch.rand_like(scen_low[:, 0:1]) * Pmax  # per-sample random low P
                out_low = model.forward_phys(xytzn.detach(), scen_low, const, scales)
                # enforce near-zero ablation velocities and reaction rates (normalized)
                z_v = torch.zeros_like(torch.cat([out_low["v_sub"], out_low["v_mesh"]], dim=1))
                z_dr = torch.zeros_like(torch.cat([out_low["dp"], out_low["dc"], out_low["df"]], dim=1))
                L_noabl = F.mse_loss(torch.cat([out_low["v_sub"], out_low["v_mesh"]], dim=1) / (scales["v_scale"] + 1e-12), z_v) + (
                          F.mse_loss(torch.cat([out_low["dp"], out_low["dc"], out_low["df"]], dim=1) / (scales["dr_scale"] + 1e-12), z_dr)
                )


            # total loss
            w_phys_eff = phys_ramp * w_phys_ema
            w_bc_eff = phys_ramp * w_bc_ema

            if data_only:
                # supervised-only baseline: isolate spectral bias / representation without physics constraints
                w_phys_eff = 0.0
                w_bc_eff = 0.0
                L_phys = L_phys * 0.0
                L_bc = L_bc * 0.0
                L_ode = L_ode * 0.0
                L_noabl = L_noabl * 0.0

            loss = L_data + cfg.w_aux * L_aux + w_phys_eff * L_phys + w_bc_eff * L_bc + cfg.w_ode * L_ode + float(getattr(cfg, "w_noabl", 0.0)) * L_noabl

            opt.zero_grad(set_to_none=True)
            # LRA adaptive weights (every lra_every steps; MUST run before backward(), since backward frees the graph)
            if cfg.use_lra and (step % cfg.lra_every == 0) and (step >= cfg.warmup_steps):
                # compute grad norms on trunk for stability
                # (use unweighted phys/bc/ode; ramp doesn't affect this update)
                g_data = grad_norm(L_data + cfg.w_aux*L_aux, trunk_params)
                g_phys = grad_norm(L_phys + cfg.w_ode*L_ode, trunk_params)
                g_bc = grad_norm(L_bc, trunk_params)

                ratio_phys = (g_data / (g_phys + 1e-12)).clamp(1e-6, 1e6)
                ratio_bc = (g_data / (g_bc + 1e-12)).clamp(1e-6, 1e6)

                w_phys = float(np.clip(w_phys * float(ratio_phys.item()**cfg.lra_alpha), cfg.min_phys, cfg.max_phys))
                w_bc = float(np.clip(w_bc * float(ratio_bc.item()**cfg.lra_alpha), cfg.min_bc, cfg.max_bc))

                # EMA smooth
                w_phys_ema = cfg.ema_beta * w_phys_ema + (1-cfg.ema_beta) * w_phys
                w_bc_ema = cfg.ema_beta * w_bc_ema + (1-cfg.ema_beta) * w_bc

            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()

            # RAR update
            if (step > 0) and (cfg.rar_every > 0) and (step % cfg.rar_every == 0):
                model.eval()
                with torch.no_grad():
                    # candidate pool
                    N = ds.coords.shape[0]
                    Nt = ds.times.size
                    ni = np.random.randint(0, N, size=cfg.rar_candidates)
                    ti = np.random.randint(0, Nt, size=cfg.rar_candidates)
                    xyz_c = ds.coords[ni].astype(np.float32)
                    t_c = ds.times[ti][:,None].astype(np.float32)
                    xbc = torch.from_numpy(xyz_c).to(device)
                    tbc = torch.from_numpy(t_c).to(device)
                    xytzn_c = normalize_xyt(cfg, xbc, tbc, t_ref)
                    xytzn_c.requires_grad_(True)
                # need gradients; exit no_grad
                outc = model.forward_phys(xytzn_c, scen.repeat(cfg.rar_candidates,1), const, scales)
                r = pde_residual(cfg, outc, xytzn_c, const, scales).detach().abs().cpu().numpy().reshape(-1)
                # emphasize late times in the candidate selection (helps 100s low-power dynamics)
                if getattr(cfg, 'time_weight_alpha', 0.0) > 0.0:
                    tn = xytzn_c[:, 3:4].detach().cpu().numpy().reshape(-1)
                    r = r * (1.0 + float(cfg.time_weight_alpha) * np.clip(tn, 0.0, 1.0) ** float(cfg.time_weight_pow))
                topk = np.argsort(-r)[:cfg.rar_keep]
                pool = [(int(ni[i]), int(ti[i])) for i in topk]
                ds.rar_pool = pool

            # logging
            if step % 100 == 0:
                hist["step"].append(step)
                hist["lr"].append(float(opt.param_groups[0]["lr"]))
                hist["L"].append(float(loss.detach().cpu().item()))
                hist["Ldata"].append(float(L_data.detach().cpu().item()))
                hist["Laux"].append(float(L_aux.detach().cpu().item()))
                hist["Lphys"].append(float(L_phys.detach().cpu().item()))
                hist["Lbc"].append(float(L_bc.detach().cpu().item()))
                hist["Lode"].append(float(L_ode.detach().cpu().item()))
                hist["w_phys"].append(float(w_phys_ema))
                hist["w_bc"].append(float(w_bc_ema))

                dt_wall = time.time() - t0
                print(f"[{step:6d}] L={hist['L'][-1]:.4e} data={hist['Ldata'][-1]:.3e} aux={hist['Laux'][-1]:.3e} "
                      f"phys={hist['Lphys'][-1]:.3e} bc={hist['Lbc'][-1]:.3e} ode={hist['Lode'][-1]:.3e} "
                      f"wphys={w_phys_eff:.2e} wbc={w_bc_eff:.2e} tfrac={t_max_frac:.2f} wall={dt_wall/60:.1f}m")

                save_json(os.path.join(cfg.out_dir, "train_history.json"), hist)

            # checkpoint
            if (step > 0) and (cfg.ckpt_every > 0) and (step % cfg.ckpt_every == 0):
                ckpt_path = os.path.join(ckpt_dir, f"ckpt_step{step}.pt")
                torch.save({
                    "cfg": asdict(cfg),
                    "scales": scales,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "sched": sched.state_dict(),
                    "step": step,
                    "w_phys": w_phys,
                    "w_bc": w_bc,
                    "w_phys_ema": w_phys_ema,
                    "w_bc_ema": w_bc_ema,
                }, ckpt_path)

            # evaluation
            if (step > 0) and (cfg.eval_every > 0) and (step % cfg.eval_every == 0):
                model.eval()
                eval_dir = os.path.join(cfg.out_dir, f"eval_step{step}")
                for dse in (val_dss if val_dss else train_dss[:1]):
                    eval_dataset(cfg, model, dse, const_by_name[dse.spec.name], scales, eval_dir,
                                 times_to_plot=(0.5*t_ref, 0.8*t_ref, t_ref))
                # optional: also test
                if test_dss:
                    for dse in test_dss:
                        eval_dataset(cfg, model, dse, const_by_name[dse.spec.name], scales, eval_dir,
                                     times_to_plot=(0.5*float(dse.times.max()), float(dse.times.max()),))

    
    except KeyboardInterrupt:
        interrupted = True
        print("\n[Interrupt] KeyboardInterrupt received. Saving checkpoint...")
    except Exception:
        crashed = True
        print("\n[Crash] Exception occurred. Saving checkpoint...")
        raise
    finally:
        # always save a checkpoint (final / interrupt / crash)
        if last_step < 0:
            last_step = 0
        tag = "final" if (not interrupted and not crashed and last_step >= cfg.steps) else ("interrupt" if interrupted else "crash")
        ckpt_name = "ckpt_final.pt" if tag == "final" else f"ckpt_{tag}_step{last_step}.pt"
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        torch.save({
            "cfg": asdict(cfg),
            "scales": scales,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "step": last_step,
            "w_phys": w_phys,
            "w_bc": w_bc,
            "w_phys_ema": w_phys_ema,
            "w_bc_ema": w_bc_ema,
        }, ckpt_path)
        print(f"[Save] {tag} checkpoint -> {ckpt_path}")


def main():
    cfg = CFG()

    # Optional: choose a preset via environment variable BASELINE_PRESET.
    # Examples: "mlp_pinn", "ff_mlp_pinn", "siren_nofilm_pinn", "data_only_mlp", "ours"
    preset = os.environ.get("BASELINE_PRESET", "").strip()
    if preset:
        apply_model_preset(cfg, preset)
    # If you want to manually specify datasets, fill cfg.datasets here.
    # Otherwise it will auto-discover using cfg.base_dir.
    #
    # Example manual split (edit paths to your actual files):
    # cfg.datasets = [
    #   DatasetSpec(name="75s_5W", split="train", t_ref=75.0, P_laser=5.0,
    #       T_path=r"D:\COMSOL\data\5W\75s最终温度常规5W.csv",
    #       alpha_path=r"D:\COMSOL\data\5W\75s最终温度常规5W.csv",
    #       v_sub_path=r"D:\COMSOL\data\5W\75s最终v_sub_5W.csv",
    #       v_mesh_path=r"D:\COMSOL\data\5W\75s最终v_mesh_5W.csv",
    #       Qvol_path=r"D:\COMSOL\data\5W\75s最终体热5W.csv",
    #       qbnd_path=r"D:\COMSOL\data\5W\75s最终边界热5W.csv",
    #       dalpha_path=r"D:\COMSOL\data\5W\75s最终反应速率5W.csv",
    #   ),
    # ]
    train(cfg)

if __name__ == "__main__":
    main()

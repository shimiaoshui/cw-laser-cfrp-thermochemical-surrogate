# Physics-constrained surrogate modelling of hotspot-resolved thermochemical response in CW laser heating of CFRP

This repository contains the code package reorganized for journal submission and GitHub release for the manuscript:

**Physics-constrained surrogate modelling of hotspot-resolved thermochemical response in continuous-wave laser heating of carbon-fiber-reinforced polymer**

## Main scripts

- `stable_pinn_problem6_fixed_v4_ckpt.py`  
  Main physics-constrained training and checkpoint workflow.
- `problem6_baselines_suite.py`  
  Baseline model suite and shared utilities.
- `run_baseline_data_only_mlp.py`  
  Launcher for the data-only MLP baseline.
- `run_baseline_mlp_pinn.py`  
  Launcher for the baseline MLP-PINN model.
- `run_baseline_mlp_film_pinn.py`  
  Launcher for the FiLM-conditioned MLP-PINN baseline.
- `run_baseline_ff_mlp_pinn.py`  
  Launcher for the Fourier-feature MLP-PINN baseline.
- `run_baseline_siren_nofilm_pinn.py`  
  Launcher for the SIREN baseline without FiLM.
- `fig8_power_sweep_eval_phys.py`  
  Evaluation script for power-sweep and physics metrics.
- `make_problem6_paper_figures_v5_fixed.py`  
  Script for generating manuscript figures.

## Quick start

Create a Python environment and install the minimal dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then run scripts from the `src/` directory or add `src/` to your `PYTHONPATH`. For example:

```bash
cd src
python stable_pinn_problem6_fixed_v4_ckpt.py
python problem6_baselines_suite.py
python fig8_power_sweep_eval_phys.py
```




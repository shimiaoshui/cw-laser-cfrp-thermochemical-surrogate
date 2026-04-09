# coding: utf-8
# Robust runner for Jupyter: reload suite module so edits take effect without kernel restart.
import importlib

import problem6_baselines_suite as suite
importlib.reload(suite)

def main():
    cfg = suite.CFG()
    suite.apply_model_preset(cfg, "mlp_pinn")
    cfg.out_dir = "./out_baseline_mlp_pinn"
    suite.train(cfg)

if __name__ == "__main__":
    main()

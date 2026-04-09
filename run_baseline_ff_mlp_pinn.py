# coding: utf-8
import importlib
import problem6_baselines_suite as suite
importlib.reload(suite)

def main():
    cfg = suite.CFG()
    suite.apply_model_preset(cfg, "ff_mlp_pinn")
    cfg.out_dir = "./out_baseline_ff_mlp_pinn"
    suite.train(cfg)

if __name__ == "__main__":
    main()

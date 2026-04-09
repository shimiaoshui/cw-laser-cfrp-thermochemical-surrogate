# coding: utf-8
import importlib
import problem6_baselines_suite as suite
importlib.reload(suite)

def main():
    cfg = suite.CFG()
    suite.apply_model_preset(cfg, "mlp_film_pinn")
    cfg.out_dir = "./out_baseline_mlp_film_pinn"
    suite.train(cfg)

if __name__ == "__main__":
    main()

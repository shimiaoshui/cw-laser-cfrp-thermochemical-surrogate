# coding: utf-8
from problem6_baselines_suite import CFG, train, apply_model_preset

def main():
    cfg = CFG()
    apply_model_preset(cfg, "siren_nofilm_pinn")
    cfg.out_dir = "./out_baseline_siren_nofilm_pinn"
    train(cfg)

if __name__ == "__main__":
    main()

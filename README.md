# cameliND

Design nanobodies *in silico* with AI and physics — inspired by nature's camelids.

## Workflow

The pipeline consists of three stages:

1. **Design** — Generate nanobody designs against a target.
2. **Simulate** — Run molecular dynamics simulations on the designs.
3. **Analyze** — Analyze simulation trajectories for stability and binding properties.

## Acknowledgements

This project builds on [BoltzGen](https://github.com/HannesStark/boltzgen) by Hannes Stark and colleagues. I gratefully acknowledge the BoltzGen team for their open-source release of model weights, data, and inference and training code. Please cite their work:

```bibtex
@article{stark2025boltzgen,
  author  = {Stark, Hannes and Faltings, Felix and Choi, MinGyu and Xie, Yuxin and Hur, Eunsu and O'Donnell, Timothy John and Bushuiev, Anton and U\c{c}ar, Talip and Passaro, Saro and Mao, Weian and Reveiz, Mateo and Bushuiev, Roman and Pluskal, Tom\'a\v{s} and Sivic, Josef and Kreis, Karsten and Vahdat, Arash and Ray, Shamayeeta and Goldstein, Jonathan T. and Savinov, Andrew and Hambalek, Jacob A. and Gupta, Anshika and Taquiri-Diaz, Diego A. and Zhang, Yaotian and Hatstat, A. Katherine and Arada, Angelika and Kim, Nam Hyeong and Tackie-Yarboi, Ethel and Boselli, Dylan and Schnaider, Lee and Liu, Chang C. and Li, Gene-Wei and Hnisz, Denes and Sabatini, David M. and DeGrado, William F. and Wohlwend, Jeremy and Corso, Gabriele and Barzilay, Regina and Jaakkola, Tommi},
  title   = {BoltzGen: Toward Universal Binder Design},
  year    = {2025},
  doi     = {10.1101/2025.11.20.689494},
  journal = {bioRxiv}
}
```

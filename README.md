# GeoMat

Official implementation accompanying **GeoMat: Bridging Data Scarcity and
Generalization in Materials Discovery via Geometric Pre-training**.

GeoMat combines masked subgraph reconstruction, periodicity-aware geometric
denoising, and chemistry-informed contrastive learning. The implementation
supports CGCNN, MEGNET, and PaiNN backbones and includes the pre-training,
fine-tuning and plotting workflows used for the manuscript.

## Repository layout

```text
GeoMat/
├── analysis/                  # Publication figure scripts
├── data/                      # Data loaders, preprocessing, and split IDs
├── models/                    # CGCNN, MEGNET, and PaiNN architectures
├── source_data/               # Numerical source data for Figs. 2–4
├── finetune.py                # Non-MatBench regression tasks
├── finetune_matbench.py       # Official MatBench outer-fold evaluation
├── pretrain.py                # GeoMat self-supervised pre-training
├── finetune_config.yaml
├── pretrain_config.yaml
└── requirements.txt
```

Generated graph caches, checkpoints, run logs, and rendered figures are not
tracked by Git.

## Installation

The release was tested with Python 3.10, PyTorch 2.2.1, CUDA 12.1,
torch-geometric 2.7.0, and torch-scatter 2.1.2. Install PyTorch and the matching
torch-scatter wheel for the CUDA or CPU platform first, then install the
remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

PyTorch Geometric publishes platform-specific installation commands at
<https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html>.

## Data and pretrained checkpoints

The full processed task datasets and pretrained checkpoints are distributed in
the companion Zenodo data record. After downloading, use this layout:

```text
GeoMat/
├── checkpoint/
│   ├── geomat_cgcnn_pretrained.pth.tar
│   ├── geomat_megnet_pretrained.pth.tar
│   └── geomat_painn_pretrained.pth.tar
└── data/
    ├── Tc_AFM/dataset_with_atom_moments_final.csv
    ├── exciton_c2db/c2db_exciton_dataset.pkl
    ├── Tc_8055/superconductors_clean_dataset.pkl
    └── MAX/M2AX_data_3681.csv
```

The fixed `train.csv` and `test.csv` tables remain in Git so that sample
membership can be inspected without downloading the full archive. See
`data/README.md` for field definitions, provenance and split details.

The pre-training loader expects `OQMD_MP_tot.csv` with columns `id`, `cell`,
`numbers`, `pos_cart`, and the optional provenance column `dataset`.

## Pre-training

The full three-objective workflow is enabled with:

```bash
python pretrain.py /path/to/pretrain_data \
  --gnn MEGConv \
  --mask-subgraph random_walk \
  --pertlatt \
  --task union \
  --pretrain-config pretrain_config.yaml \
  --save-dir runs/pretrain_megnet
```

Use `CGConv` or `Painn` for the other backbones. The two-epoch gradient pilot
starts from a new model, uses unit loss weights, and writes the observed
gradient norms and suggested inverse-gradient coefficients:

```bash
python pretrain.py /path/to/pretrain_data \
  --gnn MEGConv \
  --mask-subgraph random_walk \
  --pertlatt \
  --task union \
  --gradient-pilot \
  --save-dir runs/gradient_pilot
```

Copy the selected fixed coefficients into `pretrain_config.yaml`, then start the
full pre-training run from a newly initialized model.

## MatBench fine-tuning

Set `data_name` and `train_ratio` in `finetune_config.yaml`, then run, for
example, the pretrained MEGNET model:

```bash
python finetune_matbench.py /path/to/matbench_cache \
  --gnn MEGConv \
  --resume pretrain \
  --pretrain-model checkpoint/geomat_megnet_pretrained.pth.tar \
  --save-dir runs/matbench
```

The script uses the five official MatBench outer folds. CGCNN and PaiNN reserve
one quarter of each non-test pool for validation. MEGNET follows its fixed-epoch
protocol and uses the complete non-test pool.

## Other fine-tuning tasks

`finetune.py` provides one entry point for AFM Néel temperature, C2DB exciton
binding energy, superconducting transition temperature, MAX phases, and JARVIS
regression. For example:

```bash
python finetune.py data/Tc_8055 \
  --dataset tc8055 \
  --gnn MEGConv \
  --pretrain-model checkpoint/geomat_megnet_pretrained.pth.tar \
  --save-dir runs/tc8055
```

AFM, exciton, Tc_8055, and MAX use the checked-in split-ID files in automatic
mode. The training command repeats three runs with training seeds 42, 43, and
44 by default. Split construction is documented in `data/README.md`.

## Reproduce the figures

The plotting scripts read only the checked-in CSV files under `source_data/`:

```bash
python analysis/plot_fig2a.py
python analysis/plot_fig2b.py
python analysis/plot_fig2c.py
python analysis/plot_fig3b.py
python analysis/plot_fig3d.py
python analysis/plot_fig3e_g.py --task afm
python analysis/plot_fig3e_g.py --task exciton
python analysis/plot_fig3e_g.py --task tc8055
python analysis/plot_fig4a_c.py
python analysis/plot_fig4d_e.py
```

The scripts create `outputs/` and write the generated figures there. 
`analysis/prepare_fig4d_e.py` regenerates the two t-SNE coordinate 
tables from a PaiNN checkpoint and user-supplied MatBench
cache. Column definitions and panel mappings are listed in
`source_data/README.md`.

## Citation

Citation metadata are provided in `CITATION.cff`. Please cite the accompanying
article and the archived software release once their identifiers are available.

## Contact

Questions about the code or data can be sent to Xiaolong Zou at
<xlzou@sz.tsinghua.edu.cn>.

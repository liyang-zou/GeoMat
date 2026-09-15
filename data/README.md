# Data

Full task datasets and pretrained checkpoints are distributed in the companion
Zenodo record. Extract the archive into the repository root while preserving
the following paths:

| Task | Input file | Records | Split files |
| --- | --- | ---: | --- |
| AFM Néel temperature | `Tc_AFM/dataset_with_atom_moments_final.csv` | 849 usable | `Tc_AFM/train.csv`, `Tc_AFM/test.csv` |
| C2DB exciton binding energy | `exciton_c2db/c2db_exciton_dataset.pkl` | 368 | `exciton_c2db/train.csv`, `exciton_c2db/test.csv` |
| Superconducting transition temperature | `Tc_8055/superconductors_clean_dataset.pkl` | 8,055 | `Tc_8055/train.csv`, `Tc_8055/test.csv` |
| MAX-phase formation enthalpy | `MAX/M2AX_data_3681.csv` | 3,681 | `MAX/train.csv`, `MAX/test.csv` |

The loaders create `processed/data.pt` on first use. Delete this cache after
changing an input file. `atom_init.json` provides the elemental features used
by CGCNN; MEGNET and PaiNN use learned atomic embeddings.

## Fixed splits

Each released split file contains one `id` column. The graph `id` field uses
the same stable material identifier, so the split is independent of DataLoader
ordering.

| Task | Split rule | Training | Test |
| --- | --- | ---: | ---: |
| AFM | 80/20 | 679 | 170 |
| Exciton | 80/20 | 294 | 74 |
| Tc_8055 | 80/20 | 6,444 | 1,611 |
| MAX | Types 1–3 train, Type 4 test | 2,482 | 1,199 |

The split IDs exactly match the `Data.id` values created by each loader. AFM 
uses the input table's first-column ID, exciton uses the complete C2DB `uid`,
Tc_8055 uses the `compound_id`, and MAX uses `TypeN-formula`.

`finetune.py` reads these files in automatic split mode. MEGNET trains on the
complete non-test pool. CGCNN and PaiNN reserve one quarter of that pool for
validation with seed 42, giving overall 60/20/20 train/validation/test splits.
The three repeated runs use training seeds 42, 43 and 44. JARVIS uses one fixed
80/10/10 split with seed 42 and the same three training seeds.

## Dataset preparation

### AFM

`dataset_with_atom_moments_final.csv` was derived from
[MAGNDATA](https://www.cryst.ehu.es/magndata/) after removing duplicate and
non-crystalline structures. The loader retains rows with a parseable `Tc_new`
target and an overall mean magnetic-moment vector equal to zero in PyTorch
float32 arithmetic. This leaves 849 unique structures from 1,466 input rows.

### C2DB exciton data

`getdata.py` selects entries with exciton binding energy `E_B > 0.01` eV from
the C2DB ASE database. The 368-record snapshot was checked against a source
`c2db.db` file.
The processed file can be regenerated with:

```bash
python data/exciton_c2db/getdata.py \
  --database /path/to/c2db.db \
  --skip-download \
  --output data/exciton_c2db/c2db_exciton_dataset.pkl
```

### Tc_8055

`pipeline_superconductor.py` reconstructs the snapshot from Materials Cloud
record `3kbt5-r3n56`. It retains calculations containing `geo_opt.cif` and the
Allen–Dynes transition temperature at `mu*=0.10` from `McMillan.dat`:

```bash
python data/Tc_8055/pipeline_superconductor.py \
  --output data/Tc_8055/superconductors_clean_dataset.pkl
```

### MAX phases

The MAX snapshot contains the four motif labels used in the manuscript.
Type1 consists of source `type1` structures with one distinct M element; Type4
contains source `type1` structures with two distinct M elements. Source
`type2` and `type3` rows form Type2 and Type3. All Type4 structures are held
out for testing.

The CSV supplies the DFT-evaluated CIF geometries and formation enthalpies used
for screening. It does not contain the historical common-cell initial
geometries described for the IS2RE setup.

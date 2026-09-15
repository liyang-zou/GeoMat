# Figure source data

Each CSV contains the numerical values used by one or more manuscript panels.
All plotting scripts read these tables directly and write images below
`outputs/`.

| File | Panel | Contents |
| --- | --- | --- |
| `Source_Data_Fig2a.csv` | Fig. 2a | Per-task MAEs and relative improvements for the three backbones |
| `Source_Data_Fig2b.csv` | Fig. 2b | Five-fold data-efficiency means and standard deviations |
| `Source_Data_Fig2c.csv` | Fig. 2c | Pre-training scale and downstream relative improvement |
| `Source_Data_Fig3b_3d.csv` | Fig. 3b,d | MAX IDs, DFT values, scratch/pretrained predictions, and stability labels |
| `Source_Data_Fig3e.csv` | Fig. 3e | AFM test IDs, ground truth, prediction, and target labels |
| `Source_Data_Fig3f.csv` | Fig. 3f | Exciton test IDs, ground truth, prediction, and target labels |
| `Source_Data_Fig3g.csv` | Fig. 3g | Tc_8055 test IDs, ground truth, prediction, and target labels |
| `Source_Data_Fig4a.csv` | Fig. 4a | JDFT2D five-fold ablation results |
| `Source_Data_Fig4b.csv` | Fig. 4b | GVRH five-fold ablation results |
| `Source_Data_Fig4c.csv` | Fig. 4c | MP_Gap five-fold ablation results |
| `Source_Data_Fig4d.csv` | Fig. 4d | MP_E_Form material IDs, fold, t-SNE coordinates, and labels |
| `Source_Data_Fig4e.csv` | Fig. 4e | MP_Is_Metal material IDs, fold, t-SNE coordinates, and labels |

Rows in Fig. 3e-g correspond exactly to the released task-specific test-ID
sets. Stability and enrichment labels are derived from ground-truth values;
model predictions determine only candidate ranking.

改好了，结构和事实全部保留（文件名、命令、数据说明、Chronos-2 的范围说明、引用条目），只把措辞和排版理顺，删掉重复表述和口语化句子。

```markdown
# C-MAPSS Remaining Useful Life (RUL) Prediction

Research code for remaining useful life (RUL) prediction on the NASA C-MAPSS
turbofan engine degradation benchmark. The repository provides two candidate
network implementations together with scripts for visualising degradation
trajectories and model performance.

> **The dataset is not distributed with this repository.** Obtain C-MAPSS
> independently and either place the files in the expected directory or point
> `CMAPSS_DATA_DIR` at the directory that contains them (see §2).

## Contents

| File | Description |
|---|---|
| `LGTF-Net.py` | Local-Global Temporal Fusion Network (LGTF-Net): a local convolutional branch, an LSTM temporal branch, and adaptive gated fusion. |
| `proposed_method(1).py` | Sensor-aware Bidirectional Temporal Stage-Gated Network (SBT-SGNet): sensor-level attention, a bidirectional LSTM, stage-aware temporal fusion, and a monotonic-consistency loss. |
| `visualize(1).py` | Sensor and degradation visualisations for FD001, and operating-setting visualisations for FD004. |
| `visua_plot.py` | Model-comparison metrics, prediction-versus-ground-truth scatter plots, absolute-error distributions, and training curves, built from the prediction CSV files. |
| `requirements.txt` | Python package dependencies. |

> **Note on the two models.** LGTF-Net and SBT-SGNet are separate implementations
> and should be treated as distinct candidate methods, not as variants of one
> architecture. Before submission, check that the method described in the
> manuscript matches the code actually used to produce the reported results.

## 1. Environment

Python 3.10 or 3.11 is recommended. Create and activate a virtual environment,
then install the dependencies:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

**Scope of the current scripts.** None of the four scripts in this repository
loads the Chronos-2 foundation model. A manuscript that describes a frozen
Chronos-2 adaptation method therefore needs the corresponding implementation and
reproducible instructions added here before this repository can be presented as
reproducing that method.

## 2. Obtaining the dataset

Download the NASA C-MAPSS data from its official distribution source and comply
with the terms that accompany it. The raw dataset should not be uploaded to this
repository.

By default the scripts expect the following layout:

```text
data/
└── CMAPSSData/
    ├── train_FD001.txt
    ├── test_FD001.txt
    ├── RUL_FD001.txt
    ├── train_FD004.txt
    ├── test_FD004.txt
    └── RUL_FD004.txt
```

Only the files required by the experiment you intend to run need to be present.
Alternatively, set `CMAPSS_DATA_DIR` to the directory holding the C-MAPSS text
files.

**Windows PowerShell**
```powershell
$env:CMAPSS_DATA_DIR = "D:\datasets\CMAPSSData"
```

**macOS/Linux**
```bash
export CMAPSS_DATA_DIR="/path/to/CMAPSSData"
```

## 3. Running the model scripts

From the repository root:

```bash
python "LGTF-Net.py"
python "proposed_method(1).py"
```

Both scripts default to FD001 and write prediction CSVs, training histories, and
model checkpoints to `results/`. Runtime depends on the hardware and on the
PyTorch configuration, and may be considerable.

The code uses a fixed random seed and an engine-wise train/validation split.
Results can nevertheless differ across hardware and software versions; re-run
the experiments and compare the produced metrics with the manuscript before
reporting them as reproduced.

## 4. Dataset visualisations

`visualize(1).py` expects the FD001 and FD004 training files:

```bash
python "visualize(1).py"
```

Figures are written to `figures/`.

## 5. Performance comparison plots

`visua_plot.py` expects prediction CSV files for the models listed in the script
(CNN, LGTF-Net, Linear, LSTM, MLP, Proposed, and Transformer) in `results/`,
together with the SBT-SGNet training-history CSV. The two model scripts in this
repository do not generate every baseline file on their own: run the baseline
implementations first and place their prediction files in `results/` under the
filenames configured near the top of `visua_plot.py`.

```bash
python visua_plot.py
```

Publication-oriented figures and a performance-comparison CSV are written to
`figures/`.

## Reproducibility and reporting

- Keep the train/validation/test protocol identical across all methods being compared.
- Report the dataset subset, random seed, preprocessing, RUL cap, window length,
  early-stopping settings, and software versions.
- Do not state that a model outperforms a baseline unless the comparison was run
  under a consistent protocol and the measured results support the claim.
- Prediction CSVs, checkpoints, and the raw dataset are not committed by default;
  regenerate them locally.

## Citation

If you use this code in academic work, please cite the paper associated with the
method you use, the original C-MAPSS dataset source, and the relevant
model/software papers. The final paper reference will be added here once it is
published:

```bibtex
@article{xie2026rul,
  title   = {Sensor-Relation-Guided Adaptation of a Frozen Time-Series Foundation
             Model for Remaining Useful Life Prediction},
  author  = {Xie, C. and Xu, W. and Zhang, M. and Shao, H.},
  journal = {Under Review},
  year    = {2026},
  doi     = {DOI}
}
```

## Academic homepage

Publications, citations, and research updates:

[Google Scholar Profile](https://scholar.google.com/citations?user=d3mvChQAAAAJ&hl=en)

If this project is useful in your research, citing the related publications is
much appreciated.

❤️ Have a nice day! (English)
🌸 素敵な一日を！ (Japanese)
✨ Bonne journée ! (French)
😊 ¡Que tengas un buen día! (Spanish)
🌿 Einen schönen Tag noch! (German)
☕ 좋은 하루 보내세요! (Korean)
🚀 Желаю хорошего дня! (Russian)
🍃 愿你此行如风，自有繁花相送。 (Chinese)
```

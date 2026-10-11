# C-MAPSS Remaining Useful Life (RUL) Prediction

This repository contains research code for remaining useful life (RUL) prediction on the NASA C-MAPSS turbofan engine degradation benchmark. It provides two candidate neural-network implementations and scripts for visualizing degradation trajectories and model performance.

**The dataset is not included.** Each user must obtain C-MAPSS independently and place the files in the expected directory or set the `CMAPSS_DATA_DIR` environment variable.

## Contents

| File | Description |
|---|---|
| `LGTF-Net.py` | Local-Global Temporal Fusion Network (LGTF-Net): a local convolutional branch, an LSTM temporal branch, and adaptive gated fusion. |
| `proposed_method(1).py` | Sensor-aware Bidirectional Temporal Stage-Gated Network (SBT-SGNet): sensor-level attention, bidirectional LSTM, stage-aware temporal fusion, and a monotonic-consistency loss. |
| `visualize(1).py` | Generates sensor/degradation visualizations for FD001 and operating-setting visualizations for FD004. |
| `visua_plot.py` | Plots model-comparison metrics, prediction-vs-ground-truth scatter plots, absolute-error distributions, and training curves from prediction CSV files. |
| `requirements.txt` | Python package dependencies. |

> **Important research note:** LGTF-Net and SBT-SGNet are two distinct model implementations. They should be treated as separate candidate methods/experiments, not as the same architecture. Verify that the method described in the manuscript matches the code and the reported results before submission.

## 1. Environment

Python 3.10 or 3.11 is recommended. Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

The Chronos-2 foundation model is **not called by the four scripts currently included in this repository**. If the manuscript describes a frozen Chronos-2 adaptation method, its actual Chronos-2 implementation and corresponding reproducible instructions must be added before claiming this repository reproduces that method.

## 2. Obtain the dataset yourself

Obtain the NASA C-MAPSS dataset from its official distribution source and follow the dataset's applicable terms. Do not upload the raw dataset to this repository.

By default, the scripts expect this layout:

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

Only the files required for the experiment you run need to be present. Alternatively, set `CMAPSS_DATA_DIR` to the directory containing the C-MAPSS text files.

**Windows PowerShell**
```powershell
$env:CMAPSS_DATA_DIR = "D:\datasets\CMAPSSData"
```

**macOS/Linux**
```bash
export CMAPSS_DATA_DIR="/path/to/CMAPSSData"
```

## 3. Run the model scripts

From the repository directory:

```bash
python "LGTF-Net.py"
python "proposed_method(1).py"
```

Both scripts use FD001 by default and write prediction CSVs, training histories, and model checkpoints to `results/`. The scripts may require substantial runtime depending on your hardware and PyTorch configuration.

The current code uses a fixed random seed and an engine-wise train/validation split. Results can still vary across hardware/software versions. Re-run experiments and compare the produced metrics with the manuscript before reporting them as reproduced.

## 4. Generate dataset visualizations

`visualize(1).py` expects FD001 training data and FD004 training data:

```bash
python "visualize(1).py"
```

It saves figures to `figures/`.

## 5. Generate performance comparison plots

`visua_plot.py` expects prediction CSV files from the listed models (CNN, LGTF-Net, Linear, LSTM, MLP, Proposed, and Transformer) in `results/`, as well as the SBT-SGNet training-history CSV. The two model scripts in this repository do not generate every baseline file by themselves. Run your baseline implementations first and place their prediction files in `results/` using the filenames configured near the top of `visua_plot.py`.

```bash
python visua_plot.py
```

The script writes publication-oriented figures and a performance-comparison CSV under `figures/`.

## Reproducibility and reporting

- Keep the train/validation/test protocol consistent across methods.
- Report the dataset subset, random seed, preprocessing, RUL cap, window length, early-stopping settings, and software versions.
- Do not describe a model as outperforming a baseline unless the comparison was run under a consistent protocol and the measured results support that claim.
- The prediction CSVs, checkpoints, and raw dataset are not committed by default; regenerate them locally.

## Citation

If you use this code in academic work, please cite the related paper(s) associated with the method you use. Please also cite the original C-MAPSS dataset source and the relevant model/software papers. Add the final paper citation here once the manuscript is published:

```bibtex
@article{your_rul_paper,
  title   = {Sensor-Relation-Guided Adaptation of a Frozen Time-Series Foundation Model for Remaining Useful Life Prediction},
  author  = {CC Xie, WY Xu, M Zhang and HJ Shao},
  journal = {Under Review},
  year    = {2026},
  doi     = {DOI}
}
```

## Academic Homepage

For publications, citations, and research updates, please visit my Google Scholar profile:

[Google Scholar Profile](https://scholar.google.com/citations?user=d3mvChQAAAAJ&hl=en)

If you find this project useful in your research, I would greatly appreciate it if you cite my related publications. Thank you for your support and citations!

❤️ Have a nice day! (English)  
🌸 素敵な一日を！ (Japanese)  
✨ Bonne journée ! (French)  
😊 ¡Que tengas un buen día! (Spanish)  
🌿 Einen schönen Tag noch! (German)  
☕ 좋은 하루 보내세요! (Korean)  
🚀 Желаю хорошего дня! (Russian)  
🍃 愿你此行如风，自有繁花相送。 (Chinese)

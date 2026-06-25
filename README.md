# Sensing-Aided Drone Beam Prediction

This repository implements multi-modal beam prediction on the DeepSense6G Scenario 23 dataset. The task is to use the past 8 frames of sensing information to predict the optimal beam indices for the next 5 future time steps.

The current code supports:

- Proposed multi-modal Transformer model
- Position-only Transformer baseline
- Image-only Transformer baseline
- Multi-modal GRU baseline
- Result visualization and communication-metric evaluation in `results_eval.ipynb`

## Project Structure

```text
.
|-- main.py                  # Training entry point
|-- model.py                 # Model definitions
|-- data_loader.py           # PyTorch dataset loader
|-- preprocess_data.py       # Scenario 23 preprocessing script
|-- results_eval.ipynb       # Result evaluation and plotting notebook
|-- Model_log/               # Saved model weights
|-- Figures/                 # Evaluation caches, figures, and result tables
|-- scenario23_dev/          # DeepSense6G Scenario 23 dataset, created by the user
|-- data_windows.csv         # Generated training windows
`-- test_windows.csv         # Generated test windows
```

## Dataset Preparation

The raw dataset is not included in this repository. Please download **Scenario 23** from the official DeepSense6G website:

[https://www.deepsense6g.net/scenarios/Scenarios%2020-29/scenario-23](https://www.deepsense6g.net/scenarios/Scenarios%2020-29/scenario-23)

After downloading and extracting the dataset, place it under the project root as:

```text
beam-prediction/
`-- scenario23_dev/
    |-- scenario23.csv
    |-- unit1/
    `-- unit2/
```


## Preprocessing

Run:

```bash
python preprocess_data.py
```

This script reads:

```text
scenario23_dev/scenario23.csv
```

and generates sliding-window samples:

```text
data_windows_all.csv
data_windows.csv
test_windows.csv
```

The window setting is:

```text
T_in  = 8   # past frames used as input
T_out = 5   # future beam indices to predict
```

Each generated sample contains:

- `unit1_rgb`: past 8 RGB image paths
- `unit2_loc`: past 8 normalized UAV locations
- `unit2_speed`: past 8 UAV speed values, kept for analysis
- `unit2_distance`: past 8 normalized distances
- `unit2_height`: past 8 normalized heights
- `unit1_beam_index`: future 5 beam labels
- `unit1_pwr_60ghz`: future beam power values for communication-metric evaluation
- `current_speed`, `current_distance`, `current_height`: physical values used for subset analysis

## Training

The training entry point is:

```bash
python main.py
```

Available model names:

```text
multimodal       # Proposed Multi-Modal Transformer
multimodal_gru   # Multi-Modal GRU baseline
position         # Position-only Transformer
image            # Image-only Transformer
```

Example commands:

```bash
# Proposed multi-modal Transformer
python main.py --model multimodal --data-csv data_windows.csv --root-dir . --epochs 30 --batch-size 16 --lr 1e-3

# Multi-modal GRU baseline
python main.py --model multimodal_gru --data-csv data_windows.csv --root-dir . --epochs 30 --batch-size 16 --lr 1e-3

# Position-only baseline
python main.py --model position --data-csv data_windows.csv --root-dir . --epochs 30 --batch-size 16 --lr 1e-3

# Image-only baseline
python main.py --model image --data-csv data_windows.csv --root-dir . --epochs 30 --batch-size 16 --lr 1e-3
```


Saved weights are written to:

```text
Model_log/
```

with names similar to:

```text
multimodal_best_model_train_YYYYMMDD_HHMMSS.pth
multimodal_gru_best_model_train_YYYYMMDD_HHMMSS.pth
position_best_model_train_YYYYMMDD_HHMMSS.pth
image_best_model_train_YYYYMMDD_HHMMSS.pth
```

## Result Evaluation

Evaluation and plotting are handled by:

```text
results_eval.ipynb
```

Before running the notebook, check the model weight paths in the `MODEL_CONFIGS` and `GRU_CONFIG` cells. They should match the actual files under `Model_log/`.

The notebook uses:

```text
test_windows.csv
```

as the test set.


The notebook saves figures and result files under:

```text
Figures/
```

Common outputs include:

```text
modal_accuracy.pdf              # Overall Top-1/Top-3/Top-5 bar chart
acc_curve.pdf                   # Per-step Top-k accuracy curves
spectral_efficiency.pdf         # Spectral efficiency CDF
effective_throughput.pdf        # Effective throughput CDF
speed_distance_height.pdf       # Accuracy under speed/distance/height groups
combined_cm.pdf                 # Confusion matrices for future steps
power_comparison_curve.pdf      # Predicted vs. optimal beam power curve
index_rsrp.pdf                  # One-sample RSRP spectrum illustration
```

## Reproducibility Notes

- Download Scenario 23 manually from DeepSense6G and place it in `scenario23_dev/`.
- Run `preprocess_data.py` before training if `data_windows.csv` and `test_windows.csv` are not available.
- Run all model trainings needed by `results_eval.ipynb`, or update the notebook to point to your available weights.
- The notebook intentionally caches inference outputs in `Figures/*.pt` so later plots and tables do not repeatedly run model inference.



## Citation

If you find this repository useful for your research, please consider citing our paper:

```bibtex
@article{nie2026robust,
  title   = {Robust Predictive mmWave Beamforming for UAV Communications via Vision-Position Fusion},
  author  = {Nie, Jiali and others},
  journal = {IEEE Transactions on Vehicular Technology},
  year    = {2026},
  note    = {Under review}
}



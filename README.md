# PhysioNet - ECG Image Digitization

ECGs are vital for heart disease diagnosis, but many exist only as images, not time-series data needed for analysis. Converting these into digital signals is challenging due to artifacts, variations, and noise, yet essential to unlock decades of data for improved AI-driven cardiovascular diagnosis and care.

## Competition Overview

This project tackles the [PhysioNet ECG Image Digitization competition](https://kaggle.com/competitions/physionet-ecg-image-digitization) on Kaggle. The goal is to extract time-series ECG data from scanned/photographed paper ECG printouts.

### Key Challenges
- **Image Quality Variations**: Original, scanned, photographed, damaged, stained, moldy images
- **12 ECG Leads**: I, II, III, aVR, aVL, aVF, V1-V6
- **Different Durations**: Lead II (10s), others (2.5s)
- **Variable Sampling Rates**: Different fs values across records
- **Evaluation**: Modified SNR (Signal-to-Noise Ratio) with alignment compensation

### Prizes
Total prize pool: **$50,000**

## Project Structure

```
Digitization-of-ECG-Images/
├── data/
│   ├── raw/                    # Raw competition data (download from Kaggle)
│   └── processed/              # Processed data
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
├── src/
│   ├── config.py              # Project configuration
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataloader.py      # Data loading utilities
│   ├── models/                # Model implementations
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py         # Evaluation metrics (SNR)
│       └── visualization.py   # Plotting utilities
├── models/                     # Saved model checkpoints
├── experiments/                # Experiment logs
├── submissions/                # Submission files
├── requirements.txt            # Python dependencies
└── README.md
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Digitization-of-ECG-Images
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Competition Data

#### Option A: Using Kaggle API (Recommended)

```bash
# Install Kaggle API
pip install kaggle

# Configure Kaggle credentials
# Download kaggle.json from Kaggle account settings
# Place it in ~/.kaggle/ (Linux/Mac) or C:\Users\<username>\.kaggle\ (Windows)

# Download competition data
kaggle competitions download -c physionet-ecg-image-digitization

# Extract to data/raw/
unzip physionet-ecg-image-digitization.zip -d data/raw/
```

#### Option B: Manual Download

1. Go to the [competition data page](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/data)
2. Accept the competition rules
3. Download all files
4. Extract to `data/raw/`

### 5. Verify Setup

Run the EDA notebook to verify everything is working:

```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

## Dataset Description

### Training Data
- **train.csv**: Metadata (id, fs, sig_len)
- **train/[id]/[id].csv**: Time-series data for 12 leads
- **train/[id]/[id]-[segment].png**: Multiple image variants:
  - 0001: Original color ECG
  - 0003: Color scan
  - 0004: Black & white scan
  - 0005: Mobile photos
  - 0006: Screen photos
  - 0009: Stained images
  - 0010: Damaged images
  - 0011: Mold (color)
  - 0012: Mold (B&W)

### Test Data
- **test.csv**: Metadata (id, lead, fs, number_of_rows)
- **test/[id].png**: Single image per record

### Submission Format
- **submission.parquet**: Predictions in parquet format
  - `id`: {base_id}_{row_id}_{lead}
  - `value`: Predicted amplitude in mV

## Evaluation Metric

The competition uses a **modified Signal-to-Noise Ratio (SNR)** metric:

```python
SNR(dB) = 10 * log10(signal_power / noise_power)
```

Key features:
1. **Time Alignment**: Corrects horizontal shifts up to 0.2 seconds
2. **Vertical Adjustment**: Removes constant amplitude offsets
3. **Multi-lead Aggregation**: Combines all 12 leads before computing SNR
4. **Final Score**: Average SNR across all test records

See `src/utils/metrics.py` for implementation.

## Approach

### Phase 1: Understanding (Current)
- [x] Project setup
- [x] Data exploration (EDA)
- [x] Evaluation metric implementation
- [ ] Literature review

### Phase 2: Baseline
- [ ] Image preprocessing pipeline
- [ ] Simple signal extraction (CV-based)
- [ ] Baseline model evaluation
- [ ] Initial submission

### Phase 3: Deep Learning
- [ ] CNN-based encoder
- [ ] Sequence decoder (LSTM/Transformer)
- [ ] End-to-end training
- [ ] Model optimization

### Phase 4: Advanced
- [ ] Ensemble methods
- [ ] Test-time augmentation
- [ ] Post-processing refinement
- [ ] Final submission

## Key Insights

### ECG Paper Specifications
- **Amplitude**: 10mm = 1mV (standard calibration)
- **Speed**: 25mm/s or 50mm/s (paper speed)
- **Grid**: Standard ECG paper has grid lines for measurement

### Technical Considerations
1. **Lead Layout**: ECGs typically show 12 leads in standard arrangement
2. **Rhythm Strip**: Lead II usually has longer duration (10s)
3. **Calibration Pulse**: Look for calibration markers in images
4. **Artifacts**: Handle rotations, stains, damage, scanning artifacts

## Useful Resources

### Competition
- [Competition Homepage](https://www.kaggle.com/competitions/physionet-ecg-image-digitization)
- [Discussion Forum](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/discussion)

### Papers
1. Shivashankara et al. (2024) - ECG-Image-Kit: Synthetic image generation
2. Reyna et al. (2024) - ECG-Image-Database: Real-world artifacts
3. PhysioNet Challenge 2024 - Previous challenge solutions

### Tools
- [ECG-Image-Kit](https://github.com/alphanumericslab/ecg-image-kit): Image generation toolbox
- OpenCV: Image processing
- SciPy: Signal processing

## Contributing

Feel free to contribute by:
- Opening issues for bugs or suggestions
- Submitting pull requests with improvements
- Sharing insights in discussions

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgements

Competition organized by:
- Emory University
- PhysioNet
- Kaggle

Dataset citations available in competition documentation.

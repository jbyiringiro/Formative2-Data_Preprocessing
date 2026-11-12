# Multimodal User Authentication & Product Recommendation

Formative 2 — Multimodal Data Preprocessing.

A user is authenticated with **face recognition**, the **product recommendation model** predicts the product category they are likely to buy, and the prediction is only released after **voiceprint verification** confirms the same user. Any failed check ends in *access denied*.

```
face image ──► face model ──► recognised? ──no──► ACCESS DENIED
                                   │yes
                                   ▼
                     product model (customer features)
                                   │
voice sample ──► voiceprint model ──► same user? ──no──► ACCESS DENIED
                                   │yes
                                   ▼
                       display predicted product
```

## Repository structure

```
data/
  customer_social_profiles.csv     raw social-media profiles (155 rows)
  customer_transactions.csv        raw transactions (150 rows)
  merged_customer_data.csv         merged dataset with engineered features (150 × 15)
  images/<Member>/<Member>_<expression>.jpg   3 expressions per member
  audio/<Member>/<Member>_recording<N>.m4a    2 recordings per member
  image_features.csv               image features (48 × 99)
  audio_features.csv               audio features (32 × 42)
models/
  face_model.pkl                   facial recognition model
  voiceprint_model.pkl             voiceprint verification model
  product_category_model.joblib    product recommendation model (+ feature list)
notebooks/
  01_data_merge_eda_product_model.ipynb
  02_image_processing_face_model.ipynb
  03_audio_processing_voiceprint_model.ipynb
src/                               pipeline scripts and command-line app (in progress)
```

## Pipeline

### 1. Data merge and EDA — `notebooks/01_data_merge_eda_product_model.ipynb`
- **Cleaning:** 5 exact duplicate social-profile rows removed; `purchase_date` converted to datetime; 10 missing `customer_rating` values kept (not imputed, and not used as a feature).
- **Join key:** social IDs (`A178`) and transaction IDs (`178`) are standardised into `customer_id_key`.
- **Merge:** social records are aggregated to one row per customer (mean engagement, mean purchase interest, record and platform counts, sentiment counts), then **left-joined onto transactions** (`validate="many_to_one"`). All 150 transactions are kept; 117 have a matching social profile, 33 do not (`has_social_profile = 0`).
- **EDA:** summary statistics, variable types, distributions, outlier boxplots, correlation heatmaps, and features by product category.

### 2. Images — `notebooks/02_image_processing_face_model.ipynb`
- 4 members × 3 expressions (neutral, smiling, surprised) = 12 images, resized to 128 × 128.
- Augmentations per image: 90° rotation, horizontal flip, grayscale.
- Features: normalised colour histograms, 32 bins × 3 channels = 96 features → `data/image_features.csv`.

### 3. Audio — `notebooks/03_audio_processing_voiceprint_model.ipynb`
- 4 members × 2 recordings = 8 clips; waveforms and spectrograms plotted and interpreted for each member.
- Augmentations per recording: pitch shift (+1 semitone), time stretch (0.95×), background noise (SNR 25 dB).
- Features: 13 MFCCs plus spectral roll-off, RMS energy, spectral centroid, spectral bandwidth and zero-crossing rate (mean and std of each) = 36 features → `data/audio_features.csv`.

### 4. Models
| Model | Algorithm | Evaluation (from the notebooks) |
|---|---|---|
| Face recognition | Random Forest on colour histograms | held-out *surprised* images: accuracy 1.00, macro F1 1.00, log loss 0.378 |
| Voiceprint verification | StandardScaler + Logistic Regression | held-out second recordings: 4/4 speakers identified (confusion matrix) |
| Product recommendation | median imputation + StandardScaler + tuned Logistic Regression | customer-grouped test set: accuracy 0.194, macro F1 0.142 |

Consolidated training scripts reporting accuracy, F1 and loss for all three models are being added under `src/` (Task 4).

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

- The notebooks run in **Google Colab** (data read from the team Drive) or **locally** (data read from `data/`). They detect which one automatically.
- Loading `.m4a` audio locally needs [FFmpeg](https://ffmpeg.org/) on your `PATH`.
- `scikit-learn` is pinned to the version the saved models were built with.

## Team

| Member | Contribution |
|---|---|
| Janviere | Data cleaning and merge, EDA, product recommendation model |
| Yohan | Image collection and processing, facial recognition model |
| Thomas | Audio collection and processing, voiceprint verification model |
| Josue | Model creation and evaluation (Task 4), repository organisation |

## Status

- [x] Data merge, EDA, merged dataset
- [x] Image and audio collection, augmentation, feature files
- [x] Face, voiceprint and product models (notebook versions)
- [ ] Consolidated model training and evaluation scripts (`src/`)
- [ ] Command-line demo: full transaction and unauthorized attempts
- [ ] Report, demo video, contribution sheet

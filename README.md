# Multimodal User Authentication & Product Recommendation

Formative 2 Multimodal Data Preprocessing.

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
  member_customer_map.csv          customer profile linked to each team member (demo assumption)
models/
  face_recognition.joblib          final facial recognition model (+ classes, features, threshold)
  voiceprint_verification.joblib   final voiceprint verification model (+ classes, features, threshold)
  product_recommendation.joblib    final product recommendation model (+ classes, features)
  face_model.pkl, voiceprint_model.pkl, product_category_model.joblib
                                   earlier versions saved by notebooks 01–03
notebooks/
  01_data_merge_eda_product_model.ipynb
  02_image_processing_face_model.ipynb
  03_audio_processing_voiceprint_model.ipynb
  04_model_training_evaluation.ipynb
src/
  features.py                      feature extraction for new photos, recordings and customers
  evaluation.py                    algorithms compared, metrics, open-set thresholds
  train_face_model.py              facial recognition training and evaluation
  train_voice_model.py             voiceprint verification training and evaluation
  train_product_model.py           product recommendation training and evaluation
  train_all.py                     trains all three models, writes outputs/
  inference.py                     multimodal decision logic (face → product → voice)
outputs/
  model_metrics.csv                accuracy, macro F1 and log loss per model / algorithm / evaluation
  open_set_thresholds.csv          rejection thresholds with FAR / FRR
  figures/                         comparison, confusion matrices, threshold analysis
```

## Pipeline

### 1. Data merge and EDA `notebooks/01_data_merge_eda_product_model.ipynb`

- **Cleaning:** 5 exact duplicate social-profile rows removed; `purchase_date` converted to datetime; 10 missing `customer_rating` values kept (not imputed, and not used as a feature).
- **Join key:** social IDs (`A178`) and transaction IDs (`178`) are standardised into `customer_id_key`.
- **Merge:** social records are aggregated to one row per customer (mean engagement, mean purchase interest, record and platform counts, sentiment counts), then **left-joined onto transactions** (`validate="many_to_one"`). All 150 transactions are kept; 117 have a matching social profile, 33 do not (`has_social_profile = 0`).
- **EDA:** summary statistics, variable types, distributions, outlier boxplots, correlation heatmaps, and features by product category.

### 2. Images `notebooks/02_image_processing_face_model.ipynb`

- 4 members × 3 expressions (neutral, smiling, surprised) = 12 images, resized to 128 × 128.
- Augmentations per image: 90° rotation, horizontal flip, grayscale.
- Features: normalised colour histograms, 32 bins × 3 channels = 96 features → `data/image_features.csv`.

### 3. Audio `notebooks/03_audio_processing_voiceprint_model.ipynb`

- 4 members × 2 recordings = 8 clips; waveforms and spectrograms plotted and interpreted for each member.
- Augmentations per recording: pitch shift (+1 semitone), time stretch (0.95×), background noise (SNR 25 dB).
- Features: 13 MFCCs plus spectral roll-off, RMS energy, spectral centroid, spectral bandwidth and zero-crossing rate (mean and std of each) = 36 features → `data/audio_features.csv`.

### 4. Model training and evaluation `src/`, `notebooks/04_model_training_evaluation.ipynb`

Logistic Regression, Random Forest and XGBoost are compared for every model against a no-skill baseline, always on data the model has never seen:

| Model                   | Evaluation                                    | Selected            | Accuracy | Macro F1 | Log loss | Threshold (FAR / FRR) |
| ----------------------- | --------------------------------------------- | ------------------- | -------- | -------- | -------- | --------------------- |
| Face recognition        | leave-one-expression-out CV (48 images)       | Random Forest       | 1.00     | 1.00     | 0.23     | 0.85 (0% / 27%)       |
| Voiceprint verification | leave-one-recording-out CV (32 clips)         | Logistic Regression | 0.91     | 0.90     | 0.24     | 0.83 (25% / 25%)      |
| Product recommendation  | customer-grouped 5-fold CV (150 transactions) | Random Forest       | 0.21     | 0.18     | 1.87     | —                     |

- **Thresholds** come from an open-set test: each member is left out of training in turn and presented as an unknown person (FAR = unknown people accepted, FRR = genuine members rejected).
- **Face and voice** are selected on the verification error (FAR + FRR); the **product** model on macro F1.
- **Product data:** no column (including amount, rating and date) predicts the product category better than chance, so the product model performs at baseline level (most-frequent-category accuracy 0.23). See notebook 04, section 6.

### 5. Multimodal decision logic `src/inference.py`

1. Face recognised as a team member with confidence ≥ 0.85, otherwise **access denied**.
2. The member's customer profile is sent to the product model; the prediction is withheld.
3. Voice identified as the **same** member with confidence ≥ 0.83, otherwise **access denied**.
4. The predicted product category is displayed.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python -m src.train_all      # retrain and re-evaluate all three models
jupyter notebook             # open the notebooks
```

- The notebooks run in **Google Colab** (data read from the team Drive) or **locally** (data read from `data/`). They detect which one automatically.
- `src/` decodes `.m4a` recordings with the FFmpeg bundled in `imageio-ffmpeg`; running notebook 03 locally needs [FFmpeg](https://ffmpeg.org/) on your `PATH`.
- `scikit-learn` is pinned to the version the saved models were built with.

## Team

| Member   | Contribution                                                    |
| -------- | --------------------------------------------------------------- |
| Janviere | Data cleaning and merge, EDA, product recommendation model      |
| Yohan    | Image collection and processing, facial recognition model       |
| Thomas   | Audio collection and processing, voiceprint verification model  |
| Josue    | Model creation and evaluation (Task 4), repository organisation |

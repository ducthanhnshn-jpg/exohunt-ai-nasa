# 🌌 ExoHunt AI - NASA Kepler Exoplanet Discovery Engine
**An Artificial Intelligence Application for Exploring Exoplanets with NASA Kepler Space Telescope Data**

📄 **Published paper:** Dang, D. T., Pham, H. B. C., Nguyen, M. D., & Trinh, T. D. (2026). *What Do Machine Learning Models Actually Learn from the Kepler Object of Interest Catalogue? Separating Label Leakage from Label-Set Choice.* International Journal of Academic Engineering Research, 10(9), 237–247. [Read the paper (PDF)](http://ijeais.org/wp-content/uploads/2026/9/IJAER260918.pdf)

**Authors:** Đặng Đức Thanh¹, Pham Hoang Bao Chau², Nguyen Minh Duc³, Trinh Thanh Do⁴\*
¹Hanoi - Amsterdam High School for the Gifted, Hanoi · ²Experimental High School of Ho Chi Minh City University of Education · ³Techcom Life Insurance Joint Stock Company · ⁴Rikai Technology, Danang · \*Corresponding author

---

## 📌 1. PROJECT OVERVIEW
**ExoHunt AI** benchmarks three classifiers — a random forest, XGBoost and a PyTorch one-dimensional convolutional network — on NASA's cumulative **Kepler Object of Interest (KOI)** table (**N = 9,564** records, obtained through its Kaggle distribution). All three models are trained on an identical set of **36 features** with every vetting-derived column removed. The light curves themselves are not used: each record holds parameters derived from earlier analysis of the light curve and the host star.

Each signal belongs to one of 3 dispositions (`koi_disposition`):
1. 🟢 **CONFIRMED:** confirmed planets.
2. 🔵 **CANDIDATE:** signals whose vetting is unfinished — a position in an administrative queue rather than a property of an object.
3. 🔴 **FALSE POSITIVE:** signals rejected as impostors, such as grazing eclipsing binaries, starspots or instrument artefacts.

---

## 🤖 2. COMPARING THE 3 MODEL ARCHITECTURES

Missing values are imputed with the column median, the data are split by stratified random sampling into an 80% training partition (n = 7,651) and an independent 20% test partition (n = 1,913), features are standardised with parameters estimated on the training partition alone, and SMOTE is applied only inside the training partition (4,018 records per class). The six vetting-derived columns (the four Robovetter false-positive flags, `koi_score` and `koi_pdisposition`) and five identifier columns are removed.

| Model | Core technique | Test accuracy | Macro F1 | ROC-AUC (CONFIRMED) | Role |
| :--- | :--- | :---: | :---: | :---: | :--- |
| 🌲 **Random Forest** | 100 trees under a fixed seed, random feature subset at each split | **.79** | **.75** | **.961** | Conventional baseline for tabular problems of this size |
| ⚡ **XGBoost** | Gradient boosting: multi:softprob, learning rate .08, max depth 6, 250 estimators (fixed, not tuned) | **.79** | **.76** | **.966** | Re-estimated across three feature sets and two label sets (Section 3 below) |
| 🧠 **PyTorch 1D-CNN** | Conv1d(1→16) and Conv1d(16→32) with batch normalisation, ReLU and max pooling; 64-unit dense layer with dropout .3 | **.75** | **.73** | **.959** | Architectural comparison under a fixed representation (convolution along the 36-feature vector, not over light curves) |

*Test partition n = 1,913, three labels, no vetting columns. Five-fold stratified cross-validation of XGBoost: macro F1 .739 (SD = .012). On the paired test predictions XGBoost beats the 1D-CNN with a McNemar statistic of 14.01 (p < .001), while the three AUCs differ by only .007: architecture makes a small difference to hard classification and almost none to ranking.*

Errors concentrate on CANDIDATE: XGBoost's recall is .55 on CANDIDATE against .85 on CONFIRMED and .88 on FALSE POSITIVE (1,517 of 1,913 test KOIs correct, 396 errors).

---

## 🔬 3. SCIENTIFIC FINDING: LABEL LEAKAGE VS. LABEL-SET CHOICE

Published accuracies on the KOI catalogue range from below .80 to above .98. Two things differ across these studies and are rarely stated:
* **The feature set.** Six columns encode the outcome of NASA's automated vetting (the Robovetter), which is also the source of the class label: `koi_fpflag_nt` (not transit-like), `koi_fpflag_ss` (stellar eclipse), `koi_fpflag_co` (centroid offset), `koi_fpflag_ec` (ephemeris match indicating contamination), the disposition score `koi_score` and the pipeline disposition `koi_pdisposition`. A model given them predicts an outcome from its own antecedents — label leakage in its textbook form.
* **The label set.** Whether CANDIDATE is a third class, or excluded. The highest catalogue-based figure, 98% (Sturrock et al.), is a two-class result (CONFIRMED vs FALSE POSITIVE) obtained with the four flags still in the feature set.

The XGBoost configuration was re-estimated across three feature sets and two label sets on the same split, seed and hyperparameters:

| Cell | Feature set | Labels | Accuracy | Macro F1 | Increment |
| :---: | :--- | :---: | :---: | :---: | :--- |
| A1 | 36 columns, no vetting | Three | .793 | .758 | reference |
| A2 | 36 + four false-positive flags | Three | .907 | .878 | +.114 over A1 |
| A3 | 36 + four flags + koi_score + koi_pdisposition | Three | .925 | .897 | +.018 over A2 |
| B1 | 36 columns, no vetting | Two | .939 | .930 | +.146 over A1 |
| B2 | 36 + four false-positive flags | Two | .994 | .993 | +.055 over B1 |
| B3 | 36 + four flags + koi_score + koi_pdisposition | Two | .997 | .997 | +.003 over B2 |

*Binary cells exclude CANDIDATE records from both partitions (test n = 1,464 against 1,913), so levels are not directly comparable across label sets and the increments carry the argument. Cell B2 corresponds to the design of Sturrock et al.*

👉 **Conclusions:**
* Cell B2 reaches **.994**, above the 98% that Sturrock et al. reported: the published figure is reproducible, and its distance from .79 is a matter of design alone.
* Averaged over both orders, **11.7** of the **20.1** accuracy points belong to the label set and **8.5** to the four flags — the label set is the larger term.
* `koi_score` and `koi_pdisposition` add only .018 (three-class) and .003 (binary) once the flags are present.
* Accuracy on the KOI catalogue is therefore not interpretable unless both the feature set and the label set are disclosed.

---

## 📌 4. WHAT THE RESULTS DO NOT SUPPORT (LIMITATIONS)

* **Host-star split:** 23.9% of records share a host with at least one other record. A split grouped by Kepler Input Catalog ID gives accuracy .779 and macro F1 .748 (against .793 and .758) — real leakage, worth about 1.4 points.
* **Measurement uncertainties:** dropping every uncertainty column (leaving 16 features) lowers accuracy from .793 to .764 and macro F1 from .758 to .733 — a residual channel worth about three points.
* **Cross-validation:** for the five-fold check, feature scaling was estimated across the complete dataset and no resampling was applied inside the folds.
* **Interpretability:** SHAP was applied during development, but the multi-class output was handled incorrectly and the resulting figures proved unusable; a corrected analysis reporting attributions separately by class is the natural next step.
* **Representation:** the 1D-CNN operated on a tabular vector rather than on photometry, so the comparison with light-curve architectures remains indirect.
* **Snapshot:** a single catalogue snapshot was used; repeating the analysis on a later release would separate genuine misclassification from anticipation of reclassification.

---

## 🚀 5. HOW TO INSTALL THE LIBRARIES & RUN THE PROJECT

The project can run in 2 ways: **with the Python backend server (FastAPI + PyTorch engine)** or **directly in the browser (offline client-side mode)**.

### 🛠️ Step 1: Install the Python libraries (prerequisites & requirements)
For the Python server to load the `.pkl`, `.json` and `.pth` models from the `./weight/` folder, install the libraries listed in **`requirements.txt`**:

1. Open a terminal (Command Prompt / PowerShell / VS Code Terminal) in the project folder `exohunt-ai-nasa`:
   ```bash
   pip install -r requirements.txt
   ```

2. **⚠️ SPECIAL NOTE FOR WINDOWS (fixing the PyTorch WinError 127):**
   If you get an error about a missing CUDA DLL on Windows (for example `[WinError 127] The specified procedure could not be found... c10_cuda.dll`), it is because the default PyTorch package looks for an incompatible NVIDIA CUDA graphics card.
   👉 **Fix:** uninstall the current PyTorch and install the **CPU-only PyTorch** build:
   ```bash
   pip uninstall torch torchvision torchaudio -y
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

---

### ⚡ Step 2: Start the server & use the app

#### Option A: Run the Python FastAPI backend (recommended)
With the Python server, the app runs inference directly with the real model weights in `./weight/`:
1. Start the server:
   ```bash
   python app.py
   ```
2. Open a web browser (Google Chrome, Microsoft Edge, Firefox, Safari) and go to the address shown in the terminal:
   👉 **`http://127.0.0.1:8000`**
3. The **ExoHunt AI Observatory** interface loads. Moving any slider on the *A/B Testing* tab sends a real-time request to the Python server. The server receives only the 6 slider values and sets the other 30 features to their training-set averages. The four real Kepler presets always show the precomputed model outputs for their full catalogue data.

#### Option B: Run directly in the browser (fastest, offline)
If you do not want to install Python, or you are on a computer without a programming environment, you can open the page on its own:
1. Open the `exohunt-ai-nasa` project folder.
2. Double-click **`index.html`** (or right-click and choose *Open with -> Google Chrome / Microsoft Edge*).
3. All features work: the A/B Testing demo, the simulated transit light curve and the 600-DPI figures. Without the server, the trained models run only for the four real Kepler presets (their outputs were precomputed from the saved weights); the example presets and custom slider values show a clearly labelled rule-based illustration instead.

---

## 📁 6. DIRECTORY STRUCTURE

```text
exohunt-ai-nasa/
│
├── 📄 index.html                        # HTML5 observatory interface (5 tabs)
├── 🎨 style.css                         # Interface styling (light theme, glassmorphism)
├── ⚡ script.js                         # Interactivity, A/B Testing demo & canvas light curve
├── 🐍 app.py                            # FastAPI server & PyTorch inference engine
├── 📦 requirements.txt                  # Library list (FastAPI, PyTorch, XGBoost, SHAP...)
├── 📖 README.md                         # Report & user guide (this file)
├── ⚙️ .gitignore                        # Ignores Python build files & caches
│
├── 📓 exohunt-ai-nasa-dangducthanh.ipynb # Research notebook (about 500 lines of code: EDA, ML, DL, SHAP)
│
├── 📂 data/
│   └── cumulative.csv                   # NASA cumulative KOI table (Kaggle "Kepler Exoplanet Search Results")
│
├── 📂 exoplanet_research_plots/         # Figures exported by the notebook (600 DPI)
│   ├── eda_exoplanet_analysis.png       # Label distribution & orbital period vs. transit depth
│   ├── roc_auc_comparison.png           # One-vs-rest ROC curves for CONFIRMED (AUC .959 - .966)
│   ├── confusion_matrix_errors.png      # XGBoost confusion matrix (1,913 test KOIs)
│   ├── shap_feature_importance.png      # SHAP figure from development: unusable (see Section 4), not shown on the site
│   └── shap_pure_physics.png            # SHAP figure from development: unusable (see Section 4), not shown on the site
│
└── 📂 weight/                           # Packaged model weights
    ├── random_forest_baseline.pkl       # Random Forest (24.2 MB)
    ├── xgboost_physics_model.json       # XGBoost (3.96 MB)
    ├── 1d_cnn_weights.pth               # PyTorch 1D-CNN state dict (86.4 KB)
    ├── scaler_main.pkl                  # Z-score StandardScaler fitted on the training partition (2.2 KB)
    └── label_encoder.pkl                # Label encoder: CANDIDATE=0, CONFIRMED=1, FALSE POSITIVE=2 (514 B)
```

---

## 📝 7. NOTES ON THE CODE (OCTOBER 2026)

* **Exported scaler:** `weight/scaler_main.pkl` now holds the scaler fitted on the training partition, as described in the paper. The earlier file had been refit on the full dataset (9,564 rows) during the cross-validation step; with it, the saved XGBoost model's test accuracy fell to .671.
* **Backend:** `app.py` used to map the model probabilities to the wrong labels (CANDIDATE and CONFIRMED were swapped). It now maps them by class name and reports whether a result came from a model or from the rule-based fallback.
* **Web demo:** the page shows the real outputs of the saved models for the four real Kepler targets and labels everything else as a rule-based illustration.

---
*Authors: Đặng Đức Thanh, Pham Hoang Bao Chau, Nguyen Minh Duc & Trinh Thanh Do (corresponding author) · Web app development: Đặng Đức Thanh - ExoHunt AI NASA Exoplanet Research Project.*

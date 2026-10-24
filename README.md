# 🌌 ExoHunt AI - NASA Kepler Exoplanet Discovery Engine
**An Artificial Intelligence Application for Exploring Exoplanets with NASA Kepler Space Telescope Data**

---

## 📌 1. PROJECT OVERVIEW
**ExoHunt AI** is an in-depth data-science and machine-learning research project. It applies modern AI architectures to classify the light-dip (transit) signals of **9,564 Kepler Objects of Interest (KOIs)** from NASA's Kepler Space Telescope, using the parameters listed for each signal in NASA's cumulative KOI catalogue (the light curves themselves are not used).

The core goal of the research is to classify each signal into one of 3 groups of astronomical objects:
1. 🟢 **CONFIRMED (confirmed planet):** real exoplanets orbiting their host star, with the characteristic flat-bottomed, U-shaped transit signal.
2. 🔵 **CANDIDATE (planet candidate):** promising signals that need more observations before they can be verified.
3. 🔴 **FALSE POSITIVE (impostor signal):** astronomical phenomena mistaken for exoplanets (mainly **eclipsing binary stars**, which produce deep V-shaped dips, or noise from the host star's activity).

---

## 🤖 2. COMPARING 3 AI MODEL ARCHITECTURES (PURE PHYSICS BENCHMARK)

In the latest version of the research, all 3 models use the same **Pure Physics** setup: **36 features without NASA's vetting flags** — the four false-positive flags produced by NASA's automated vetting (the Robovetter) are removed completely. All models were trained in a Kaggle notebook (the recorded run used the CPU); XGBoost was additionally checked with 5-fold stratified cross-validation:

| Model Architecture | Core Technique | Test Accuracy | ROC-AUC | Role in the Research |
| :--- | :--- | :---: | :---: | :--- |
| 🌲 **Random Forest** | Ensemble of decision trees (100 estimators), SMOTE balancing | **78.5%** | **0.961** | **Tree-ensemble model:** trained on the 36 transit, orbital & stellar features, with high stability. |
| ⚡ **XGBoost Pure Physics** | Gradient boosting, anti-leakage protection | **79.3%** | **0.966** | **Pure Physics model (core result):** 5-fold CV macro-F1 0.739 ± 0.010. Forces the model to learn from transit, orbital and stellar parameters instead of the vetting flags. |
| 🧠 **PyTorch 1D-CNN** | Deep learning: Conv1D(16→32) + BatchNorm + MaxPool1D + Dropout(0.3) | **74.9%** | **0.959** | **Convolutional deep network:** automatically extracts complex non-linear interactions between the period, depth and stellar-radius parameters. |

*Test accuracy is measured on 1,913 test KOIs with all 3 labels. ROC-AUC is for the CONFIRMED class (one-vs-rest).*

---

## 🔬 3. SCIENTIFIC FINDING: DATA LEAKAGE (DEEP DIVE)

The key parts of this analysis are **Sections 4, 6 and 6c** of the notebook:

### 🚨 The "98.24% Illusion" (Why did the baseline reach 98%?)
When the initial baseline model was trained on 43 features, its accuracy reached an extremely high 98.24%. But four of those columns are false-positive flags produced by NASA's own vetting (the automated Robovetter) — the same process that assigns the labels — so the model was partly reading the answer:
* `koi_fpflag_nt` (Not Transit-Like Flag)
* `koi_fpflag_ss` (Stellar Eclipse Flag - eclipsing binary)
* `koi_fpflag_co` (Centroid Offset Flag)
* `koi_fpflag_ec` (Ephemeris Match Flag)

👉 **Conclusion:** the old model suffered from serious **data leakage** — the AI did not "understand" astrophysics, it was "memorizing" the result of NASA's vetting! Such a model would be **useless** on data from new telescope missions (such as TESS or PLATO), where the incoming data have not yet been flagged by anyone.

### 💡 The Fix: the Pure Physics Benchmark
To build an AI that could genuinely find new exoplanets, we **removed the 4 leakage flags from the entire pipeline** and retrained all 3 models (Random Forest, XGBoost, 1D-CNN) on the **36 remaining features** (transit, orbital and stellar parameters).
* **Honest accuracy:** **75% - 79%** (AUC **0.959 - 0.966**) — an honest figure that reflects how hard the task really is and what the models can actually do.
* **Corrected SHAP analysis:** the original SHAP figures were drawn incorrectly. For a 3-class model, recent SHAP versions return an array of shape (samples, features, classes); `shap.summary_plot` read it as interaction values and drew only the first three columns of the table (`koi_period`, `koi_period_err1`, `koi_period_err2`), so those columns only *looked* most important. The corrected per-class SHAP summary (regenerated from the saved XGBoost model in October 2026) ranks the features as follows:
  1. `koi_model_snr` (transit signal-to-noise ratio) — by far the strongest input
  2. `koi_prad` (planet radius)
  3. `koi_steff_err1` & `koi_duration_err1` (measurement uncertainties — catalogue properties rather than physics)
  4. `koi_period` & `koi_duration` (orbital period and transit duration)

  Sky coordinates (`ra`, `dec`) also rank in the top 11, so not every signal the model uses is physical.

---

## 🚀 4. HOW TO INSTALL THE LIBRARIES & RUN THE PROJECT

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
3. All features work: the A/B Testing demo, the simulated transit light curve and the 600-DPI figures. Without the server, the trained models run only for the four real Kepler presets (their outputs were precomputed from the saved weights); the hypothetical presets and custom slider values show a clearly labelled rule-based illustration instead.

---

## 📁 5. DIRECTORY STRUCTURE

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
├── 📓 exohunt-ai-nasa-dangducthanh.ipynb # Original research notebook (about 500 lines of code: EDA, ML, DL, SHAP)
│
├── 📂 data/
│   └── cumulative.csv                   # NASA cumulative KOI table (Kaggle "Kepler Exoplanet Search Results")
│
├── 📂 exoplanet_research_plots/         # 5 high-resolution research figures (600 DPI)
│   ├── eda_exoplanet_analysis.png       # Exploratory data analysis & physical relationships
│   ├── roc_auc_comparison.png           # ROC-AUC curves comparing the 3 models (AUC 0.959 - 0.966)
│   ├── shap_feature_importance.png      # Corrected SHAP summary of the Section 6 XGBoost model (same 36 features)
│   ├── shap_pure_physics.png            # Corrected SHAP summary of the Pure Physics XGBoost model (Section 6c)
│   └── confusion_matrix_errors.png      # Confusion matrix & error analysis (1,913 test samples)
│
└── 📂 weight/                           # Packaged model weights
    ├── random_forest_baseline.pkl       # Random Forest Pure Physics (24.2 MB)
    ├── xgboost_physics_model.json       # XGBoost Pure Physics (3.96 MB)
    ├── 1d_cnn_weights.pth               # PyTorch 1D-CNN state dict (86.4 KB)
    ├── scaler_main.pkl                  # Z-score StandardScaler fitted on the training set (2.2 KB)
    └── label_encoder.pkl                # Label encoder: CANDIDATE=0, CONFIRMED=1, FALSE POSITIVE=2 (514 B)
```

---

## 📝 6. NOTES & CORRECTIONS (OCTOBER 2026)

* **SHAP figures:** both SHAP figures were regenerated with a corrected per-class plot (see Section 3). Because Cell 4 of the notebook already removes the four flags, the Section 6 and Section 6c XGBoost models are identical, so the two figures show the same model.
* **Exported scaler:** `weight/scaler_main.pkl` used to be refit on the full dataset (9,564 rows) during the cross-validation step, instead of the training-set scaler the models were trained with. With that file the XGBoost model's test accuracy fell to 67.1%. The file now holds the training-set scaler (7,651 rows), which reproduces the notebook's test results exactly.
* **Backend:** `app.py` mapped the model probabilities to the wrong labels (CANDIDATE and CONFIRMED were swapped). It now maps them by class name and reports whether a result came from a model or from the rule-based fallback.
* **Web demo:** the static page used to show hard-coded probabilities chosen from each preset's known label. It now shows the real outputs of the saved models for the four real Kepler objects and labels everything else as a rule-based illustration.
* **Cross-validation:** the earlier figure of F1 0.840 came from cross-validating on the SMOTE-oversampled training set, where synthetic samples made from the same KOIs land in both training and validation folds. The leakage-free 5-fold CV in Section 6b gives macro-F1 0.739 ± 0.010.

---
*Research & app development: Đặng Đức Thanh - ExoHunt AI NASA Exoplanet Research Project.*

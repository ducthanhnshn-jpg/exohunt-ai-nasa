# ==============================================================================
# ExoHunt AI - NASA Exoplanet Discovery Engine (Backend Server)
# FastAPI Server & Model Inference Pipeline Loading Saved Weights
# ==============================================================================

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from pydantic import BaseModel

# Try importing FastAPI and Uvicorn
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse, JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("⚠️ WARNING: FastAPI/uvicorn not found. Please run: pip install fastapi uvicorn")

# Try importing Machine Learning libraries
try:
    import joblib
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    ML_AVAILABLE = True
except (ImportError, OSError) as e:
    ML_AVAILABLE = False
    print(f"⚠️ WARNING: could not load the ML libraries (scikit-learn/XGBoost): {e}")

# Try importing PyTorch for 1D-CNN
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except (ImportError, OSError) as e:
    TORCH_AVAILABLE = False
    print(f"⚠️ WARNING: could not load PyTorch (or CUDA DLL error WinError 127): {e}")

# ==============================================================================
# 1. DEFINITION OF PYTORCH 1D-CNN ARCHITECTURE (From "Cell Code 7" of the Notebook)
# ==============================================================================
if TORCH_AVAILABLE:
    class Exoplanet1DCNN(nn.Module):
        def __init__(self, input_features, num_classes):
            super(Exoplanet1DCNN, self).__init__()
            # Block 1
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm1d(16)
            self.pool1 = nn.MaxPool1d(kernel_size=2)

            # Block 2
            self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm1d(32)
            self.pool2 = nn.MaxPool1d(kernel_size=2)

            # Flatten & Dense
            self.fc1 = nn.Linear(32 * (input_features // 4), 64)
            self.dropout = nn.Dropout(0.3)
            self.fc2 = nn.Linear(64, num_classes)

        def forward(self, x):
            # x shape: (batch_size, 1, input_features)
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

# ==============================================================================
# 2. INFERENCE ENGINE & WEIGHTS LOADER
# ==============================================================================
class ExoHuntInferenceEngine:
    def __init__(self, weights_dir: str = "./weight"):
        self.weights_dir = weights_dir
        self.rf_model = None
        self.xgb_model = None
        self.cnn_model = None
        self.scaler = None
        self.label_encoder = None
        # Column order of the models' probability outputs = LabelEncoder order (alphabetical).
        # Overwritten below with label_encoder.classes_ when the file is available.
        self.classes = ['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']
        self.num_features = 36

        self.load_all_weights()

    def load_all_weights(self):
        print("--- 🔄 LOADING MODEL WEIGHTS FROM ./weight/ ---")

        # Load Preprocessing Weights
        if ML_AVAILABLE:
            try:
                scaler_path = os.path.join(self.weights_dir, "scaler_main.pkl")
                le_path = os.path.join(self.weights_dir, "label_encoder.pkl")
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
                    if hasattr(self.scaler, 'n_features_in_'):
                        self.num_features = self.scaler.n_features_in_
                    print(f"  ✅ Loaded StandardScaler: scaler_main.pkl ({self.num_features} features)")
                if os.path.exists(le_path):
                    self.label_encoder = joblib.load(le_path)
                    self.classes = list(self.label_encoder.classes_)
                    print(f"  ✅ Loaded LabelEncoder: {self.classes}")
            except Exception as e:
                print(f"  ❌ Error loading the preprocessing files: {e}")

            # Load Random Forest
            try:
                rf_path = os.path.join(self.weights_dir, "random_forest_baseline.pkl")
                if os.path.exists(rf_path):
                    self.rf_model = joblib.load(rf_path)
                    print("  ✅ Loaded the Random Forest baseline model (24.2 MB - Pure Physics)")
            except Exception as e:
                print(f"  ❌ Error loading Random Forest: {e}")

            # Load XGBoost Physics Model
            try:
                xgb_path = os.path.join(self.weights_dir, "xgboost_physics_model.json")
                if os.path.exists(xgb_path):
                    self.xgb_model = xgb.Booster()
                    self.xgb_model.load_model(xgb_path)
                    print("  ✅ Loaded the XGBoost Pure Physics model (3.96 MB)")
            except Exception as e:
                print(f"  ❌ Error loading XGBoost: {e}")

        # Load PyTorch 1D-CNN
        if TORCH_AVAILABLE:
            try:
                cnn_path = os.path.join(self.weights_dir, "1d_cnn_weights.pth")
                if os.path.exists(cnn_path):
                    self.cnn_model = Exoplanet1DCNN(input_features=self.num_features, num_classes=len(self.classes))
                    state_dict = torch.load(cnn_path, map_location=torch.device('cpu'))
                    self.cnn_model.load_state_dict(state_dict)
                    self.cnn_model.eval()
                    print("  ✅ Loaded the PyTorch 1D-CNN weights (86.4 KB)")
            except Exception as e:
                print(f"  ❌ Error loading the PyTorch 1D-CNN: {e}")

    def predict(self, model_type: str, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict the class probabilities for one target.
        Uses the 36 Pure Physics features (Pure Physics Benchmark - no vetting flags).
        Only 6 values come from the user; the other 30 features stay at their training-set means.
        """
        # Start from the training-set mean of every feature (or zeros)
        if self.scaler and hasattr(self.scaler, 'mean_'):
            input_vec = np.array([self.scaler.mean_])
        else:
            input_vec = np.zeros((1, self.num_features))

        # Map interactive UI features to exact Kepler dataset column indices:
        # 0: koi_period, 9: koi_duration, 12: koi_depth, 24: koi_steff, 27: koi_slogg, 30: koi_srad
        input_vec[0, 0] = features.get('period', 5.0)     # koi_period (orbital period)
        if self.num_features > 9:
            input_vec[0, 9] = features.get('duration', 3.0) # koi_duration (transit duration)
        if self.num_features > 12:
            input_vec[0, 12] = features.get('depth', 500.0) # koi_depth (transit depth)
        if self.num_features > 24:
            input_vec[0, 24] = features.get('steff', 5800.0)# koi_steff (stellar temperature)
        if self.num_features > 27:
            input_vec[0, 27] = features.get('slogg', 4.4)   # koi_slogg (stellar log g)
        if self.num_features > 30:
            input_vec[0, 30] = features.get('srad', 1.0)    # koi_srad (stellar radius)

        # Try real model inference
        try:
            if self.scaler:
                if hasattr(self.scaler, 'feature_names_in_') and self.scaler.feature_names_in_ is not None:
                    import pandas as pd
                    input_df = pd.DataFrame(input_vec, columns=self.scaler.feature_names_in_)
                    input_scaled = self.scaler.transform(input_df)
                else:
                    input_scaled = self.scaler.transform(input_vec)
            else:
                input_scaled = input_vec

            if model_type == "Random Forest" and self.rf_model:
                probs = self.rf_model.predict_proba(input_scaled)[0]
                return self._format_output(probs)

            elif model_type == "XGBoost" and self.xgb_model:
                dmatrix = xgb.DMatrix(input_scaled, feature_names=getattr(self.xgb_model, 'feature_names', None))
                probs = self.xgb_model.predict(dmatrix)[0]
                return self._format_output(probs)

            elif model_type == "1D-CNN" and self.cnn_model and TORCH_AVAILABLE:
                tensor_in = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(1)
                with torch.no_grad():
                    out = self.cnn_model(tensor_in)
                    probs = F.softmax(out, dim=1).numpy()[0]
                return self._format_output(probs)
        except Exception as e:
            print(f"⚠️ Error while running the {model_type} model ({e}). Falling back to the rule-based illustration.")

        # Fallback: rule-based illustration (NOT a trained model)
        return self._simulate_physics(model_type, features)

    def _format_output(self, probs: np.ndarray, source: str = "model") -> Dict[str, Any]:
        # The models output one probability per class in LabelEncoder order
        # (CANDIDATE, CONFIRMED, FALSE POSITIVE), so map them by name, not by position.
        prob_dict = {str(name): float(p) for name, p in zip(self.classes, probs)}
        pred_class = max(prob_dict, key=prob_dict.get)
        conf_score = round(prob_dict[pred_class] * 100, 1)

        return {
            "probs": prob_dict,
            "prediction": pred_class,
            "confidence": str(conf_score),
            "source": source  # "model" = real model output, "rule-based fallback" = no model was run
        }

    def _simulate_physics(self, model_type: str, f: Dict[str, float]) -> Dict[str, Any]:
        """Rule-based illustration used only when a model cannot be run."""
        depth = f.get('depth', 500.0)
        srad = f.get('srad', 1.0)
        period = f.get('period', 5.0)

        prob_conf, prob_cand, prob_fp = 0.33, 0.33, 0.34

        # Simple astrophysical rules of thumb
        if depth > 8000 and srad < 1.0:
            prob_fp, prob_cand, prob_conf = 0.85, 0.10, 0.05
        elif 100 <= depth <= 5000 and period > 1.0 and 0.5 <= srad <= 2.0:
            prob_conf, prob_cand, prob_fp = 0.68, 0.22, 0.10
        elif depth < 100 or period > 100:
            prob_cand, prob_conf, prob_fp = 0.60, 0.20, 0.20
        else:
            prob_fp, prob_cand, prob_conf = 0.45, 0.35, 0.20

        total = prob_conf + prob_cand + prob_fp
        by_name = {"CANDIDATE": prob_cand / total, "CONFIRMED": prob_conf / total, "FALSE POSITIVE": prob_fp / total}
        return self._format_output(np.array([by_name[str(c)] for c in self.classes]), source="rule-based fallback")

# Initialize global engine
engine = ExoHuntInferenceEngine()

# ==============================================================================
# 3. FASTAPI REST API & STATIC FILE SERVER
# ==============================================================================
if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="ExoHunt AI - NASA Exoplanet Discovery API",
        description="REST API for classifying exoplanet signals from Kepler data",
        version="1.0.0"
    )

    class TelemetryRequest(BaseModel):
        model: str = "XGBoost"
        srad: float = 1.01
        steff: float = 5845.0
        period: float = 2.13
        depth: float = 269.4
        slogg: float = 4.47
        duration: float = 2.95

    @app.post("/api/predict")
    async def predict_exoplanet(req: TelemetryRequest):
        features = {
            "srad": req.srad,
            "steff": req.steff,
            "period": req.period,
            "depth": req.depth,
            "slogg": req.slogg,
            "duration": req.duration
        }
        result = engine.predict(req.model, features)
        return JSONResponse(content=result)

    # Serve static frontend files
    @app.get("/")
    async def read_index():
        return FileResponse("index.html")

    # Mount static directories
    if os.path.exists("exoplanet_research_plots"):
        app.mount("/exoplanet_research_plots", StaticFiles(directory="exoplanet_research_plots"), name="plots")

    app.mount("/", StaticFiles(directory="."), name="static")

    if __name__ == "__main__":
        print("\n🚀 Starting the ExoHunt AI server at: http://127.0.0.1:8000")
        uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
else:
    if __name__ == "__main__":
        print("\n--- 🪐 TEST RUN OF THE INFERENCE ENGINE IN THE TERMINAL (CLI MODE) ---")
        demo_features = {"srad": 1.01, "steff": 5845.0, "period": 2.13, "depth": 269.4, "slogg": 4.47, "duration": 2.95}
        for model in ["Random Forest", "XGBoost", "1D-CNN"]:
            res = engine.predict(model, demo_features)
            print(f"Model [{model}] -> Prediction: {res['prediction']} (confidence: {res['confidence']}%, source: {res['source']}) | Probabilities: {res['probs']}")

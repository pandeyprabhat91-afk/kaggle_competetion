# Kaggle Competition: Embedding-Based Ensemble Regressor  
**Author:** Major Prabhat Pandey (DA25M002)  
**Program:** M.Tech Artificial Intelligence & Data Science  

This repository contains the full training and inference pipeline for a Kaggle regression challenge using embedding-based feature engineering, Siamese-style vector interactions, and a 5-model ensemble (CatBoost, XGBoost, LightGBM, Random Forest, Deep CatBoost).  
The system processes 768-dimensional embeddings from `google/embeddinggemma-300m`, constructs high-dimensional similarity features, trains multiple regressors, averages predictions, applies calibration, and generates the final submission file.

---

## ** Pipeline Overview**

### **Step 1 — Import Libraries**
Core libraries for:
- Data loading (NumPy, Pandas, JSON)
- File handling (`Path`)
- ML Models (CatBoost, XGBoost, LightGBM, RandomForest)
- Evaluation metrics (RMSE)

The environment is optimized for GPU acceleration (CatBoost, XGBoost, LightGBM).

---

## **Step 2 — Load Data**
Loads:
- `train_data_fixed_embeddings.json`
- `test_data_fixed_embeddings.json`
- `metric_name_embeddings.npy`

These contain:
- Prompt embeddings  
- Response embeddings  
- Metric name embeddings (145 × 768)  

File paths are resolved using a portable workspace root.

Logs displayed:
- Training samples: **5,000**  
- Test samples: **3,638**  
- Embedding shape: **(145, 768)**  

---

## **Step 3 — Feature Engineering (Siamese Architecture)**

### **3.1 Feature Construction**
A custom function builds **6,912-dimensional** Siamese features using:
- Base embeddings  
- Difference vectors  
- Absolute difference vectors  
- Element-wise products  

These capture:
- Individual meaning of embeddings  
- Interaction patterns  
- Similarity relations inspired by Siamese/contrastive networks  

### **3.2 Dataset Transformation**
`prepare_features()` converts raw JSON samples into:
- `X_train`, `y_train`
- `X_test`

Final shapes:
- Training: **(5000, 6912)**
- Test: **(3638, 6912)**  
Target score distribution:
- Mean: **9.12**
- Std: **0.94**

---

## **Step 4 — Train 5-Model Ensemble**

Five independent regressors are trained:

### **Model List**
1. **CatBoostRegressor (GPU)**  
2. **XGBRegressor (GPU)**  
3. **LGBMRegressor (GPU)**  
4. **RandomForestRegressor (CPU)**  
5. **Deep CatBoostRegressor (CPU, depth=8)**  

Each model is trained on the full 6,912-dimensional Siamese features.

Training RMSE results:
- CatBoost (shallow): **0.41**
- XGBoost: **0.109**
- LightGBM: **0.110**
- RandomForest: **0.713**
- Deep CatBoost: **0.150**

---

## **Step 5 — Ensemble Predictions**

A simple average across all models works best.

Metrics:
- **Ensemble Train RMSE:** **0.2566**
- Ensemble test distribution before calibration:
  - Mean: **9.104**
  - Std: **0.296**

---




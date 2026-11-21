# 🏠 Cairo Apartment Price Prediction

End-to-end machine learning pipeline that predicts apartment prices (EGP) in Greater Cairo from listing features.

![Results](images/results.png)

## Results (test set)
| Metric | Value |
|---|---|
| R² | **0.932** |
| MAPE | 10.2% |
| MAE | ~498K EGP |

Best model: **Gradient Boosting** (tuned with RandomizedSearchCV), compared against Ridge Regression and Random Forest using 5-fold cross-validation.

## Pipeline
1. **Data** – listings with area, size (m²), bedrooms, bathrooms, floor, finishing, compound, age, elevator, view
2. **Cleaning** – missing-value imputation, IQR outlier removal on price per m²
3. **EDA** – price per m² by area, price vs size

   ![EDA](images/eda.png)
4. **Feature engineering** – total rooms, m² per room, log-transformed target
5. **Modeling** – Scikit-learn Pipelines + ColumnTransformer (scaling, one-hot encoding)
6. **Evaluation** – MAE, RMSE, R², MAPE, permutation feature importance
7. **Deployment-ready** – saved model (`joblib`) + example prediction for a new listing

## Run it
```bash
pip install -r requirements.txt
jupyter notebook Cairo_Apartment_Price_Prediction.ipynb
```

## Tech stack
Python · Pandas · NumPy · Scikit-learn · Matplotlib · Jupyter

> **Note:** this demo uses a synthetic sample built to mimic Cairo listings. The same pipeline runs on real scraped data by replacing `data/cairo_apartments.csv`.

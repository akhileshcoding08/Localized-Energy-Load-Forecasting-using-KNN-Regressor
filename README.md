# Localized Energy Load Forecasting using KNN Regressor

A machine learning project that predicts hourly, region-level electricity (energy) load in megawatts (MW) using weather, socio-economic, and temporal features. The model is built with a **K-Nearest Neighbors (KNN) Regressor** and evaluated across multiple values of *K* to identify the best-performing configuration.

---

## 📌 Project Overview

Electricity demand ("load") varies constantly based on weather conditions, time of day, day of the week, holidays, and the level of industrial/commercial activity in a region. Accurately forecasting this localized load helps utilities and grid operators with:

- Efficient power generation and distribution planning
- Peak-load management and demand response
- Renewable energy integration (solar/wind availability)
- Cost optimization and reduced wastage

This project performs end-to-end analysis from raw data exploration to model building and evaluation and to predict `energy_load_mw` for different regions.

---

## 📂 Repository Structure

```
localized-energy-load-forecasting/
│
├── localized_energy_load.csv                     # Raw dataset
├── Localized_energy_load_KNN_Regressor.ipynb      # Main Jupyter Notebook (EDA + Modeling)
├── README.md                                      # Project documentation (this file)
└── Localized_Energy_Load_Project_Documentation.docx  # Detailed Word documentation
```

---

## 📊 Dataset Description

The dataset (`localized_energy_load.csv`) contains **~510,000 hourly records** across multiple regions, with the following columns:

| Column | Description |
|---|---|
| `timestamp` | Date and hour of the observation |
| `region` | Region name (North, South, East, West, Central) |
| `temperature_c` | Ambient temperature (°C) |
| `humidity_pct` | Relative humidity (%) |
| `wind_speed_mps` | Wind speed (m/s) |
| `solar_radiation_wm2` | Solar radiation (W/m²) |
| `population_density` | Population density of the region |
| `industrial_index` | Index representing industrial activity |
| `commercial_index` | Index representing commercial activity |
| `hour` | Hour of the day (0–23) |
| `day_of_week` | Day of week (0 = Monday … 6 = Sunday) |
| `month` | Month of the year (1–12) |
| `weekend_flag` | 1 if weekend, else 0 |
| `holiday_flag` | 1 if holiday, else 0 |
| `energy_load_mw` | **Target variable** — energy load in megawatts |

---

## 🔧 Tech Stack / Libraries Used

- **Python 3**
- **pandas, numpy** – data manipulation
- **matplotlib, seaborn** – data visualization
- **scikit-learn** – preprocessing, modeling, and evaluation
  - `KNeighborsRegressor`
  - `StandardScaler`, `OneHotEncoder`
  - `train_test_split`
  - Evaluation metrics: `r2_score`, `MAE`, `MSE`, `RMSE`, `MAPE`

---

## 🚀 Project Workflow

1. **Data Loading** – Import the CSV dataset using pandas.
2. **Data Understanding** – Inspect shape, data types, and summary statistics (`head`, `tail`, `info`, `describe`).
3. **Data Cleaning**
   - Remove duplicate records
   - Handle missing values by imputing numeric columns with the median
4. **Exploratory Data Analysis (EDA)**
   - Outlier detection using boxplots
   - Target variable distribution
   - Regional comparison of energy load
   - Hourly/weekday vs weekend demand patterns
   - Monthly seasonal trends by region
   - Correlation heatmap of numeric features
   - Relationship of industrial/commercial index and weather variables with load
5. **Feature Engineering**
   - One-Hot Encoding of the categorical `region` column
   - Feature/target split (`X`, `y`)
6. **Data Sampling** – A random sample of 100,000 rows used for efficient KNN computation.
7. **Feature Scaling** – Standardization using `StandardScaler` (fit on train, applied to test).
8. **Model Building** – KNN Regressor trained and evaluated for odd values of *K* from 1 to 19.
9. **Model Evaluation** – R², MAE, MSE, RMSE, and MAPE computed for each K.
10. **Result Visualization**
    - Actual vs Predicted scatter plot
    - Residual error distribution
    - Timeline comparison of actual vs predicted load (200-hour snapshot)

---

## 📈 Model Performance

The KNN Regressor was tested with K = 1, 3, 5, 7, 9, 11, 13, 15, 17, 19. Performance improved sharply up to K ≈ 9–15 and then plateaued.

| K | R² Score (%) | MAE | RMSE | MAPE (%) |
|---|---|---|---|---|
| 1 | 52.60 | 60.30 | 76.14 | 9.28 |
| 5 | 73.70 | 45.41 | 56.71 | 7.01 |
| 9 | 75.53 | 43.86 | 54.71 | 6.79 |
| **15 (best)** | **75.89** | **43.55** | **54.30** | **6.75** |
| 19 | 75.84 | 43.69 | 54.36 | 6.78 |

**Best Model: K = 15**, achieving an R² score of **~75.9%**, indicating the model explains close to 76% of the variance in localized energy load using the given weather, temporal, and socio-economic features.

---

## ▶️ How to Run This Project

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/localized-energy-load-forecasting.git
   cd localized-energy-load-forecasting
   ```

2. **Create and activate a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate        # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

4. **Launch the notebook**
   ```bash
   jupyter notebook Localized_energy_load_KNN_Regressor.ipynb
   ```

5. Run all cells sequentially to reproduce the EDA, model training, and evaluation results.

---

## 🔮 Future Improvements

- Hyperparameter tuning with `GridSearchCV` / `RandomizedSearchCV`
- Try other algorithms (Random Forest, XGBoost, LSTM for time-series) for comparison
- Use the full dataset instead of a 100,000-row sample (with a more optimized KNN implementation, e.g., `KDTree`/`BallTree`)
- Add time-based feature engineering (lag features, rolling averages)
- Deploy the trained model as a REST API or interactive dashboard

---

## 🙋 Author

Feel free to raise issues or submit pull requests for improvements.

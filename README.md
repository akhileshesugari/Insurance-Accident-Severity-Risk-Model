# Insurance Accident Severity Risk Model

> Machine learning–based insurance risk model using Random Forest to predict accident severity from driver demographics, vehicle attributes, and residential data. Designed to enhance underwriting accuracy, risk profiling, and data-driven premium pricing decisions.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-green)](https://flask.palletsprojects.com)
[![Docker](https://img.shields.io/badge/Docker-Containerised-blue)](https://docker.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange)](https://scikit-learn.org)

---

## Overview

This project develops a machine learning–driven risk assessment model designed to predict accident severity for insurance applications. Using structured accident data that includes driver demographics, vehicle characteristics, and residential area information, the model identifies patterns that influence the likelihood of severe accident outcomes.

The primary objective is to enhance insurance underwriting accuracy and support data-driven premium pricing strategies. Traditional risk evaluation often relies on historical averages and rule-based frameworks. In contrast, this project applies advanced predictive modelling techniques to uncover complex, non-linear relationships between demographic, environmental, and vehicle-related factors.

Beyond technical implementation, this project demonstrates clear business impact — enabling insurers to improve risk profiling, identify high-risk segments, optimise pricing strategies, and strengthen decision-making processes.

---

## Project Highlights

| Item | Detail |
|---|---|
| **Dataset** | UK Road Safety Data 2022 (STATS19) — 106,000+ collision records across 3 linked datasets |
| **Model** | Random Forest Classifier |
| **Target Variable** | Binary: `Slight` vs `Serious or Fatal` |
| **Deployment** | Flask REST API containerised with Docker |
| **Key Techniques** | SMOTE, GridSearchCV, Bayesian Optimisation, RFECV, Isolation Forest |
| **Best Accuracy** | 98% (full preprocessing pipeline) |

---

## Project Structure

```
accident_api/
├── app.py                  # Flask REST API (single + batch prediction endpoints)
├── model.pkl               # Trained Random Forest model
├── imputer.pkl             # Fitted SimpleImputer
├── scaler.pkl              # Fitted StandardScaler
├── features.pkl            # Feature order list
├── Dockerfile              # Container definition
├── requirements.txt        # Python dependencies
├── data/
│   └── README.md           # Dataset download instructions (CSVs not committed)
├── notebooks/
│   ├── stage_1.ipynb       # Data preparation & preprocessing pipeline
│   └── stage_2.ipynb       # Model training, tuning & evaluation
└── README.md
```

---

## ML Pipeline

### Stage 1 — Data Preparation (`notebooks/stage_1.ipynb`)

The workflow begins with comprehensive data preprocessing across three linked datasets (Collisions, Vehicles, Casualties):

- **Data Merging** — joined 3 STATS19 datasets on `accident_reference` to create a unified record (~10,000 records used for modelling)
- **Missing Value Handling** — replaced encoded unknowns (-1, 9, 99) with NaN; applied SimpleImputer, KNN Imputer, and Iterative Imputer (comparative analysis)
- **Feature Engineering** — time-of-day categorisation (Morning/Afternoon/Evening/Night), vehicle manoeuvre grouping, severity binary encoding
- **Encoding** — OneHotEncoder for categorical variables
- **Outlier Detection** — Isolation Forest (~380 outliers removed from training set)
- **Feature Selection** — SelectKBest (top 29 features retained)
- **Log Transformation** — applied to `age_of_vehicle` to correct left-skewed distribution
- **Train/Test Split** — stratified 80/20 split to preserve class distribution

### Stage 2 — Model Training & Evaluation (`notebooks/stage_2.ipynb`)

Exploratory Data Analysis (EDA) was conducted to understand key trends and correlations across variables such as driver age, vehicle type, location category, and accident severity. Four classifiers were trained and compared:

| Model | Accuracy |
|---|---|
| Logistic Regression | 95% |
| KNN Classifier | 97% |
| **Random Forest** | **98%** |
| XGBoost | 97.3% |

Additional techniques applied:
- **SMOTE** — synthetic oversampling to address class imbalance
- **GridSearchCV + Bayesian Optimisation** — hyperparameter tuning
- **RFECV** — recursive feature elimination with cross-validation
- **Evaluation** — accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, learning curves

Random Forest was selected as the final model due to its robustness, ability to capture non-linear relationships, and strongest performance on this structured dataset.

---

## REST API

The trained model is served via a Flask REST API, containerised with Docker for reproducible deployment.

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check — returns model info and expected features |
| `POST` | `/predict` | Predict severity for a single accident record |
| `POST` | `/predict/batch` | Predict severity for multiple records at once |

### Single Prediction — Request

```json
POST /predict
{
    "number_of_vehicles": 2,
    "number_of_casualties": 1,
    "day_of_week": 3,
    "speed_limit": 30,
    "road_surface_conditions": 1,
    "vehicle_manoeuvre": 9,
    "sex_of_driver": 1,
    "age_of_driver": 35,
    "age_of_vehicle": 5,
    "driver_home_area_type": 1,
    "sex_of_casualty": 1,
    "age_of_casualty": 30,
    "pedestrian_location": 0,
    "car_passenger": 0,
    "casualty_home_area_type": 1
}
```

### Single Prediction — Response

```json
{
    "prediction": "Slight",
    "confidence": 0.87,
    "probabilities": {
        "Slight": 0.87,
        "Serious or Fatal": 0.13
    }
}
```

---

## Running the Project

### Option 1 — Python directly

```bash
pip install -r requirements.txt
python app.py
```

API available at `http://localhost:5000`

### Option 2 — Docker (recommended)

```bash
# Build the image
docker build -t accident-severity-api .

# Run the container
docker run -p 5000:5000 accident-severity-api
```

### Test with curl

```bash
# Health check
curl http://localhost:5000/

# Single prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "number_of_vehicles": 2,
    "number_of_casualties": 1,
    "day_of_week": 3,
    "speed_limit": 30,
    "road_surface_conditions": 1,
    "vehicle_manoeuvre": 9,
    "sex_of_driver": 1,
    "age_of_driver": 35,
    "age_of_vehicle": 5,
    "driver_home_area_type": 1,
    "sex_of_casualty": 1,
    "age_of_casualty": 30,
    "pedestrian_location": 0,
    "car_passenger": 0,
    "casualty_home_area_type": 1
  }'
```

---

## Feature Reference

| Feature | Description | Values |
|---|---|---|
| `number_of_vehicles` | Number of vehicles involved | Integer |
| `number_of_casualties` | Number of casualties | Integer |
| `day_of_week` | Day (1=Sunday … 7=Saturday) | 1–7 |
| `speed_limit` | Road speed limit (mph) | 20, 30, 40, 50, 60, 70 |
| `road_surface_conditions` | Surface condition at time of accident | 1=Dry, 2=Wet, 3=Snow/Ice |
| `vehicle_manoeuvre` | Driver manoeuvre at time of accident | Encoded integer |
| `sex_of_driver` | Driver sex | 1=Male, 2=Female |
| `age_of_driver` | Driver age in years | Integer |
| `age_of_vehicle` | Vehicle age in years | Integer |
| `driver_home_area_type` | Driver residential area type | 1=Urban, 2=Small Town, 3=Rural |
| `sex_of_casualty` | Casualty sex | 1=Male, 2=Female |
| `age_of_casualty` | Casualty age in years | Integer |
| `pedestrian_location` | Pedestrian position relative to road | 0=Not a pedestrian |
| `car_passenger` | Passenger seating position | 0=Not a car passenger |
| `casualty_home_area_type` | Casualty residential area type | 1=Urban, 2=Small Town, 3=Rural |

---

## Business Impact

This predictive framework enables insurers to:

- **Improve risk profiling** — identify individual risk factors beyond simple demographic averages
- **Optimise premium pricing** — support data-driven, fair pricing strategies based on predicted severity
- **Identify high-risk segments** — flag combinations of features associated with serious outcomes
- **Strengthen underwriting decisions** — replace rule-based frameworks with evidence-based ML predictions

By transforming raw accident data into actionable risk intelligence, this solution highlights the practical integration of machine learning within the insurance domain.

---

## Author

**Akhilesh Esugari**
MSc Artificial Intelligence with Business Strategy — Aston University, Birmingham (2025)
[LinkedIn](https://linkedin.com/in/akhilesh-esugari) | [GitHub](https://github.com/akhileshesugari)

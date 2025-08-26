# Financial Forecasting Project – Toyota Motors Stock (1980-2024)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Streamlit-Interactive%20Dashboard-green" alt="Streamlit">
  <img src="https://img.shields.io/badge/Status-Portfolio%20Project-yellow" alt="Project Status">
</p>

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Tech Stack](#tech-stack)
4. [Installation](#installation)
5. [Usage](#usage)
6. [System Architecture](#system-architecture)
7. [Screenshots](#screenshots)
8. [Forecasting Models](#forecasting-models)
9. [Performance Metrics](#performance-metrics)
10. [Future Enhancements](#future-enhancements)
11. [Portfolio & Demo](#portfolio--demo)
12. [License](#license)

---

## Project Overview

This project aims to provide a **financial forecasting dashboard** for Toyota Motors stock data spanning 1980 to 2024. The goal is to help analysts and investors **predict future revenue, expenses, and profit trends** using robust time series and regression models.

The project demonstrates **data preprocessing, time series modeling, interactive visualization, and scenario analysis**, making it a strong portfolio example of a real-world Data Science application.

---

## Features

* **Data Preprocessing:** Clean, handle missing values, normalize, and aggregate financial data.
* **Forecasting Models:**

  * ARIMA & Prophet (time series forecasting)
  * Linear Regression / Ridge / Lasso (scenario forecasting)
* **Interactive Dashboard (Streamlit):**

  * Revenue, expenses, and profit trends
  * Forecast plots
  * KPI indicators: Profit Margin %, ROI
  * Drill-down by business units or time periods
  * Scenario analysis: change assumptions and view results dynamically
  * Export forecasts to Excel/PDF

---

## Tech Stack

* **Programming Language:** Python 3.10+
* **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn, statsmodels, prophet, streamlit
* **Deployment:** Streamlit Cloud / Heroku
* **Version Control:** Git & GitHub

---

## Installation

```
# Clone the repository
git clone https://github.com/HoussemELM/financial-forecasting-dashboard.git
cd financial-forecasting-dashboard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

1. Place your Toyota Motors stock CSV file in the `data/raw/` folder.
2. Preprocess the data:

```
python data/data_loader.py
```

3. Run the forecasting models:

```
python models/run_forecast.py
```

4. Launch the interactive dashboard:

```
streamlit run dashboard/app.py
```

5. Explore trends, forecasts, KPIs, and export reports.

---

## System Architecture

```
financial-forecasting-dashboard/
│
├── data/
│   ├── raw/           # Original CSV/Excel files
│   └── cleaned/       # Preprocessed data
│
├── models/
│   ├── arima_model.py
│   ├── prophet_model.py
│   └── regression_models.py
│
├── dashboard/
│   └── app.py         # Streamlit dashboard
│
├── utils/
│   └── helper_functions.py
│
├── requirements.txt
└── README.md
```

---

## Screenshots

*Add screenshots of your dashboard here after development.*

* Revenue Forecast Plot
* KPI Overview
* Scenario Analysis

---

## Forecasting Models

* **Time Series:** ARIMA, Prophet
* **Regression:** Linear, Ridge, Lasso
* **Scenario Forecasting:** Users can adjust growth %, expense assumptions

---

## Performance Metrics

* MAE (Mean Absolute Error)
* RMSE (Mean Squared Error)
* MAPE (Mean Absolute Percentage Error)

---

## Future Enhancements

* Real-time data integration (Yahoo Finance, Alpha Vantage API)
* Anomaly detection for sudden market changes
* Deep Learning models: LSTM, XGBoost for improved forecasting

---

## Portfolio & Demo

* GitHub Repository: [https://github.com/HoussemELM/financial-forecasting-dashboard]([https://github.com/HoussemELM/financial-forecasting-project](https://github.com/HoussemELM/Financials-Forcasting-Project))
* Live Dashboard: [Streamlit App Link](#)
* Demo Video: [YouTube/Loom Link](#)

**Use this project to showcase:**

1. End-to-end data pipeline
2. Financial forecasting techniques
3. Interactive dashboards and KPI reporting

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

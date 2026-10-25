# Smart City Public Transport Optimizer

A data-driven dashboard for analyzing public transport performance, identifying congestion patterns, and predicting transport delays using machine learning.

## Project Overview

The Smart City Public Transport Optimizer, also known as **TransitIQ**, is a web-based analytics application built with Python and Streamlit. It helps users explore transport data, analyze route performance, identify congestion hotspots, and estimate delays using a machine learning model.

## Features

- **Overview Dashboard:** Displays key transport statistics and performance summaries.
- **Route Intelligence:** Explore route-level transport indicators.
- **Congestion Analysis:** Identify congestion hotspots and traffic patterns.
- **Delay Prediction:** Estimate transport delays using machine learning.
- **Model Insights:** Explore model performance and prediction results.
- **Data Explorer:** Inspect and explore transport datasets.
- **Interactive Visualizations:** Plotly charts and Folium maps.
- **Data Processing:** Clean and prepare transport data using Pandas and NumPy.

## Technology Stack

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Folium
- Streamlit-Folium
- Scikit-learn
- Joblib

## Project Structure

```text
Smart_City_Transport_Optimizer/
├── app.py
├── data_processing.py
├── generate_data.py
├── train_model.py
├── requirements.txt
├── data/
│   ├── cleaned_transport.csv
│   ├── kaggle_transport_cleaned.csv
│   └── raw_transport.csv
├── models/
│   ├── delay_prediction_model.joblib
│   └── test_predictions.csv
└── outputs/
    ├── hourly_transport_summary.csv
    ├── kaggle_hourly_summary.csv
    └── top_3_congestion.csv
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Smart_City_Transport_Optimizer
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the dashboard

```bash
streamlit run app.py
```

The application should open in your browser. If it does not, open the local URL displayed in the terminal.

## Machine Learning

The application includes a machine learning model for transport delay prediction. Model training and data preparation are handled by separate Python scripts.

If the required processed datasets and trained model are not included in the repository, run the relevant preparation and training scripts before launching the dashboard.

## Data Disclaimer

This project is intended for educational and demonstration purposes. The dashboard should not be interpreted as a live traffic monitoring service unless connected to a verified real-time data source.

## Future Improvements

- Real-time transport data integration
- Route optimization recommendations
- Advanced congestion forecasting
- Additional machine learning models
- Deployment as a public web application

## Author

Developed as a Smart City transport analytics project using Python, data analysis, interactive visualization, and machine learning.

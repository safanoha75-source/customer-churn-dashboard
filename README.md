# ⚡ Customer Churn Prediction & Real-Time Intelligence System

An end-to-end Data Science and Machine Learning project designed to analyze customer churn behavior and deliver real-time predictive insights. This project includes an executive Plotly Dash web application built with a modern dark theme, animated KPI cards, dynamic visualizations, and a real-time machine learning inference engine.

---

## 📌 Project Overview

Customer churn is a critical metric for subscription and enterprise businesses. This system leverages machine learning to help business stakeholders identify key risk factors driving customer departure and evaluate real-time churn risk for new or existing clients.

### Key Features
- **Executive Analytics Dashboard:** Dynamic KPI cards displaying total customers, churn counts, churn rate, and average purchase amounts.
- **Interactive Visualizations:** High-impact Plotly charts analyzing churn distribution by Account Manager, geographical breakdown, and customer age vs. spend scatter plots.
- **Real-Time Predictive Engine:** On-demand inference powered by a trained Logistic Regression pipeline. Allows users to enter customer parameters and obtain instant churn predictions (`Class 0: Stay` vs. `Class 1: Churn`).
- **Resilient Fallback Architecture:** Automatically handles missing or incompatible pre-trained model files by dynamically retraining on the dataset at runtime.
- **Production-Ready Deployment:** Formatted with `Gunicorn` WSGI setup for seamless cloud deployment (Render / Plotly Cloud).

---

## 🛠️ Tech Stack & Tools

- **Language:** Python 3.10+
- **Machine Learning & Analytics:** Scikit-Learn, Pandas, NumPy
- **Interactive Web UI:** Plotly, Dash, Dash HTML & Core Components
- **Server & Deployment:** Gunicorn, Git, GitHub
- **IDE & Tools:** VS Code, Jupyter Notebooks

---

## 📁 Repository Structure

```text
├── dash_app.py             # Main Dash web application code with UI layout & callbacks
├── customer_churn.csv      # Primary customer dataset
├── churn_model.pkl         # Serialized trained machine learning pipeline
├── requirements.txt        # Dependencies required for local execution & cloud build
├── Procfile                # Web service deployment configuration for WSGI servers
└── README.md               # Detailed project documentation

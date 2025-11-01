# OCD-Classification using Machine Learning
A Machine Learning-based OCD Prediction Model that analyzes psychological and demographic features to predict the severity of Obsessive-Compulsive Disorder (OCD). The project includes data preprocessing, model training, calibration, and subgroup fairness evaluation to ensure accurate and interpretable results for clinical decision support.
---

##  **Project Overview**
The pipeline performs:
- Data preprocessing & encoding  
- Feature scaling  
- Model training with cross-validation  
- Calibration using **Isotonic Regression** for reliable probability outputs  
- Subgroup fairness evaluation by gender and ethnicity  
- Visualization of calibration curves for interpretability  

---
## **Features:**
- Predicts OCD likelihood using patient’s clinical data
- Estimates severity score (0–10 scale)
- Provides probability confidence after model calibration
- Generates detailed patient report with results
- Supports downloadable CSV report
- Interactive, clinician-friendly Streamlit dashboard
---
## **Tech Stack**
- **Python 3.x**
- **Streamlit** – Interactive UI framework  
- **scikit-learn** – Model building and preprocessing  
- **XGBoost** – Gradient boosting algorithm  
- **pandas, numpy** – Data analysis and numerical operations  
- **matplotlib** – Visualization and calibration plots  
- **statsmodels** – Statistical validation  
- **joblib** – Model persistence  

---

## **Project Structure**
Project2_OCD_Classification/
- │
- ├── data/ # Dataset files (CSV or Excel)
- ├── models/ # Saved ML models (joblib format)
- ├── notebooks/ # Jupyter notebooks for analysis
- ├── app.py # Streamlit web app
- ├── requirements.txt # All dependencies
- ├── README.md # Project documentation
- └── ocd_model_pipeline.ipynb # Main notebook used for analysis and model building

---

## ⚙️ Installation & Setup

### Step 1️: Clone the repository
```bash
git clone https://github.com/ShravaniB18/OCD-Classification.git
cd OCD-Classification 
```
Step 2: Install dependencies
```bash
pip install -r requirements.txt
```
Step 3: Run the Streamlit app
```bash
streamlit run app.py
```



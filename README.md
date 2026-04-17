# Point Cloud Classification

This project classifies feasible vs infeasible manufacturing parts using 3D point cloud data.

## Methods
- Feature engineering (basic + augmented)
- PCA visualization
- Logistic Regression, SVM, Random Forest
- Pipeline comparison using F1 score

## Best Pipeline
Logistic Regression + StandardScaler  
F1 Score: 0.86

## Files
- `ISE5334HW04.ipynb` – main analysis
- `app.py` – Streamlit app
- `requirements.txt` – dependencies

## Run Locally
pip install -r requirements.txt  
streamlit run app.py

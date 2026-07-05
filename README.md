# Credit Scoring Model


## 📌 Objective
Predict whether a person is **creditworthy (good credit)** or **not (bad credit)** using financial history features such as income, debt, payment history, credit history length, etc.

## 🧠 Approach
- Generates a realistic synthetic financial dataset (2,000 records) — no external download needed.
- Trains and compares 3 classification models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
- Evaluates each model using **Accuracy, Precision, Recall, F1-Score, and ROC-AUC**.
- Produces visualizations: model comparison chart, confusion matrix, feature importance chart.

> Want to use a real dataset instead (e.g. the UCI "German Credit Data")? Just replace the `create_dataset()` function in `credit_scoring.py` with `pd.read_csv("your_file.csv")` — the rest of the pipeline works unchanged.

## 📂 Project Structure
```
CreditScoringModel/
│
├── credit_scoring.py              # Main script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── credit_data.csv                 # Generated dataset (created after running)
├── model_comparison_results.csv    # Metrics table (created after running)
├── model_comparison.png            # Bar chart of all metrics (created after running)
├── confusion_matrix.png            # Confusion matrix for best model (created after running)
└── feature_importance.png          # Random Forest feature importance (created after running)
```

---

## 🛠 Setup Instructions (Anaconda + VS Code)

### 1. Install prerequisites
- [Anaconda](https://www.anaconda.com/download) (includes Python + package manager)
- [VS Code](https://code.visualstudio.com/)
- VS Code extensions: **Python** (by Microsoft) and **Jupyter** (optional)

### 2. Create and activate a conda environment
Open **Anaconda Prompt** (Windows) or your terminal (Mac/Linux):
```bash
conda create -n codealpha-ml python=3.10 -y
conda activate codealpha-ml
```

### 3. Get the project files
Copy the `CreditScoringModel` folder anywhere on your computer, e.g.:
```bash
cd Desktop
```
(place the folder there)

### 4. Install dependencies
```bash
cd CreditScoringModel
pip install -r requirements.txt
```

### 5. Open the project in VS Code
```bash
code .
```
In VS Code, press `Ctrl+Shift+P` → **"Python: Select Interpreter"** → choose the `codealpha-ml` conda environment.

### 6. Run the script
In the VS Code terminal (make sure the conda env is activated):
```bash
python credit_scoring.py
```

You'll see all metrics printed in the terminal, and the CSV/PNG output files will appear in the project folder.


---

## 📊 Sample Results (synthetic dataset)

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|----------------------|----------|-----------|--------|----------|---------|
| Logistic Regression  | ~0.91    | ~0.92     | ~0.94  | ~0.93    | ~0.97   |
| Decision Tree         | ~0.87    | ~0.89     | ~0.92  | ~0.90    | ~0.90   |
| Random Forest         | ~0.89    | ~0.89     | ~0.94  | ~0.92    | ~0.95   |

(Exact numbers will vary slightly depending on the random seed and Python/library versions.)

## 📜 License

MIT

---

## ⭐ Support

If you found this project useful:

* Give it a ⭐ on GitHub

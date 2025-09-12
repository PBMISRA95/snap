# Fraud Detection Challenge

A machine learning project for credit card fraud detection using various classification algorithms with imbalanced dataset handling.

## Repository Structure

```
├── dataset.csv                   # Raw fraud detection dataset
├── dataprep_module.py            # Data preprocessing utilities
├── requirements.txt              # Python dependencies
├── 1-eda.ipynb                   # Task 1 - Exploratory Data Analysis
├── 2-dataprep.ipynb              # Task 2 - Data preprocessing pipeline
├── 3-featureselection.ipynb      # Task 3 - Feature selection techniques
├── 4-basemodel.ipynb             # Task 4 - Baseline model implementation
├── 5_&_6-model_sel_n_final.ipynb # Task 5 & 6 - Model selection and Final Model
├── rough_work                    # repo housing experimentation 
├──├── 2-dataprep.ipynb             # dataprep experimentation 
├──├── 5-modelselection.ipynb       # model selection experimentation    
├──├── eda.ipynb                    # EDA experimentation
├──├── rough_work.ipynb             # Rough Work experimentation
└── README.md                      # Project documentation
```

## Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook

### Installation
```bash
pip install -r requirements.txt
```

### Usage
Run the notebooks/scripts in sequence:
1. **1-eda.ipynb** - Explore dataset characteristics and fraud patterns
2. **2-dataprep.py** - Module to handle missing values, outliers, and class imbalance
3. **3-featureselection.ipynb** - Select optimal features for modeling
4. **4-basemodel.ipynb** - Implement baseline classification models
5. **5-modelselection.ipynb** - Compare models and select best performer

## Key Features

• **Multiple Models used** - Random Forest, Decision Tree, XGBoost, Extra Trees, Gradient Boosting
• **Cross-Validation** - 5-fold stratified CV with comprehensive metrics
• **Performance Metrics** - ROC-AUC, Precision, Recall, F1-Score, Accuracy

## Dataset
Credit card transactions with ~20K samples, 1.91% fraud rate, requiring careful handling of class imbalance.

## Results
Best performing model: **Extra Trees Classifier**
- ROC-AUC: 0.699
- Handles class imbalance effectively
- Optimized for fraud detection recall



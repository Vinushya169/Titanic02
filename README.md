🚢 Titanic Survival Prediction

An End-to-End Data Science & Machine Learning Project

<p align="center">
  <strong>Transforming Historical Data into Meaningful Insights Through Data Science</strong>
</p><p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter" />
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas" />
  <img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikitlearn" />
  <img src="https://img.shields.io/badge/Project-Completed-success" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" />
</p>---

📌 01. Project Overview

The Titanic Survival Prediction project is an end-to-end Data Science project focused on analyzing passenger information and predicting survival outcomes using Machine Learning algorithms.

The project explores historical passenger data to identify the key factors that influenced survival during the Titanic disaster.

Rather than focusing only on prediction accuracy, this project emphasizes:

- Exploratory Data Analysis (EDA)
- Data Cleaning & Preprocessing
- Feature Engineering
- Statistical Analysis
- Machine Learning Model Development
- Model Evaluation & Comparison
- Feature Importance & Explainable AI
- Executive Dashboard & Data Storytelling

Objective: Build an interpretable and reliable Machine Learning model that predicts passenger survival based on demographic and travel-related attributes.

🎯 02. Project Objectives

- Understand passenger demographics and survival patterns.
- Identify relationships between passenger characteristics and survival.
- Handle missing values and inconsistent data.
- Perform meaningful feature engineering.
- Build and compare multiple classification models.
- Identify the most influential survival predictors.
- Present analytical findings through an executive dashboard.

📂 03. Dataset Description

Dataset: "titanic_large.csv"

Dataset Information| Details
Total Records| 1,309
Total Columns| 11
Problem Type| Binary Classification
Target Variable| Survived
Missing Data| Present
Data Format| CSV

Dataset Attributes

Feature| Description
Survived| Survival status (0 = No, 1 = Yes)
Pclass| Passenger ticket class
Sex| Passenger gender
Age| Passenger age
SibSp| Siblings / Spouses aboard
Parch| Parents / Children aboard
Fare| Ticket fare
Embarked| Port of embarkation
Cabin| Cabin identification
Ticket| Ticket number
Name| Passenger name

Data Preprocessing

Feature| Issue| Treatment
Age| Missing values| Median imputation by title
Cabin| High missing percentage| HasCabin feature creation
Embarked| 2 missing values| Mode imputation
Fare| Skewed distribution| Log transformation

🔄 04. Project Workflow

The project follows a structured Data Science lifecycle:

1. Data Collection & Loading
2. Data Understanding
3. Data Cleaning & Preprocessing
4. Exploratory Data Analysis
5. Feature Engineering
6. Data Transformation
7. Machine Learning Model Development
8. Model Evaluation & Comparison
9. Feature Importance Analysis
10. Executive Dashboard Development

flowchart TD
    A["Dataset Loading"] --> B["Data Understanding"]
    B --> C["Data Cleaning"]
    C --> D["Exploratory Data Analysis"]
    D --> E["Feature Engineering"]
    E --> F["Machine Learning"]
    F --> G["Model Evaluation"]
    G --> H["Feature Importance"]
    H --> I["Executive Dashboard"]
    I --> J["Insights & Conclusions"]

📊 05. Exploratory Data Analysis (EDA)

Performed 25 analytical steps to understand passenger characteristics, survival distributions, and relationships between variables.

A. Passenger Demographics & Overview

1. Overall Survival Distribution
2. Passenger Class Distribution
3. Gender Distribution
4. Age Distribution with KDE
5. Fare Distribution
6. Embarkation Port Analysis

B. Survival Analysis

7. Gender vs Survival
8. Passenger Class vs Survival
9. Age vs Survival
10. SibSp vs Survival
11. Parch vs Survival
12. Family Size vs Survival
13. Fare vs Survival
14. Embarkation Port vs Survival

C. Advanced & Multivariate Analysis

15. Passenger Class × Gender Heatmap
16. Age Group × Passenger Class
17. Fare Per Person Analysis
18. Passenger Title Extraction
19. Title vs Survival
20. Alone vs Family Survival
21. Age × Class Interaction
22. Correlation Heatmap
23. Numerical Features Pairplot

Key EDA Findings:

- Female passengers demonstrated significantly higher survival rates.
- First-class passengers had better survival outcomes.
- Children showed comparatively higher survival rates.
- Family size influenced survival probability.
- Higher ticket fares were associated with better survival outcomes.

🧩 06. Feature Engineering

Created six additional features to improve the analytical depth and predictive performance.

Engineered Features

1. Title Extraction

Extracted passenger titles such as Mr, Mrs, Miss, and Master from names.

df['Title'] = df['Name'].str.extract(
    ' ([A-Za-z]+)\.', expand=False
)

2. Family Size

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

3. IsAlone

df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

4. HasCabin

df['HasCabin'] = df['Cabin'].notna().astype(int)

5. Age Group

df['AgeGroup'] = pd.cut(
    df['Age'],
    bins=[0, 12, 20, 40, 60, 100],
    labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior']
)

6. Fare Per Person

df['FarePerPerson'] = df['Fare'] / df['FamilySize']

Final Modeling Features

"Pclass", "Sex", "Age", "Fare", "FamilySize", "IsAlone", "HasCabin", "Title", "Embarked"

Categorical features were transformed using One-Hot Encoding.

🤖 07. Machine Learning Models

Developed and compared three classification algorithms.

Model 1: Logistic Regression

- Used as a baseline classification model.
- Applied StandardScaler for numerical feature scaling.
- Used L2 regularization.
- Provides interpretable model coefficients.

Model 2: Random Forest Classifier

- Ensemble learning algorithm.
- Used 200 decision trees.
- Maximum depth: 8.
- Captures nonlinear relationships.
- Provides feature importance measurements.

Model 3: Gradient Boosting Classifier

- Sequential ensemble learning approach.
- Number of estimators: 150.
- Learning rate: 0.1.
- Focuses on correcting previous model errors.

Training Configuration

- Train-Test Split: 80:20
- Random State: 42
- Stratified Sampling: Enabled
- Cross Validation: 5-Fold

📈 08. Model Performance Evaluation

Models were evaluated using multiple classification metrics.

Algorithm| Accuracy| Precision| Recall| F1-Score| ROC-AUC
Logistic Regression| 80.9%| 78%| 72%| 75%| 0.86
Random Forest| 83.1%| 80%| 78%| 79%| 0.88
Gradient Boosting| 84.3%| 82%| 80%| 81%| 0.89

Best Performing Model: Gradient Boosting Classifier

The Gradient Boosting model achieved the highest reported accuracy of 84.3%.

Confusion Matrix Analysis

Metric| Value
True Positives| 85
True Negatives| 136
False Positives| 16
False Negatives| 25

⭐ 09. Feature Importance & Explainable AI

Analyzed the influence of different passenger attributes using three feature importance approaches.

A. Random Forest Feature Importance

Feature| Importance
Sex_female| 0.24
Fare| 0.18
Age| 0.15
Pclass_3| 0.11
Title_Mr| 0.09

B. Permutation Importance

Permutation importance measures the decrease in model performance when a feature is shuffled.

Feature| ROC-AUC Decrease
Sex| 0.12
Fare| 0.05
Pclass| 0.04

C. Logistic Regression Coefficients

- Female passengers: Positive coefficient (+2.62).
- First-class passengers: Positive coefficient (+1.1).
- Fare: Positive coefficient (+0.6).
- Third-class passengers: Negative coefficient (-1.2).
- Male passengers: Negative coefficient (-2.62).

Key Finding: Gender was the strongest predictor in the reported feature importance analysis, followed by passenger class and fare.

Note: Feature importance and coefficients are model-specific associations, not proof of causation.

👔 10. Executive Dashboard

Designed a professional 6-panel Executive Dashboard to communicate the most important analytical findings.

Dashboard File: "25_executive_dashboard.png"

Panel| Visualization| Purpose
01| KPI Cards| Passenger and survival overview
02| Gender vs Survival| Gender-based survival comparison
03| Class vs Survival| Passenger class analysis
04| Class × Gender Heatmap| Combined survival patterns
05| Age Distribution| Age-based survival comparison
06| Model Performance| Machine Learning comparison

The dashboard transforms analytical results into a simple visual presentation for quick decision-making.

💡 11. Key Insights & Findings

1. Gender: Female passengers had considerably higher survival rates than male passengers.
2. Passenger Class: First-class passengers experienced better survival outcomes than third-class passengers.
3. Family Size: Small family groups showed comparatively better survival patterns.
4. Age: Children demonstrated higher survival rates than several adult age groups.
5. Fare: Higher ticket fares were associated with increased survival probability.

Overall Conclusion

The analysis highlights how passenger demographics, socioeconomic status, and family structure were associated with Titanic survival outcomes.

Machine Learning helped identify important patterns and predict survival probabilities using historical passenger data.

🛠️ 12. Technology Stack

Category| Technologies
Programming Language| Python
Data Manipulation| Pandas, NumPy
Data Visualization| Matplotlib, Seaborn
Machine Learning| Scikit-learn
Development Environment| Jupyter Notebook, Anaconda
Version Control| Git, GitHub Desktop
Dataset Format| CSV

📁 13. Repository Structure

Titanic02/
│
├── Titanic02.ipynb
├── titanic_large.csv
├── README.md
├── requirements.txt
├── .gitignore
├── .gitattributes
│
└── visuals/
    ├── 07_gender_survival.png
    ├── 15_class_gender_heatmap.png
    ├── 24_feature_importance.png
    └── 25_executive_dashboard.png

▶️ 14. Installation & Execution

Prerequisites

- Python 3.9 or above
- Jupyter Notebook
- Required Python libraries

Step 1: Clone Repository

git clone https://github.com/Vinushya169/Titanic02.git
cd Titanic02

Step 2: Create Virtual Environment

python -m venv venv

Windows:

venv\Scripts\activate

Mac/Linux:

source venv/bin/activate

Step 3: Install Dependencies

pip install -r requirements.txt

Alternatively:

pip install pandas numpy matplotlib seaborn scikit-learn jupyter joblib

Step 4: Launch Jupyter Notebook

jupyter notebook Titanic02.ipynb

Google Colab

1. Open Google Colab.
2. Upload "Titanic02.ipynb".
3. Upload "titanic_large.csv".
4. Execute all notebook cells.

🚀 15. Future Enhancements

The project can be further enhanced through:

- [ ] XGBoost and LightGBM model comparison.
- [ ] Hyperparameter optimization using GridSearchCV.
- [ ] SHAP-based model explainability.
- [ ] Interactive Streamlit prediction application.
- [ ] Model deployment using Hugging Face Spaces.
- [ ] Docker-based deployment.
- [ ] Improved cross-validation and model generalization.

👩‍💻 16. About the Author

Vinushya M

Aspiring Data Analyst | Data Science & Machine Learning Enthusiast

📍 Kallakurichi, Tamil Nadu, India

Technical Interests

- Data Analytics
- Data Visualization
- Machine Learning
- Python Programming
- UI/UX Design

Connect With Me

- LinkedIn: https://www.linkedin.com/in/vinushya-m-80b92b400
- GitHub: https://github.com/Vinushya169
- Email: vinumuthu3@gmail.com

🙏 17. Acknowledgements

- Dataset: Kaggle Titanic Competition
- Historical Reference: RMS Titanic
- Machine Learning Documentation: Scikit-learn
- Data Analysis: Pandas & NumPy

📜 18. License

This project is licensed under the MIT License.

⭐ Support

If you find this project interesting, consider giving this repository a ⭐ on GitHub.

---

<p align="center">
  <strong>Turning Data into Insights, and Insights into Better Decisions.</strong>
</p><p align="center">
  Built with ❤️ using Python, Data Science & Machine Learning.
</p>

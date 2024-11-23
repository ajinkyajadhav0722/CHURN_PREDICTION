# Telecom Customer Churn Prediction

## Overview
This project focuses on predicting customer churn for a telecom company by leveraging **Exploratory Data Analysis (EDA)** and **Machine Learning Models**. Two distinct approaches were implemented:
1. **EDA and Feature Engineering**: Understanding the data and deriving meaningful insights to address churn.
2. **Model Building and Optimization**: Developing predictive models, addressing class imbalance, and improving performance with advanced techniques like SMOTEENN.

## Key Highlights
- Implemented comprehensive **EDA** to uncover factors influencing churn.
- Developed **Decision Tree** and **Random Forest** models to predict churn.
- Applied **SMOTEENN** to handle class imbalance and improve minority class performance.
- Achieved a maximum accuracy of **95.2%** with Random Forest after optimization.
- Visualized customer behavior using plots like KDE, bar charts, and heatmaps.

## Dataset
The dataset contains customer information from a telecom provider, including:
- **Demographics**: Gender, senior citizen status.
- **Services**: Internet service, tech support, and streaming services.
- **Account Information**: Monthly charges, total charges, payment method.
- **Target Variable**: `Churn` (1 = Yes, 0 = No).

## Project Workflow
1. **Data Cleaning**
   - Converted `TotalCharges` to numeric and handled missing values.
   - Created new features like `tenure_group` to categorize customer tenure.
   - Dropped irrelevant columns (`customerID`) to streamline modeling.

2. **EDA**
   - Visualized key relationships between features and churn using:
     - **KDE Plots**: Monthly and total charges by churn status.
     - **Correlation Heatmaps**: To understand feature importance.
     - **Univariate/Multivariate Plots**: For features like contract type, payment method, and services.

3. **Feature Engineering**
   - One-hot encoding was applied to categorical variables, resulting in 51 features.
   - Standardized numerical features for consistency.

4. **Modeling**
   - Built and evaluated baseline models:
     - **Decision Tree**: Initial accuracy of **79%**.
     - **Random Forest**: Slightly better results but struggled with class imbalance.

5. **Model Optimization**
   - Optimized models post-SMOTEENN:
     - **Decision Tree**: Accuracy improved to **94%**.
     - **Random Forest**: Achieved the best results with **95.2% accuracy** and **96% F1-score** for churned customers.
   - Dimensionality reduction using PCA was tested but did not yield improvements.

6. **Business Insights**
   - Monthly contract customers churn more than yearly contracts.
   - Customers without online security or tech support are at higher risk.
   - Electronic check users have the highest churn rates.

## Results
| Model                  | Accuracy | Precision | Recall | F1-Score |
|------------------------|----------|-----------|--------|----------|
| Decision Tree          | 79.0%    | 65%       | 44%    | 52%      |
| Decision Tree + SMOTEENN | 94.0%    | 94%       | 95%    | 95%      |
| Random Forest          | 79.4%    | 68%       | 41%    | 51%      |
| Random Forest + SMOTEENN | 95.2%    | 94%       | 98%    | 96%      |

## Technologies Used
- **Programming Languages**: Python
- **Libraries**: `pandas`, `numpy`, `seaborn`, `matplotlib`, `scikit-learn`, `imbalanced-learn`
- **Machine Learning Models**: Decision Tree, R

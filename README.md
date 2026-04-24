# Used-Car-Price-Prediction-Model-🚗

## 📋 Project Overview
In the used-vehicle market, price evaluation is often subjective and varies due to multiple factors such as vehicle age, manufacturer, mileage, fuel type, and condition. Buyers and sellers often lack reliable and data-backed valuation methods. 

The goal of this project is to develop a machine learning model that accurately predicts the resale value of used vehicles using historical data. Among the models tested, the **XGBoost Regressor** emerged as the highest-performing model, which was then deployed as a web application to provide real-time price estimates.

## 🔗 Important Links
* **Live Web Application (Streamlit)**: [Vehicle Price Prediction App](https://car-prediction-model-app-mrittick.streamlit.app/)
* **Dataset Source (Kaggle)**: [Craigslist Car/Truck Data](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data)

## 🛠️ Tools and Technologies
* **Programming:** Python 3.x
* **Data Manipulation:** NumPy, Pandas
* **Data Visualization:** Matplotlib, Seaborn, Power BI
* **Machine Learning:** Scikit-learn, XGBoost
* **Development/Deployment:** Jupyter Notebook, Google Colab, Streamlit

## ⚙️ Methodology & Implementation
1. **Data Loading and Cleaning**:
   * Filtered records for relevant vehicles: `year >= 2015`, `price` between \$500 and \$100,000, and `odometer` between 0 and 500,000 miles.
   * Dropped unnecessary columns (URLs, image links, VIN, lat/long, county, etc.).
   * Handled missing values in categorical features (e.g., model, condition, fuel) by replacing them with `'Unknown'`.
2. **Feature Engineering**:
   * Created `car_age` = Current Year - Manufacture Year.
   * Created `price_per_mile` = Price / Odometer.
3. **Exploratory Data Analysis (EDA)**:
   * Visualized price and odometer distributions using histograms.
   * Analyzed the price distribution across the top manufacturers and vehicle conditions using boxplots.
   * Generated correlation heatmaps and pair plots to visualize multi-feature relationships.
4. **Model Building & Preprocessing**:
   * Selected relevant numerical and categorical features.
   * Applied One-Hot Encoding (`get_dummies()`) to categorical features.
   * Split the dataset into 80% training and 20% testing sets.
5. **Model Training & Evaluation**:
   * Trained multiple regression models: Linear Regression, Random Forest Regressor, and XGBoost Regressor.
   * Evaluated performance using R² Score, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).

## 📊 Model Performance Comparison
Based on the final code execution, the models performed as follows on the testing set:

| Model | Test R² Score | Test MAE | Test RMSE |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | ~ 0.59 | 5998.78 | 8873.47 |
| **Random Forest Regressor** | ~ 0.91 | 1764.03 | 4179.07 |
| **XGBoost Regressor** | ~ 0.985 | 862.52 | 1700.13 |

**XGBoost** vastly outperformed the other models, successfully capturing complex, non-linear relationships without heavily overfitting, making it highly suitable for real-world predictive tasks.

## 💡 Intrinsic Feature Importance
According to the XGBoost model, the features that most significantly impact a vehicle's resale value are:
1. Car Age (Year of Manufacture)
2. Odometer Reading (Mileage)
3. Manufacturer
4. Fuel Type
5. Transmission Type
6. Condition of the Vehicle

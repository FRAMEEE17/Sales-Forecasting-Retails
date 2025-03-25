#  Multi-Model Approach to Store Sales Forecasting in Retail Business
## Problem Context
The dataset provided represents historical sales data spanning a five-year period from January 2013 to December 2017. The primary objective of this project is to accurately forecast item-level daily sales for each store location for the next three months (90-day forecast horizon). The granularity of forecasting at the item-store level allows for precise demand planning, inventory management, and operational decisions.

## Challenges:
- High dimensionality: Predicting across 500 distinct item-store combinations
- Seasonality and Trends: Accurately capturing and forecasting seasonal patterns, especially during peak periods.
- Handling Outliers and Noise in time series
- Model Selection and Validation
  
## 🔍 Methodology Overview
We utilized the following structured approach:
### Data Preparation:
- Imputed missing sales data via forward/backward fills.
- Outlier detection and smoothing using Z-score and rolling-window mean.
- Created extensive time-based and rolling features for models that required engineered features (XGBoost).
- Constructed unified datasets (chronos_train_df, chronos_test_df) for transformer-based Chronos forecasting.

## Dataset Analysis
Time Frame:
- Start: January 1, 2013
- End: December 31, 2017
Frequency:
- Daily observations
Dimensions:
- Stores: 10 unique store locations (Store IDs: 1–10)
- Items: 50 unique items/products (Item IDs: 1–50)
- Resulting in 500 unique item-store combinations.

Each record in the dataset comprises the following fields:
- date: The calendar date (daily frequency).
- store: Store identifier, ranging from 1 to 10.
- item: Item identifier, ranging from 1 to 50.
- sales: Number of units sold of the specific item at the given store on that particular date.

## EDA
![image](https://github.com/user-attachments/assets/c7dc1b77-2d74-411a-a009-9dd184bd0f52)

![image](https://github.com/user-attachments/assets/87df0b59-58c5-436e-828e-e9b6e6c05569)

### Data Characteristics
- Strong seasonal patterns are clearly visible with peaks indicating higher sales, likely corresponding to holidays, weekends, or specific promotional periods.
- Yearly seasonality: Sales peak around the same time each year, reflecting recurring consumer purchasing behaviors.
- Weekly cycles: Regular spikes typically occur during weekends, indicating higher consumer activity.
- Trend Component: An observable upward and downward trend across different years, possibly influenced by economic factors or store-specific events.

## Time Series
- White Noise Test: Ljung-Box test indicated significant autocorrelation (p-value < 0.05), confirming data is not white noise, thereby suitable for forecasting.
- Seasonal Decomposition:
1. Weekly decomposition (period = 7) exhibited limited weekly seasonality, with relatively stable fluctuations.![image](https://github.com/user-attachments/assets/828391b4-ba97-4b58-92ab-4bc3c0eb0726)
2. Monthly decomposition (period = 30) no seasonal patter quite flat with relatively the same pattern up-down ward.
![image](https://github.com/user-attachments/assets/804e06b8-5b61-41e5-8997-f9dd8486e161)

2. Annual decomposition (period = 365) revealed a clear yearly seasonal pattern with consistent seasonal amplitudes, suggesting an additive model. The trend component showed gradual growth.
![image](https://github.com/user-attachments/assets/75dc2e25-9569-4e73-8fb6-ffb57e6f6042)

- Stationarity Tests: Both ADF (p-value = 0.0486) and KPSS (p-value > 0.05) tests confirmed stationarity, eliminating the necessity for further differencing.

-  Trend Analysis using OLS Regression
To quantify the long-term trend of daily sales, we applied Ordinary Least Squares (OLS) regression with time (date) as the independent variable and daily sales as the dependent variable. Below are the key statistical takeaways

![image](https://github.com/user-attachments/assets/cd2ddb8f-38c7-4041-90f1-e84c5e55c4b1)

- Slope (β₁) = 0.0103
This implies that, on average, daily sales increased by 0.0103 units per day over the span of the dataset. Although numerically small, this reflects a gradual upward trend in baseline sales over time.
- P-value for slope = 0.000
This highly significant p-value indicates that the upward slope is unlikely due to random fluctuations. We can confidently conclude that the trend is statistically meaningful.
- R-squared = 0.141
Only 14.1% of the variation in sales is explained by the linear trend. This is expected for time series data where seasonality and cyclical patterns dominate. Hence, more complex models like SARIMA or Chronos T5 are needed to capture additional patterns beyond linear growth.
- Durbin-Watson Statistic = 1.031
This value, being close to 1, suggests positive autocorrelation in the residuals — a typical signature in time series data. It confirms that residuals are not white noise and still contain valuable time-dependent structure. This further justifies using autoregressive models over simple OLS.
    
### Forecasting Models Compared
1. AutoReg (AR)
2. SARIMA 
3. XGBoost
4. Chronos-T5 (Transformer) 


### Evaluation :

- Short-term forecasts on eval set (30 days), test set forecasts (90 days), comparing RMSE, MAE, and SMAPE metrics.

- Used robust backtesting (historical splits) to validate model performance and generalization.

## 📌 Key Findings & Analytical Insights


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
- Stationarity Tests: Both ADF (p-value = 0.0486) and KPSS (p-value > 0.05) tests confirmed stationarity, eliminating the necessity for further differencing.
  
### Decomposition:
1. Weekly decomposition (period = 7) exhibited limited weekly seasonality, with relatively stable fluctuations.
- Observed (daily sales) : มีลักษณะขึ้นลงแบบมีลูป ค่าต่างๆ แกว่งพอประมาณ มีความเป็น pattern
- Trend : pattern ขึ้นลงเหมือนกันทุกปี  ดูเหมือนจะมีบางช่วงที่ยอดขายสูงขึ้นต่อเนื่อง (น่าจะมี seasonal demand boost หรือ promotion)
- Seasonal :  Seasonal flat (แทบไม่มี pattern รายสัปดาห์)  ค่าแกว่งน้อย คงที่มาก
![image](https://github.com/user-attachments/assets/828391b4-ba97-4b58-92ab-4bc3c0eb0726)

2. Monthly decomposition (period = 30) no seasonal patter quite flat with relatively the same pattern up-down ward.
![image](https://github.com/user-attachments/assets/804e06b8-5b61-41e5-8997-f9dd8486e161)
- Observed (daily sales) : มีลักษณะขึ้นลงแบบมีลูป ค่าต่างๆ แกว่งพอประมาณ มีความเป็น pattern
- Trend : pattern ขึ้นลงเหมือนกันทุกปี  strong smooth มาก
- Seasonal :  Seasonal flat (แทบไม่มี pattern รายสัปดาห์)  ค่าแกว่งน้อย คงที่มาก

3. Annual decomposition (period = 365) revealed a clear yearly seasonal pattern with consistent seasonal amplitudes, suggesting an additive model. The trend component showed gradual growth.
![image](https://github.com/user-attachments/assets/75dc2e25-9569-4e73-8fb6-ffb57e6f6042)

### Trend Analysis using OLS Regression
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

### 🔁 ACF/PACF for SARIMA Specification
PACF: Significant spikes at lags 1–2; best fit for AR(2)
ACF: Gradual decay with seasonality at lags 7, 14, 21 — clear weekly cycles
- Insights : SARIMA(p=2,d=0,q=1)(P,D,Q,s=7) for baseline statistical forecasting

### Forecasting Models Compared  (Univariate and Multivariate Time Series)
1. AutoReg (AR) 
2. SARIMA 
3. XGBoost
4. LightGBM
5. Linear/Lasso/Ridge
6. Chronos-T5 (Transformer) 

### Modelling techniques

1. ARIMA/SARIMA
- Cons of ARIMA & SARIMA : Amount of data needed -> Both the algorithms require considerable data to work on, especially if the data is seasonal. For example, using three years of historical demand is likely not to be enough (Short Life-Cycle Products) for a good forecast. -> Ref: https://neptune.ai/blog/arima-sarima-real-world-time-series-forecasting-guide
- AR (Autoregressive) and similar classical models (ARIMA, SARIMA, etc.) are univariate and assume stationarity + structure within a single time series. So you can't batch all stores, items because you're mixing signals (trends, seasonality, noise) from different products, which is statistically invalid for AR model.
### - lags = 20
  ![image](https://github.com/user-attachments/assets/6943c876-4800-42e4-b293-db4ada9fba01)
- Since lags 1–3 contribute significantly, this strongly suggests an AR(p) structure with p=2 or p=3 may be appropriate. After lag 3, there's no consistent signal, so adding higher-order AR terms would risk overfitting and diluting model generalization.
### - lags = 48
![image](https://github.com/user-attachments/assets/003f908f-5298-4595-9dba-2ae41ee20ba9)

lags=48 — Detecting Seasonality
- Observation: Clear, repeating PACF spikes at lags 7, 14, 21, 28, etc. These lags are multiples of 7, pointing to weekly seasonality.
<img width="614" alt="image" src="https://github.com/user-attachments/assets/2fb21f68-de4f-4764-9029-d322639e0fe8" />

So, we use lags = 48, and try ARIMA(p=2, d, q) in training loop.

### Why SARIMA?
Given the seasonality embedded in the sales time series (weekly pattern), and confirmed stationarity post-differencing, SARIMA becomes a natural modeling candidate. Unlike vanilla ARIMA, SARIMA explicitly handles seasonal structure through seasonal autoregressive (P), differencing (D), and moving average (Q) terms over a seasonal period s. For our case, s = 7 (weekly).
1. Stationarity Checks
- ADF Test: p-value < 0.05 → Fail to reject H₀ → Series is stationary → d = 0
- KPSS Test: p-value > 0.05 → Accept H₀ → Series is stationary ⟶  Confirmed stationarity → no need for differencing (d = 0)
2. Identifying Weekly Seasonality (s=7)
- PACF Plot (lags=48): Clear spikes at lags 7, 14, 21, 28, etc. ⟶ This is strong visual evidence of weekly seasonality
- ACF also shows slow decay after seasonal differencing, with early spikes still prominent
🔁 Seasonal Differencing Insights: After applying .diff(7):
- ACF drops rapidly → suggests D = 1 is sufficient
- PACF shows clean structure up to seasonal lag ⟶ Final seasonal config: D = 1, s = 7
3. Order (p,d,q) and (P,D,Q) :
- PACF (lags=20): Lag 1–2 significant
- After lag 3 → falls within 95% CI → cutoff effect seen ⟶ p = 2
ACF (lags=48):
- Lag 1 clearly significant
- Rapid decay beyond lag 2 ⟶ q = 1
- stationary -> d = 0
  
🔸 Seasonal Terms
Seasonal PACF after .diff(7) shows:
- Here's why P = 1 look at the graph below.
![image](https://github.com/user-attachments/assets/d4089fe5-267b-4814-8c34-461c88025d82)

- Q = 1 below
  
![image](https://github.com/user-attachments/assets/f0bc4e76-2029-47e2-bd0b-7f281fe89d35)
- Therefore, Seasonal lags ~7 remain significant ⟶ P = 1, Q = 1
- D = 1/ 2
- Original length: 913000
- After seasonal diff: 912993
  
- As you can see in the fig below, .diff(s)) drops off quickly, → D=1 is likely enough. If not → consider D=2.
![image](https://github.com/user-attachments/assets/51fbc4de-44a9-49cc-abc2-50d27fd7f097)
###⚙ Model Configuration:
We fit a SARIMA model with the following hyperparameters:
- p = 2 (based on PACF significant spikes at lag 1–2)
- q = 1 (identified via ACF)
- d = 0 (confirmed via ADF/KPSS → data is stationary)
- s = 7 (strong weekly seasonality from PACF/ACF seasonal spikes)
- P = 1, D = 1, Q = 1 (derived from seasonal differencing and correlogram)
This configuration reflects a balance between short-term autoregressive structure and weekly seasonal trends, capturing both regular sales cycles and trend shifts across time.

### Error Analysis
![image](https://github.com/user-attachments/assets/e885e730-1c87-4b6f-9df7-b6fca21db0d4)
One of the most crucial steps in time series forecasting is understanding why the model performs poorly in certain segments — and whether it's due to structural limitations (underfitting) or data quality issues (e.g. noise, outliers). To that end, we conducted a local error inspection for Store 1, Item 1 — one of the more volatile SKUs.
### 📈 Rolling Window & Z-Score Approach
- To identify localized anomalies, we employed a 7-day centered rolling average, which provides a smoothed baseline by taking into account the current day ±3 days on either side. This helps us focus not on erratic daily shifts, but on weekly trends, which are more meaningful in retail environments (e.g., weekend boosts, payday spikes).
- Centering (center=True): ensures symmetry — avoids biasing trends toward future or past values.
- Z-Score Outlier Detection: any sales point with a z-score > 3 is flagged as an anomaly, i.e., it deviates more than 3 standard deviations from the 7-day mean.
### Outliers 
- are impactful even if it's rare. Red dots in the plot represent statistically significant sales spikes.
- These spikes deviate significantly from the local trend, suggesting possible events such as:
1.Promotions
2.Holidays
3.External shocks (e.g. panic buying, supply dump)
- The amplitude of these outliers exceeds 40–50 units, while the local rolling mean stays closer to 20–25 units — a ~100% deviation.
Trend + Seasonality Exist, But Are Obscured by Noise
- The raw sales (blue) show cyclical patterns and a smooth seasonal trend (orange) across years.
- However, daily volatility makes direct modeling harder — this is exactly why ARIMA-type models may underperform, especially when not robust to outliers.
- Model Implication:
1. AR(2), ARIMA, and SARIMA tend to smooth over these anomalies, failing to capture spikes (as previously observed in test forecast flattening).
2. These models are mean-reverting by design, which is ill-suited for sharp, rare bursts in data.


### Evaluation 
Short-term forecasts on eval set (30 days), test set forecasts (90 days), comparing RMSE, MAE, and SMAPE metrics. Used robust backtesting (historical splits) to validate model performance and generalization if better.
# 1. Univariate
1.1. AR(2)
  - eval set
![image](https://github.com/user-attachments/assets/954fb7ba-b3fe-424e-8da8-baa00304bcdc)
![image](https://github.com/user-attachments/assets/e386f465-d5cc-4f55-8c86-1350b2267b1f)

average rmse 11.4762
average mae 9.535560000000002
average smape 21.913059999999998
  - test forecast
![image](https://github.com/user-attachments/assets/1f3b3c61-9342-459f-8e0e-016b9f86f499)

1.2. SARIMA
   - Backtest on eval set - 30 days
![image](https://github.com/user-attachments/assets/10ffba01-fe43-442f-900c-2fc9afa9ed8f)
![image](https://github.com/user-attachments/assets/89cd8484-74a2-4e83-9dd8-c6567b505f98)
![image](https://github.com/user-attachments/assets/2043b6a9-74a7-4250-b3e7-77ebeb7d8c68)

     <img width="673" alt="image" src="https://github.com/user-attachments/assets/442ff463-a361-42fd-8468-e50a6091b7a0" />
   - test forecast
![image](https://github.com/user-attachments/assets/dc014746-c7f0-4ae4-83bc-a2f85e4d3f3c)

2. Multivariate

2.1 Feature Engineering
- We constructed four categories of features:
1. Date-Time Features
2. Lag Features
3. Rolling Window Features
4. Local Trend Features (Slope)
- Additionally, external regressors like Thai public holidays were incorporated for real-world signal enrichment.
    <img width="611" alt="image" src="https://github.com/user-attachments/assets/9a80622c-b7cc-4f1d-bd47-9ba69992fc93" />
    <img width="611" alt="image" src="https://github.com/user-attachments/assets/b60caf28-3486-467d-829d-df760da52979" />
    <img width="609" alt="image" src="https://github.com/user-attachments/assets/80837249-51f8-4dbb-973a-13fc04ec8179" />
    <img width="613" alt="image" src="https://github.com/user-attachments/assets/db56b11c-8da5-4e0d-97e4-2e024dee43ee" />

### How?
- Grouped Features: All lag, rolling, and slope features were computed per store-item group, preserving time-local structure. This prevents leakage across unrelated products.
- Shifted Targets: Features are shifted forward to avoid data leakage into the prediction window.
  
2.2 AutoGluon
2.3 XGBoost
2.4 LightGBM
2.5 Linear/Ridge/Lasso Regression
2.6 Chronos T5 Small



### Performance Assessment of AR(2)
- Mean SMAPE ≈ 21–22% across store-item pairs. Daily-level sales are naturally noisy. AR(2) is a low-capacity model, lacking external regressors or exogenous structure
Right-skewed distribution (Positive Skew) of SMAPE values : Majority of forecasts are within decent error range (centered ~20%). But there’s a long tail of underperforming forecasts, where SMAPE is much higher → Certain store-item pairs are likely harder to predict due to irregular or bursty demand, promotions, or external events
- AR(2) over-weights recent lags and lacks a memory window. Model fails to capture sudden demand spikes, dips, or seasonal bursts. This is expected behavior — AR models produce mean-reverting, smooth trajectories which cannot mimic periodic volatility
- Combined with flat predictions, this suggests the model is underfitting → not able to explain full variance. Could be due to model bias (too simple), not variance
### Performance Assessment of SARIMA (2,0,1) (1,1,1)(7)
- SARIMA outperforms ARIMA in capturing recurring demand cycles and weekly seasonal patterns. This is evident in the wave-like structure of forecasts and smoother residuals after differencing.
- Denoising via rolling means significantly boosts signal clarity, especially in items with high short-term volatility. Combined with outlier suppression, this step is vital for boosting SARIMA’s reliability.
However, SARIMA still struggles with Store-item pairs that have erratic or bursty demand (see outlier map), Sparse data (short life-cycle SKUs), Discontinuous patterns (e.g. sudden promotion spikes)
For these, multivariate or transformer-based models (Chronos, DeepAR) COULD offer better performance.


## 📌 Key Findings & Analytical Insights


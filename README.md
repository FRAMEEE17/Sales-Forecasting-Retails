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

2.2 AutoGluon 

  ```
    {'enable_ensemble': True,
     'eval_metric': WQL,
     'hyperparameters': 'default',
     'known_covariates_names': [],
     'num_val_windows': 1,
     'prediction_length': 90,
     'quantile_levels': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
     'random_seed': 123,
     'refit_every_n_windows': 1,
     'refit_full': False,
    ...
    Training complete. Models trained: ['SeasonalNaive', 'RecursiveTabular', 'DirectTabular', 'NPTS', 'DynamicOptimizedTheta', 'AutoETS', 'ChronosZeroShot[bolt_base]', 'ChronosFineTuned[bolt_small]', 'TemporalFusionTransformer', 'DeepAR', 'PatchTST', 'TiDE', 'WeightedEnsemble']
    Total runtime: 1302.49 s
    Best model: WeightedEnsemble
    Best model score: -0.0820
   ```
2.3 XGBoost
### Feature Importance
- Date features Only
  
    ![image](https://github.com/user-attachments/assets/8f4c9b19-ff94-4487-9374-206e0617fe59)
  
- Full feature
  
    ![image](https://github.com/user-attachments/assets/27ce420e-84c0-4a47-ba68-57ee4c322d28)
  
```
xgb_model = xgb.XGBRegressor(
    n_estimators=2449,
    max_depth=16,
    learning_rate=0.0997,
    subsample=0.65,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=42
)
```

- Validation RMSE: 10.349061160245094
- Validation MAE: 7.636844550745981
- Test forecast :
 ![image](https://github.com/user-attachments/assets/04ef7333-f935-4810-8786-80aff8ec6f34)
  
~20% improvement vs ARIMA baseline.

2.4 LightGBM

  ```
  params = {
      'objective': 'regression',
      'metric': 'rmse',
      'boosting_type': 'gbdt',
      'learning_rate': 0.009,
      'num_leaves': 31,
      'feature_fraction': 0.9,
      'bagging_fraction': 0.8,
      'verbose': -1,
  }
  ```
- Eval set :
![image](https://github.com/user-attachments/assets/58688395-7083-45a3-858c-4fc77c742905)

- Validation MAE: 0.0261
- Validation RMSE: 0.0409
- Test forecast :
![image](https://github.com/user-attachments/assets/f17b33ad-b492-4a39-8ab8-ec66124d1662)

**UNBELIEVABLE PERFORMANCE!!!**

2.5 Linear/Ridge/Lasso Regression
- Eval set:
without lag/rolling features: 
![image](https://github.com/user-attachments/assets/be5d3d40-c74e-4ed7-b9b9-d284bfa1ab11)

	             MAE	 |  RMSE
      Linear   26.53 | 5.57
      Lasso	 25.71 | 5.49
      Ridge    26.53 | 5.57

I have trained with full features, its results were good but not without lag/rolling features.

2.6 Chronos T5 Small (Only inference!!!)

- Eval set:
<img width="972" alt="image" src="https://github.com/user-attachments/assets/ada29dd6-502c-401b-9f48-02701f034b00" />

- Validation Average RMSE: 10.400113745497572
- Validation Average MAE : 8.22729452085495

**LightGBM > XGBoost > Chronos > Regression**
- Feature engineering significantly improve our performances especially lag, rolling mean, std. But unfortunately, our test set doesn't contain sales features so we can't train with them.


# 📌 Key Findings & Analytical Insights
Our multi-model forecasting pipeline reveals a nuanced landscape of forecasting performance across store-item combinations in a high-noise retail environment. Through a rigorous evaluation of both univariate (ARIMA/SARIMA) and multivariate approaches (XGBoost, LightGBM, Chronos T5, and regressions), several critical insights emerged.

    1. Classical Models Are Stable Baselines — But Not Sufficient:
    SARIMA, with carefully selected hyperparameters (p=2, d=0, q=1, P=1, D=1, Q=1, s=7), performed reasonably well and captured underlying seasonality — particularly weekly cycles — across most store-item pairs. However, its core limitation became apparent when faced with erratic demand, sparse patterns, or external disruptions (e.g. promotions, holidays). With a mean SMAPE of ~30.12%, SARIMA served as a reliable benchmark but failed to adapt to bursty or anomalous behaviors due to its smooth, mean-reverting nature. Error analyses further confirmed underfitting in volatile segments.
    
    2. Feature Engineering Is a Game-Changer:
    When transitioning to multivariate models, the addition of domain-aware engineered features — such as lags, rolling windows, trend slopes, and holiday indicators — dramatically improved performance. Models like XGBoost and LightGBM, when supplied with these features, not only learned temporal dynamics but also handled heterogeneity across items and stores more effectively. In particular, lag and rolling std/mean features were repeatedly ranked top in importance, proving essential for capturing short-term demand momentum and variability.
    
    3. LightGBM Dominated in Performance, Even Outshining Chronos:
    Among all models tested, LightGBM emerged as the top performer with unbelievably low RMSE and MAE values on both evaluation and test sets. Its gradient-based one-sided sampling (GOSS) and histogram-based tree learning allowed it to generalize well even across complex demand patterns. Despite not using advanced architectures like transformers, its structured-data-optimized design, coupled with strong feature engineering, delivered state-of-the-art results. XGBoost followed closely behind, slightly trailing in test accuracy but proving more stable under extreme noise.
    
    4. Transformer-Based Chronos T5 Was Competitive — But Not Superior:
    Chronos T5, a transformer-based sequence model, delivered good performance in inference-only mode, achieving competitive RMSE and MAE. However, without the ability to incorporate historical lag/rolling sales features during inference (due to test set limitations), its true potential was somewhat handicapped. Given more flexible input formats or autoregressive fine-tuning, Chronos could potentially outperform tree-based models — especially in sparse or irregular sales scenarios where attention mechanisms can shine.
    
    5. Regression Models Reinforced the Importance of Feature Complexity:
    Linear, Ridge, and Lasso regressions showed that without lag/rolling-based features, they severely underfit the data. Their simplicity, while fast and interpretable, limited their forecasting expressiveness. This reinforced a key finding: in time series, temporal memory matters — and models must be explicitly guided to capture it, either through autoregressive structures or engineered features.
    
    6. Outliers & Noise Matter — But Can Be Mitigated:
    Our Z-score based outlier detection, coupled with 7-day rolling average smoothing, was critical in suppressing localized anomalies (e.g. demand spikes due to promotions or holidays). These preprocessing steps not only stabilized univariate model forecasts (like SARIMA) but also improved generalization in ML models. While true outliers remained difficult to predict, the models performed better when noise was removed from the training signal.
    
    7. Practical Constraints Shaped Modeling Choices:
    Although lag/rolling features were powerful, we encountered a key limitation: they couldn’t be used in test-time inference unless sales values were available — which they aren’t for future prediction. This made autoregressive ML models (e.g. Recursive XGBoost) challenging to deploy at scale, and raised interest in direct forecasting models (e.g. Chronos T5, LightGBM with calendar features only).

In conclusion, while classical models like SARIMA provide a good foundation, feature-rich multivariate models — particularly LightGBM — delivered the best blend of accuracy, robustness, and interpretability for our retail forecasting problem. Our feature engineering strategy was validated not only by model performance but also by feature importance rankings across experiments. For store-item pairs with irregular behavior, transformer-based models offer promise for future exploration. Moving forward, integrating external signals like promotions, pricing, or macroeconomic indicators could further enhance predictive power — especially in those long-tail, high-error cases.

## 🔭 Future Plan
To further elevate the forecasting performance and unlock deeper insights across item-store combinations, our next phase will focus on three strategic axes: feature expansion, advanced ensembling, and model operationalization at scale.

1. Feature Expansion with Business Context
        While our current pipeline already incorporates rich temporal and statistical features, the inclusion of external covariates — especially business-specific signals — presents a high-impact opportunity to improve model explainability and accuracy:
   
        Incorporating historical and promotional price data would allow the model to learn price elasticity patterns, capturing how sensitive demand is to price changes. Items that are price-driven (e.g., commodities or fast-moving consumer goods) will benefit significantly. We can engineer delta features (e.g., % price change), discount indicators, and rolling price averages.
        
        Geographic & Demographic Metadata (Store Location):
        Encoding store location allows us to embed spatial heterogeneity into the model — accounting for regional demand shifts, urban/rural store dynamics, and local economic conditions. Potential features include:
        
        Store-level population density (proxy for demand)
        
        Regional income tier (e.g., store clusters by GDP zone)
        
        Distance to major cities (e.g., urban vs rural segmentation)
        
        Event Calendars / Local Promotions:
        Aligning the timeline with promotion campaigns, holidays, and seasonal marketing events (e.g., Chinese New Year, Back-to-School) will help models capture burst behavior typically missed by statistical baselines. These can be binary indicators or campaign duration windows.
        
        By incorporating these business-aware features, we shift from pure pattern modeling to contextual forecasting, bridging machine learning with real-world drivers of demand.

3. Advanced Model Ensembling: Weighted Forecast Fusion
   While model diversity is already built into our pipeline (SARIMA, XGBoost, LightGBM, Chronos T5, etc.), we plan to upgrade our ensembling strategy from naïve averaging to performance-aware weighted ensembles.
        
        Use validation metrics (e.g., SMAPE, MAE) per store-item segment to assign optimized weights to each base model.
                
        Consider Bayesian model averaging, stacking (meta-learning), or grid-based ensemble search to dynamically balance accuracy and robustness.
                
        Adjust weights per segment or SKU cluster, since some models (e.g. SARIMA) perform better in stable demand, while others (e.g. XGBoost) excel with irregular, sparse series.
                
        This allows us to leverage each model’s strength where it shines, rather than forcing a one-size-fits-all solution — a key advantage in complex retail environments.

4. Forecasting at Scale: Toward Deployment-Readiness
   
        Instead of training one global model or one per SKU, we will explore clustering items/stores into similar demand profiles (e.g. seasonal vs non-seasonal, bursty vs stable) and assigning best-fit models per segment. 
        
        Set up automated weekly pipelines to ingest recent sales, prices, holidays, and retrain or fine-tune models incrementally using online learning or transfer learning (especially for Chronos).
        
        As we move closer to production, we’ll incorporate SHAP, LIME, and feature impact plots to help business users understand why a forecast looks the way it does — crucial for trust in decision-making.





# Bengaluru Ride-Sharing Predictive Models  

This repository contains predictive models for analyzing ride-sharing data in Bengaluru, focusing on **cancellation prediction**, **demand forecasting**, and **fare estimation**.  

## Models  

1. **Cancellation Prediction**  
   - Predicts whether a booking will be canceled using `XGBoost`.  
   - **Key Features:** Hour, day of week, peak hours, vehicle type.  
   - **Performance:** 68% accuracy, 96% recall for non-cancellations.  

2. **Demand Forecasting**  
   - Forecasts hourly ride demand by vehicle type using `RandomForestRegressor`.  
   - **Key Features:** Lagged demand (previous hour/day/week), rolling averages.  
   - **Performance:** MAE: 1.23, RMSE: 1.64.  

3. **Fare Prediction**  
   - Estimates fare prices for successful rides using `GradientBoostingRegressor`.  
   - **Key Features:** Ride distance, time of day, pickup/drop locations.  
   - **Performance:** RMSE: 327.09 (suggests further tuning).  

## Usage  

1. **Install Dependencies**:  
   ```bash  
   pip install pandas numpy scikit-learn xgboost joblib matplotlib  
   ```  

2. **Run Models**:  
   - Load pre-trained models (`.pkl` files) or retrain using the provided notebooks.  

3. **Deployment**:  
   Models are serialized with `joblib` for integration into applications.  

## Files  

- `bengaluru_ride_data.csv`: Raw dataset (not included; contact for sample).  
- `Predictive_Modelling.pdf`: Full analysis and code walkthrough.  
- `cancellation_model.pkl`, `demand_model.pkl`, `fare_model.pkl`: Pretrained models.  

## Next Steps  
- Improve fare model with additional features (e.g., traffic data).  
- Address class imbalance in cancellation predictions.  



# Hotel Booking Cancellation Prediction

✅Description
This project focuses on creating and maintaining a prediction model to identify whether a hotel reservation will be canceled or not. The goal is to assist a specific hotel in increasing its revenue by optimizing reservation management. The model utilizes a dataset that includes detailed information about hotel reservations.

✅Installation and Usage
To run this project, Python and the following libraries are required: Pandas, Matplotlib, Seaborn, Scikit-Learn, and Joblib. Install the dependencies with pip install -r requirements.txt. To use the model, execute the main script with python main.py.

✅Data
The dataset 'hotel_bookings.csv' contains the following columns:

Booking_ID: Unique reservation identifier.
Demographics: no_of_adults, no_of_children.
Stay details: no_of_weekend_nights, no_of_week_nights.
Reservation information: type_of_meal_plan, required_car_parking_space, room_type_reserved.
Dates: lead_time, arrival_year, arrival_month, arrival_date.
Market segment: market_segment_type, repeated_guest.
Reservation history: no_of_previous_cancellations, no_of_previous_bookings_not_canceled.
Financial details: avg_price_per_room.
Additional services: no_of_special_requests.
Reservation status: booking_status.

✅Analysis
Exploratory data analysis is conducted to better understand the features and patterns that may influence reservation cancellations.

✅Modeling
A machine learning approach is employed to predict cancellations. Various models are experimented with, and the best one is selected based on accuracy and other metrics.

✅Evaluation
The model is evaluated using metrics such as accuracy, recall, and F1-score. A confusion matrix is also provided for a better understanding of the model's performance.

# Hotel Booking Cancellation Prediction Project (Stacking Models)

🧑‍💻Project Overview
This project aims to predict hotel booking cancellations using stacking model techniques. The primary goal is to assist hotels in understanding factors contributing to cancellations, allowing them to implement strategies to minimize them. The project uses the 'hotel_bookings.csv' dataset.

🧑‍💻Data Preprocessing
Imputation: Missing values in numerical and categorical variables are imputed using median and most frequent values, respectively.
Feature Engineering: New features like 'total_stay' and 'weekend_proportion' are created.
Encoding: Categorical variables are encoded using Label Encoding.
Feature Engineering
'total_stay': Sum of 'no_of_weekend_nights' and 'no_of_week_night'.
'weekend_proportion': Proportion of weekend nights in the total stay.

🧑‍💻Model Building
Splitting Data: Data is split into training and testing sets.
Scaling: Features are scaled using StandardScaler.
Stacking Classifier: A Stacking model is created using RandomForestClassifier, LogisticRegression, DecisionTreeClassifier, and SVC.
Training: The model is trained on the training dataset.
Model Saving: The scaler, logistic regression model, and stacking classifier are saved as joblib files.

🧑‍💻Model Evaluation
Classification Report: Provides precision, recall, and F1-score for each class and overall accuracy.
Confusion Matrix: Visual representation of the model's performance.

🧑‍💻Insights from Evaluation
The model shows high precision and recall across both classes.
The overall accuracy is 89%, indicating strong model performance.

🧑‍💻Visualization
Confusion Matrix: A plot is provided to visually interpret the model's performance.

🧑‍💻Usage
Loading Models: Use joblib.load() to load the saved models.
Prediction: Pass the processed input data to the model for prediction.

🧑‍💻Requirements
Python libraries: pandas, joblib, matplotlib, seaborn, sklearn
Files Included
'hotel_bookings.csv': Dataset file.
'scaler_model.joblib': Saved scaler model.
'logistic_regression_model.joblib': Saved logistic regression model.
'stacking_model.joblib': Saved stacking classifier model.

🧑‍💻Notes
Adjust hyperparameters or model selection as needed based on the latest data trends or requirements.
Regularly update the model with new data for maintaining its accuracy.




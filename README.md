Titanic Survival Intelligence Platform
It's an end-to-end, Machine Learning system that predicts passenger survival during the RMS Titanic disaster using historical data, advanced ML models, interactive dashboards, and real-time streaming.

Project Objective: To develop a robust survival prediction system that:
1. Predicts whether a passenger survived or not
2. Uses passenger attributes such as:
    i. Age
    ii. Gender
    iii. Passenger class
    iv. Fare
    v. Family size
3. Allows customized filtering & sorting of predictions
4. Supports:
    i. Batch predictions
    ii. Individual predictions
    iii. Real-time Kafka streaming

Project Highlights
1. Advanced Machine Learning models (Logistic Regression, Random Forest)
2. Feature engineering (family size, titles, encoded variables)
3. Interactive Streamlit Dashboard (KPIs, filtering, sorting)
4. Batch & individual predictions
5. Real-time predictions using Kafka
6. Scalable & container-ready architecture
7. Clean UI

Dataset
Source: https://www.kaggle.com/competitions/titanic
Key Features Used:
Feature	        Description
PassengerId	    Unique ID
Pclass	        Passenger class (1, 2, 3)
Sex	            Gender
Age	            Passenger age
SibSp	        Siblings / Spouse aboard
Parch	        Parents / Children aboard
Fare	        Ticket fare
Cabin	        Cabin number
Embarked	    Port of embarkation

![image alt](https://github.com/DeekshithMP/titanic-survival-prediction/blob/b05d2c8daaef531458e8f710c11f19c116fbeb61/Titanic%20Prediction's%20Flowchart.png)

Machine Learning Pipeline
1. Data Cleaning
    a. Missing value handling (Age, Embarked)
    b. Cabin normalization
    c. Outlier treatment for Fare
2. Feature Engineering
    a. FamilySize = SibSp + Parch + 1
    b. Title extraction from names (Mr, Mrs, Miss, etc.)
    c. One-Hot Encoding for categorical features
3. Models Trained
    a.Logistic Regression
    b.Random Forest (Best Performer)
    c.Hyperparameter tuning with GridSearchCV
    d.Cross-validation for generalization
4. Evaluation Metrics
    a. Accuracy
    b. Precision / Recall / F1-Score
    c. Confusion Matrix
    d. ROC-AUC Curve

Dashboard Capabilities (Streamlit)
1. KPI Indicators
    i. Total passengers
    ii. Average survival probability
    iii. High-risk passenger count
    iv. Model ROC-AUC score
2. Filtering & Sorting
    i.Filter predictions by:
        a. Passenger class
        b. Age range
        c. Gender
        d. Sort passengers by survival probability
3. Batch Predictions
    i. View survival predictions for filtered groups
    ii. Risk scores computed dynamically
4. Individual Prediction Studio
    i. Enter passenger details
    ii. Get:
        a. Survival probability
        b. Risk index
        c. Model prediction
        d. Visual confidence indicators
5. Real-Time Streaming (Kafka)
    i. Live passenger data ingestion
    ii. On-the-fly survival predictions
    iii. Scalable streaming architecture

Real-Time Architecture (Kafka)
    Passenger Data Producer
            ↓
    Kafka Topic (titanic_stream)
            ↓
    ML Prediction Consumer
            ↓
    Live Dashboard Updates
✔ Handles high-volume data
✔ Ready for real-world streaming use cases

Installation & Setup
1. Clone Repository
    git clone https://github.com/DeekshithMP/titanic-survival-prediction.git
    cd titanic-survival-prediction

2. Create Environment
    conda create -n titanic-ai python=3.10
    conda activate titanic-ai

3. Install Dependencies
    pip install -r requirements.txt

4. Run Application
    python -m streamlit run frontend/app.py
    (in new terminal)
    docker compose up -d
    python kafka_pipeline/producer.py

URL of Web Application: 
https://titanic1912-survival-prediction.streamlit.app/

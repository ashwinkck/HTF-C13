import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

def retrain_model(feedback_path):
    feedback_df = pd.read_csv(feedback_path)

    # Simple example — adjust according to your schema
    X = feedback_df.drop("assigned_employee_id", axis=1)
    y = feedback_df["assigned_employee_id"]

    model = RandomForestClassifier()
    model.fit(X, y)

    joblib.dump(model, "model.pkl")
    print("Model retrained and saved as model.pkl")

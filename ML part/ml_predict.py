import pandas as pd
import joblib
from ml_utils import preprocess_dataframe

# Load saved model
model = joblib.load("trained_model.pkl")

# Load new raw input data
df = pd.read_csv(r"C:\Users\ashwi\OneDrive\Desktop\Sycamore\Hackathons\HackToFuture sjec\ML part\sample_scheduling_dataset.csv")

# Preprocess the new data
X_new = preprocess_dataframe(df)

# Predict
y_pred = model.predict(X_new)

# Save predictions
df['predicted_employee'] = y_pred
df.to_csv("predicted_assignments.csv", index=False)
print("✅ Predictions saved to 'predicted_assignments.csv'")

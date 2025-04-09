import pandas as pd
import numpy as np
import json
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer

# Define the same MultiHotEncoder class that was used in training
class MultiHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = MultiLabelBinarizer()
        self.column_name = None

    def fit(self, X, y=None):
        self.column_name = X.columns[0]
        self.encoder.fit(X[self.column_name].str.split(','))
        return self

    def transform(self, X):
        col = self.column_name
        values = X[col].str.split(',').fillna('')
        transformed = self.encoder.transform(values)
        new_cols = [f"{col}__{c}" for c in self.encoder.classes_]
        return pd.DataFrame(transformed, columns=new_cols, index=X.index)

# Now load the model (after defining the MultiHotEncoder class)
model = joblib.load('smart_scheduler_model.pkl')

# Load your test data from CSV
test_data = pd.read_csv(r"C:\Users\ashwi\OneDrive\Desktop\Sycamore\Hackathons\HackToFuture sjec\ML part\sample_scheduling_dataset.csv")

# Preprocess the test data
def preprocess_test_data(test_df):
    test_df = test_df.copy()
    list_columns = ['employee_skills', 'employee_availability', 'task_required_skills']
    for col in list_columns:
        if col in test_df.columns:
            def to_string_list(x):
                if isinstance(x, list):
                    return ','.join(map(str, x))
                elif isinstance(x, str):
                    try:
                        # Attempt to parse stringified list
                        parsed = json.loads(x.replace("'", '"'))
                        if isinstance(parsed, list):
                            return ','.join(map(str, parsed))
                    except:
                        pass
                return ''
            test_df[col] = test_df[col].apply(to_string_list)
    return test_df

# Preprocess the test data
processed_test_data = preprocess_test_data(test_data)

# Make predictions
predictions = model.predict(processed_test_data)

# Add predictions to the test data
test_data['predicted_assignment_valid'] = predictions

# Save results
test_data.to_csv('test_results.csv', index=False)

print("✅ Predictions completed and saved to 'test_results.csv'")
# model.py
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report
from preprocessing import preprocess_dataframe  # Import the preprocessing function

TARGET_COLUMN = 'assignment_valid'

def load_model():
    """Load the trained model from file."""
    return joblib.load(r"C:\Users\ashwi\OneDrive\Desktop\Sycamore\Hackathons\HackToFuture sjec\ML part\smart_scheduler_model.pkl")

def build_and_train_model(df_clean, label_column=TARGET_COLUMN):
    """Build and train the model."""
    X = df_clean.drop(columns=[label_column])
    y = df_clean[label_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    categorical_cols = ['employee_id', 'task_id', 'rule_violated', 'task_priority']
    numeric_cols = ['task_duration_days', 'task_start_day']
    list_cols = ['employee_skills', 'employee_availability', 'task_required_skills']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('skills', MultiHotEncoder(), ['employee_skills']),
            ('availability', MultiHotEncoder(), ['employee_availability']),
            ('required_skills', MultiHotEncoder(), ['task_required_skills'])
        ]
    )

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    pipe.fit(X_train, y_train)

    # Evaluate the model
    y_pred = pipe.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save the trained model
    joblib.dump(pipe, 'smart_scheduler_model.pkl')
    return pipe

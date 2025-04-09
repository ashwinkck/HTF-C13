# feedback_system.py
import pandas as pd
import numpy as np
import json
import joblib
import os
from datetime import datetime
from preprocessing import preprocess_dataframe, MultiHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report

# Constants
FEEDBACK_FILE = 'feedback_history.csv'
METADATA_FILE = 'model_metadata.json'
MODEL_FILE = r"C:\Users\ashwi\OneDrive\Desktop\Sycamore\Hackathons\HackToFuture sjec\ML part\smart_scheduler_model.pkl"
TARGET_COLUMN = 'assignment_valid'

def load_model():
    """Load the trained model from file."""
    return joblib.load(MODEL_FILE)

def predict_assignments(model, data):
    """Make predictions using the model."""
    processed_data = preprocess_dataframe(data)
    predictions = model.predict(processed_data)
    
    # Add predictions to the data
    data['predicted_assignment_valid'] = predictions
    
    # Add confidence scores if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(processed_data)
        data['confidence_score'] = [prob[1] for prob in probabilities]
    
    return data

def record_feedback(assignment_id, employee_id, task_id, user_rating, feedback_text=""):
    """Record user feedback on employee-task assignments."""
    try:
        # Load existing feedback or create new dataframe
        if os.path.exists(FEEDBACK_FILE):
            feedback_df = pd.read_csv(FEEDBACK_FILE)
        else:
            feedback_df = pd.DataFrame(columns=[
                'assignment_id', 'employee_id', 'task_id', 
                'user_rating', 'feedback_text', 'timestamp'
            ])
        
        # Add new feedback
        new_feedback = pd.DataFrame({
            'assignment_id': [assignment_id],
            'employee_id': [employee_id],
            'task_id': [task_id],
            'user_rating': [user_rating],
            'feedback_text': [feedback_text],
            'timestamp': [datetime.now().isoformat()]
        })
        
        feedback_df = pd.concat([feedback_df, new_feedback], ignore_index=True)
        
        # Save updated feedback
        feedback_df.to_csv(FEEDBACK_FILE, index=False)
        
        print(f"Feedback recorded for assignment {assignment_id}")
        return True
    except Exception as e:
        print(f"Error recording feedback: {e}")
        return False

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
    joblib.dump(pipe, MODEL_FILE)
    return pipe

def retrain_model_with_feedback(original_training_data_path, frequency='weekly'):
    """Retrain the model incorporating user feedback."""
    try:
        # Check if it's time to retrain based on frequency
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r') as f:
                metadata = json.load(f)
                last_train_time = datetime.fromisoformat(metadata.get('last_trained', '2000-01-01T00:00:00'))
        else:
            last_train_time = datetime.fromisoformat('2000-01-01T00:00:00')
        
        now = datetime.now()
        should_retrain = False
        
        if frequency == 'daily':
            should_retrain = (now - last_train_time).days >= 1
        elif frequency == 'weekly':
            should_retrain = (now - last_train_time).days >= 7
        elif frequency == 'monthly':
            should_retrain = (now - last_train_time).days >= 30
        elif frequency == 'on_demand':
            should_retrain = True
            
        if not should_retrain:
            print("Skipping retraining based on frequency setting.")
            return False
            
        # Load original training data
        original_df = pd.read_json(original_training_data_path)
        
        # Load feedback data
        if not os.path.exists(FEEDBACK_FILE):
            print("No feedback data available for retraining.")
            return False
            
        feedback_df = pd.read_csv(FEEDBACK_FILE)
        
        # Extract training examples from feedback
        feedback_training = pd.DataFrame()
        
        for _, feedback in feedback_df.iterrows():
            # Find the original data point or similar examples
            matching_rows = original_df[
                (original_df['employee_id'] == feedback['employee_id']) & 
                (original_df['task_id'] == feedback['task_id'])
            ]
            
            if not matching_rows.empty:
                new_row = matching_rows.iloc[0].copy()
                # Override with user feedback
                new_row['assignment_valid'] = feedback['user_rating']
                feedback_training = pd.concat([feedback_training, pd.DataFrame([new_row])], ignore_index=True)
            
        # Combine original data with feedback data
        combined_df = pd.concat([original_df, feedback_training], ignore_index=True)
        combined_df = combined_df.drop_duplicates(
            subset=['employee_id', 'task_id'], 
            keep='last'  # Keep the feedback version if duplicate
        )
        
        # Clean and preprocess the combined data
        combined_clean = preprocess_dataframe(combined_df)
        
        # Retrain the model
        model = build_and_train_model(combined_clean, label_column='assignment_valid')
        
        # Save metadata
        with open(METADATA_FILE, 'w') as f:
            json.dump({
                'last_trained': now.isoformat(),
                'feedback_count': len(feedback_df),
                'total_training_examples': len(combined_clean)
            }, f)
            
        print(f"Model retrained with {len(feedback_df)} feedback entries")
        return True
        
    except Exception as e:
        print(f"Error retraining model: {e}")
        return False
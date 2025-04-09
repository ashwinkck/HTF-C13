# test.py
import pandas as pd
from model import load_model, predict_assignments
from feedback import record_feedback, retrain_model_with_feedback

# Test predictions
def test_predictions():
    test_data = pd.DataFrame({
        'employee_id': ['emp001', 'emp002', 'emp003'],
        'employee_skills': ['python,data,ml', 'design,ui,ux', 'java,backend'],
        'employee_availability': ['mon,tue,wed', 'wed,thu,fri', 'mon,tue,fri'],
        'task_id': ['task001', 'task002', 'task003'],
        'task_required_skills': ['python,ml', 'design,ui', 'java,api'],
        'task_priority': ['high', 'medium', 'low'],
        'task_duration_days': [3, 2, 4],
        'task_start_day': [1, 3, 1],
        'rule_violated': [False, False, False]
    })
    
    model = load_model()
    predictions = predict_assignments(model, test_data)
    print(predictions[['employee_id', 'task_id', 'predicted_assignment_valid']])

# Test feedback collection and retraining
def test_feedback_and_retraining():
    record_feedback('assign001', 'emp001', 'task001', 1, "Completed task ahead of schedule")
    retrain_model_with_feedback('scheduling_dataset.json')
    print("Model retrained with feedback!")

# Run tests
test_predictions()
test_feedback_and_retraining()

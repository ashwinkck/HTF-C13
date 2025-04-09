# test_feedback.py
import pandas as pd
from preprocessing import MultiHotEncoder
from feedback_system import load_model, predict_assignments, record_feedback, retrain_model_with_feedback

def test_feedback_system():
    print("=== Testing Feedback System ===")
    
    # 1. Load model
    print("Loading model...")
    try:
        model = load_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # 2. Create test assignments
    print("Creating test assignments...")
    test_assignments = pd.DataFrame({
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
    
    # 3. Get predictions
    print("Getting predictions...")
    results = predict_assignments(model, test_assignments)
    print("\nOriginal predictions:")
    print(results[['employee_id', 'task_id', 'predicted_assignment_valid']])
    
    # 4. Simulate user feedback
    print("\nRecording feedback...")
    record_feedback(
        assignment_id='assign001',
        employee_id='emp001',
        task_id='task001',
        user_rating=1,  # Good match
        feedback_text="Completed task ahead of schedule"
    )
    
    record_feedback(
        assignment_id='assign002',
        employee_id='emp002',
        task_id='task002',
        user_rating=0,  # Poor match
        feedback_text="Employee struggled with this task"
    )
    
    # 5. Retrain model with feedback
    print("\nRetraining model...")
    retrain_result = retrain_model_with_feedback(
        original_training_data_path=r"C:\Users\ashwi\OneDrive\Desktop\Sycamore\Hackathons\HackToFuture sjec\Google or tools\dataset\testdataformodel.json",
        frequency='on_demand'
    )
    
    if retrain_result:
        # 6. Load updated model and test
        print("\nTesting updated model...")
        updated_model = load_model()
        new_results = predict_assignments(updated_model, test_assignments)
        
        print("\nUpdated predictions:")
        print(new_results[['employee_id', 'task_id', 'predicted_assignment_valid']])
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_feedback_system()
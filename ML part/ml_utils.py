import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

def preprocess_dataframe(df):
    df = df.copy()

    # Encode list columns
    mlb_skills = MultiLabelBinarizer()
    mlb_availability = MultiLabelBinarizer()
    mlb_task_skills = MultiLabelBinarizer()

    skills_encoded = mlb_skills.fit_transform(df['employee_skills'])
    availability_encoded = mlb_availability.fit_transform(df['employee_availability'])
    task_skills_encoded = mlb_task_skills.fit_transform(df['task_required_skills'])

    skills_df = pd.DataFrame(skills_encoded, columns=[f"skill_{s}" for s in mlb_skills.classes_])
    availability_df = pd.DataFrame(availability_encoded, columns=[f"avail_day_{d}" for d in mlb_availability.classes_])
    task_skills_df = pd.DataFrame(task_skills_encoded, columns=[f"task_req_{s}" for s in mlb_task_skills.classes_])

    # Add engineered features
    df['task_duration_days'] = 1  # Placeholder
    df['task_start_day'] = df['employee_availability'].apply(lambda x: x[0].lower() if isinstance(x, list) and len(x) > 0 else 'unknown')

    # One-hot encode task priority
    priority_dummies = pd.get_dummies(df['task_priority'], prefix='task_priority')
    df = pd.concat([df, priority_dummies], axis=1)

    # Drop original categorical columns
    df = df.drop(columns=[
        'employee_skills',
        'employee_availability',
        'task_required_skills',
        'employee_id',
        'task_id',
        'task_priority'
    ])

    # Merge encoded features
    df = pd.concat([df.reset_index(drop=True), skills_df, availability_df, task_skills_df], axis=1)

    return df

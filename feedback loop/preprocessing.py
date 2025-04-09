# preprocessing.py
import pandas as pd
import numpy as np
import json
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer

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

def preprocess_dataframe(df):
    """Preprocess the dataframe for model training or prediction."""
    df = df.copy()
    list_columns = ['employee_skills', 'employee_availability', 'task_required_skills']
    for col in list_columns:
        if col in df.columns:
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
            df[col] = df[col].apply(to_string_list)
    return df
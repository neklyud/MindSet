import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_model(name):
    model = joblib.load(name)
    return model

def preprocess(filepath, ohe, scaler, poly):
    data = pd.read_csv(filepath, sep=',', index_col=0)
    idx = data.index.values
    y_test = data['DATA_TYPE'] == 'TEST '
    categorial_features = ['POLICY_BRANCH', 'VEHICLE_MAKE', 'VEHICLE_MODEL', 'POLICY_INTERMEDIARY', 'INSURER_GENDER', 'POLICY_CLM_N', 
                           'POLICY_CLM_GLT_N', 'POLICY_PRV_CLM_N', 'POLICY_PRV_CLM_GLT_N', 'POLICY_YEARS_RENEWED_N', 'CLIENT_REGISTRATION_REGION']
    numeric_features = list(set(data.columns) - set(categorial_features) - {'DATA_TYPE'} - {'POLICY_IS_RENEWED'})
    print(len(categorial_features), len(numeric_features), len(data.columns))
    cat_transform_data = ohe.transform(data[categorial_features]).toarray()

    num_transform_data = scaler.transform(data[numeric_features])
    print(num_transform_data.shape)
    num_poly_data = poly.transform(data[numeric_features])
    transformed_data = np.concatenate((cat_transform_data, num_poly_data), axis=1)
    test_indices = data['DATA_TYPE'] == 'TEST '
    X_test = transformed_data[test_indices]
    return X_test, y_test, idx


def serialize(id, y_predicted):
    return pd.DataFrame(data=[id, y_predicted], columns=[['id', 'predict']])

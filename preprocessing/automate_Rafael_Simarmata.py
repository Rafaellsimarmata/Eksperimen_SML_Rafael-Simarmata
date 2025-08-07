import pandas as pd
import numpy as np

from joblib import dump, load

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import make_pipeline as make_imb_pipeline

def preprocess_data(save_path, file_path):
    data = pd.read_csv(file_path)
    target_column = "label"

    data.drop(columns=["transaction_id"], inplace = True)
    data.drop_duplicates(inplace = True)

    numeric_features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()
    column_names = data.columns

    column_names = data.columns.drop(target_column)
    # df_header = pd.DataFrame(columns=column_names)

    # df_header.to_csv(file_path, index=False)
    # print(f"Nama kolom berhasil disimpan ke: {file_path}")

    if target_column in numeric_features:
        numeric_features.remove(target_column)
    if target_column in categorical_features:
        categorical_features.remove(target_column)

    # Pipeline untuk fitur numerik
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    # Pipeline untuk fitur kategoris
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Column Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    X = data.drop(columns=[target_column])
    y = data[target_column]
 
    # Membagi data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Create pipeline with RandomOverSampler
    pipeline = make_imb_pipeline(
        preprocessor,
        RandomOverSampler(random_state=42)
    )

    X_train, y_train = pipeline.fit_resample(X_train, y_train)
    X_test = pipeline.named_steps['columntransformer'].transform(X_test)

    # Extract categorical feature names from OneHotEncoder
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
    cat_features = []
    for i, col in enumerate(categorical_features):
        cats = cat_encoder.categories_[i]
        cat_features.extend([f"{col}_{cat}" for cat in cats])

    dump(pipeline.named_steps['columntransformer'], save_path)

    # Save the cleaned data to a csv file
    preproc_data = pd.DataFrame(X_train, columns=numeric_features + cat_features)
    preproc_data['label'] = y_train.reset_index(drop=True)
    preproc_data.to_csv(save_path, index=False)
    
    print(f'Pre-processed data successfully saved to: {save_path}\n')

    return X_train, X_test, y_train, y_test

def inference(new_data, load_path):
    # Memuat pipeline preprocessing
    preprocessor = load(load_path)
    print(f"Pipeline preprocessing dimuat dari: {load_path}")
 
    # Transformasi data baru
    transformed_data = preprocessor.transform(new_data)
    return transformed_data

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data(
        file_path='./fraud_detection_raw/fraud_detection.csv',
        save_path='preprocessing/fraud_detection_processed.csv'
    )
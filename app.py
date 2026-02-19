from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import warnings
import json
import time
import os

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

MODEL_FEATURE_MAP = {
    'CNN': {
        'numeric': ['filters1', 'filters2', 'dense_units', 'dropout_rate', 'learning_rate', 'epochs'],
        'categorical': ['optimizer']
    },
    'LeNet-5': {
        'numeric': ['batch_size', 'epochs', 'learning_rate', 'dropout_rate'],
        'categorical': ['optimizer']
    },
    'MobileNetV2': {
        'numeric': [
            'batch_size',
            'epochs_frozen_training',
            'epochs_fine_tuning',
            'fine_tuning_layers',
            'learning_rate_fine_tuning'
        ],
        'categorical': []
    },
    'Gaussian Naive Bayes': {'numeric': ['var_smoothing'], 'categorical': []},
    'Logistic Regression': {'numeric': ['C'], 'categorical': ['penalty', 'solver']},
    'SVM': {'numeric': ['C', 'gamma'], 'categorical': ['kernel']},
    'XGBoost': {'numeric': ['n_estimators', 'learning_rate', 'max_depth'], 'categorical': []},
    'KNN': {'numeric': ['n_neighbors'], 'categorical': ['weights']},
    'MultinomialNB': {'numeric': ['alpha'], 'categorical': []}
}

COMMON_ENGINEERED_NUMERIC = ['log_Size']
COMMON_ENGINEERED_CATEGORICAL = ['Data_type']

def load_and_preprocess_data(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return None, f"Error: The file '{file_path}' was not found."
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    required = ['Data_type', 'Carbon_Emissions_kg', 'Accuracy', 'Size', 'Features', 'Model', 'Hyperparameters', 'Task']
    if not all(col in df.columns for col in required):
        return None, "Error: Missing required columns."
    for col in ['Accuracy', 'Carbon_Emissions_kg', 'Size']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=required)
    df = df[df['Carbon_Emissions_kg'] > 0]
    model_name_map = {
        'SVC': 'SVM', 'LinearSVC': 'SVM', 'XGBoost Classifier': 'XGBoost',
        'LogisticRegression': 'Logistic Regression', 'GaussianNB': 'Gaussian Naive Bayes'
    }
    df['Model'] = df['Model'].replace(model_name_map)
    df = df[df['Model'].isin(MODEL_FEATURE_MAP.keys())]
    if df.empty:
        return None, "Error: No valid data after cleaning."
    return df, None

def parse_and_engineer_features(df):
    df_parsed = df.copy()
    df_parsed['log_Size'] = np.log1p(df_parsed['Size'])
    all_params = []
    for row in df_parsed['Hyperparameters']:
        params = {}
        if isinstance(row, str):
            try:
                params = json.loads(row.replace("'", '"'))
            except json.JSONDecodeError:
                parts = row.strip('{}').split(',')
                for part in parts:
                    if '=' in part:
                        key, value = part.split('=', 1)
                        params[key.strip()] = value.strip()
        all_params.append(params)
    temp_df = pd.DataFrame(all_params, index=df_parsed.index)
    df_parsed = df_parsed.join(temp_df)
    all_numeric_hyperparams = []
    all_categorical_hyperparams = []
    for v in MODEL_FEATURE_MAP.values():
        all_numeric_hyperparams.extend(v['numeric'])
        all_categorical_hyperparams.extend(v['categorical'])
    all_numeric_hyperparams = list(set(all_numeric_hyperparams))
    all_categorical_hyperparams = list(set(all_categorical_hyperparams))
    for col in all_numeric_hyperparams:
        if col in df_parsed.columns:
            df_parsed[col] = pd.to_numeric(df_parsed[col], errors='coerce').fillna(0)
        else:
            df_parsed[col] = 0
    for col in all_categorical_hyperparams:
        if col in df_parsed.columns:
            df_parsed[col] = df_parsed[col].fillna('None')
        else:
            df_parsed[col] = 'None'
    return df_parsed

def normalize_column(series):
    if series.max() == series.min():
        return pd.Series(0.5, index=series.index)
    return (series - series.min()) / (series.max() - series.min())

def calculate_score(df, accuracy_weight=0.6, emission_weight=0.4):
    df['Normalized_Accuracy'] = normalize_column(df['Accuracy'])
    df['Normalized_Emission'] = 1 - normalize_column(df['Carbon_Emissions_kg'])
    df['Score'] = (accuracy_weight * df['Normalized_Accuracy'] + emission_weight * df['Normalized_Emission'])
    return df

def train_specialized_classical_models(df, feature_map):
    trained_pipelines = {}
    print("--- Training Specialized RandomForest Meta-Models ---")
    for model_name, features in feature_map.items():
        start_time = time.time()
        model_df = df[df['Model'] == model_name].copy()
        if model_df.empty or len(model_df) < 10:
            print(f"  - Insufficient data for {model_name}, skipping.")
            continue
        numeric_features = COMMON_ENGINEERED_NUMERIC + features['numeric']
        categorical_features = COMMON_ENGINEERED_CATEGORICAL + features['categorical']
        all_features = numeric_features + categorical_features
        X = model_df[all_features]
        y_accuracy = model_df['Accuracy']
        y_emission = model_df['Carbon_Emissions_kg']
        preprocessor = ColumnTransformer(
            transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)],
            remainder=MinMaxScaler()
        )
        pipeline_accuracy = Pipeline([('preprocessor', preprocessor), ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])
        pipeline_emission = Pipeline([('preprocessor', preprocessor), ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])
        try:
            accuracy_r2_scores = cross_val_score(pipeline_accuracy, X, y_accuracy, cv=5, scoring='r2')
            accuracy_r2 = accuracy_r2_scores.mean()
            print(f"  - {model_name} Accuracy R² (CV): {accuracy_r2:.4f} (CV scores: {accuracy_r2_scores})")
            emission_r2_scores = cross_val_score(pipeline_emission, X, y_emission, cv=5, scoring='r2')
            emission_r2 = emission_r2_scores.mean()
            print(f"  - {model_name} Emission R² (CV): {emission_r2:.4f} (CV scores: {emission_r2_scores})")
        except Exception as e:
            print(f"  - Cross-validation failed for {model_name}: {e}")
            accuracy_r2 = -999
            emission_r2 = -999
        pipeline_accuracy.fit(X, y_accuracy)
        pipeline_emission.fit(X, y_emission)
        trained_pipelines[model_name] = {
            'accuracy_pipe': pipeline_accuracy,
            'emission_pipe': pipeline_emission,
            'features': all_features,
            'accuracy_r2': accuracy_r2,
            'emission_r2': emission_r2
        }
        end_time = time.time()
    return trained_pipelines

def predict_with_specialized_models(df, data_type, dataset_size, trained_pipelines):
    all_predictions = []
    log_dataset_size = np.log1p(dataset_size)
    for model_name, pipe_info in trained_pipelines.items():
        model_df = df[df['Model'] == model_name]
        pred_configs = model_df[pipe_info['features'] + ['Hyperparameters']].drop_duplicates()
        if pred_configs.empty:
            continue
        pred_data = pred_configs.copy()
        pred_data['log_Size'] = log_dataset_size
        pred_data['Data_type'] = data_type
        pred_data['Accuracy'] = pipe_info['accuracy_pipe'].predict(pred_data[pipe_info['features']])
        pred_data['Carbon_Emissions_kg'] = pipe_info['emission_pipe'].predict(pred_data[pipe_info['features']])
        pred_data['Model'] = model_name
        avg_r2 = (pipe_info.get('accuracy_r2', 0) + pipe_info.get('emission_r2', 0)) / 2
        if avg_r2 < 0:
            pred_data['Confidence'] = 'Low'
        elif avg_r2 < 0.4:
            pred_data['Confidence'] = 'Medium'
        else:
            pred_data['Confidence'] = 'High'
        all_predictions.append(pred_data)
    if not all_predictions:
        return None, "Could not generate any predictions."
    final_predictions_df = pd.concat(all_predictions, ignore_index=True)
    final_predictions_df['Carbon_Emissions_kg'] = final_predictions_df['Carbon_Emissions_kg'].clip(lower=0)
    return final_predictions_df, None

# --- Flask Application Setup ---
app = Flask(__name__, static_url_path='', static_folder='static')

try:
    file_path = 'Emission_Dataset.csv'
    df_raw, error = load_and_preprocess_data(file_path)
    if error:
        print(error)
        exit()
    df_featured = parse_and_engineer_features(df_raw)
except Exception as e:
    print(f"Failed to pre-train models: {e}")
    df_featured = None

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def send_static_file_with_path(path):
    return send_from_directory(app.static_folder, path)

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    try:
        user_input = request.json
        if not user_input:
            return jsonify({'error': 'Invalid JSON input'}), 400
        user_data_type = user_input.get('dataType')
        dataset_size = user_input.get('numSamples')
        priority = user_input.get('priority')
        user_features = user_input.get('numFeatures')
        user_task = user_input.get('mlTask')

        # Only allow 'Classification'
        allowed_tasks = ['Image Classification', 'Text Classification', 'Tabular Classification']
        if user_task not in allowed_tasks:
            return jsonify({'error': 'Model not yet trained for the specified task.'}), 404

        if user_data_type == 'Image':
            df_filtered = df_featured[
                (df_featured['Data_type'] == user_data_type) &
                (df_featured['Features'] == user_features) &
                (df_featured['Task'] == user_task)
            ]
            applicable_models = sorted(df_filtered['Model'].unique())
            applicable_feature_map = {k: v for k, v in MODEL_FEATURE_MAP.items() if k in applicable_models}
        else:
            df_filtered = df_featured[
                (df_featured['Data_type'] == user_data_type) &
                (df_featured['Task'] == user_task)
            ]
            applicable_models = sorted(df_filtered['Model'].unique())
            applicable_feature_map = {k: v for k, v in MODEL_FEATURE_MAP.items() if k in applicable_models}

        if not applicable_feature_map:
            return jsonify({'error': 'Model not yet trained for the specified data type/features.'}), 404

        trained_pipelines = train_specialized_classical_models(df_filtered, applicable_feature_map)
        predictions_df, error = predict_with_specialized_models(df_filtered, user_data_type, dataset_size, trained_pipelines)
        if error:
            return jsonify({'error': error}), 500

        if priority == 'balanced':
            scored_df = calculate_score(predictions_df, accuracy_weight=0.5, emission_weight=0.5)
        elif priority.startswith('accuracy'):
            weight = int(priority.split('_')[1]) / 100
            scored_df = calculate_score(predictions_df, accuracy_weight=weight, emission_weight=1-weight)
        elif priority.startswith('carbon'):
            weight = int(priority.split('_')[1]) / 100
            scored_df = calculate_score(predictions_df, accuracy_weight=1-weight, emission_weight=weight)
        else:
            return jsonify({'error': 'Invalid priority setting'}), 400
        best_per_model = scored_df.loc[scored_df.groupby('Model')['Score'].idxmax()]
        sorted_recommendations = best_per_model.sort_values(by='Score', ascending=False)
        results = []
        for index, row in sorted_recommendations.iterrows():
            results.append({
                'rank': len(results) + 1,
                'model': row['Model'],
                'config': row['Hyperparameters'],
                'accuracy': round(row['Accuracy'], 2),
                'carbon': round(row['Carbon_Emissions_kg'], 6),
                'score': round(row['Score'], 4),
                'confidence': row.get('Confidence', 'N/A')
            })
        return jsonify({'recommendations': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
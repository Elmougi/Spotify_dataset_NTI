import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, PowerTransformer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Streamlit interface
st.title("Spotify Track Popularity Predictor")
st.write("This app trains a deep learning model to predict track popularity based on Spotify features.")

# Load and preprocess data
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv('https://raw.githubusercontent.com/Elmougi/Spotify_dataset_NTI/refs/heads/main/Spotify-dataset.csv', low_memory=False)
    
    # Basic cleaning
    numerical_cols = ['popularity', 'duration_ms', 'danceability', 'energy', 'loudness', 'speechiness',
                     'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'key', 'mode', 'time_signature']
    for col in numerical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())
    
    categorical_cols = ['track_genre', 'explicit', 'artists', 'album_name']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    df.dropna(inplace=True)
    df.drop_duplicates(subset=['track_id', 'track_name', 'artists'], keep='first', inplace=True)
    
    # Simplified feature engineering
    df['key_sin'] = np.sin(2 * np.pi * df['key']/12)
    df['key_cos'] = np.cos(2 * np.pi * df['key']/12)
    df['duration_min'] = df['duration_ms'] / 60_000
    df['duration_log'] = np.log1p(df['duration_min'])
    pt = PowerTransformer(method='yeo-johnson')
    df['loudness_yeo'] = pt.fit_transform(df[['loudness']])
    df['is_instrumental'] = (df['instrumentalness'] > 0.8).astype('int')
    df['happy_dance'] = df['valence'] * df['danceability']
    
    le = LabelEncoder()
    df['track_genre'] = le.fit_transform(df['track_genre'])
    df['explicit'] = df['explicit'].astype(int)
    
    return df

df = load_and_preprocess_data()

# Define features and target
features = ['track_genre', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
            'instrumentalness', 'liveness', 'valence', 'tempo', 'key_sin', 'key_cos',
            'duration_log', 'loudness_yeo', 'explicit', 'is_instrumental', 'happy_dance']
target = 'popularity'

# Prepare data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
@st.cache_resource
def train_model():
    model = Sequential([
        Dense(64, input_shape=(X_train_scaled.shape[1],), kernel_regularizer=l2(0.001)),
        LeakyReLU(alpha=0.01),
        Dropout(0.2),
        Dense(32, kernel_regularizer=l2(0.001)),
        LeakyReLU(alpha=0.01),
        Dense(1)  # Output layer for regression
    ])
    model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test),
              epochs=50, batch_size=32, callbacks=[early_stopping], verbose=0)
    return model

model = train_model()

# Evaluate model
y_pred = model.predict(X_test_scaled).flatten()
test_mse = mean_squared_error(y_test, y_pred)
test_mae = mean_absolute_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)
st.write(f"Model Performance: MSE: {test_mse:.2f}, MAE: {test_mae:.2f}, R²: {test_r2:.2f}")

# User input for prediction
st.subheader("Enter Track Features for Prediction")
input_data = {}
for feature in features:
    if feature in ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']:
        input_data[feature] = st.slider(f"{feature.capitalize()}", 0.0, 1.0, 0.5)
    elif feature == 'loudness':
        input_data[feature] = st.slider("Loudness (dB)", -60.0, 0.0, -10.0)
    elif feature == 'tempo':
        input_data[feature] = st.slider("Tempo (BPM)", 0.0, 250.0, 120.0)
    elif feature == 'track_genre':
        input_data[feature] = st.selectbox("Track Genre", df['track_genre'].unique())
    elif feature == 'explicit':
        input_data[feature] = st.selectbox("Explicit", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    elif feature == 'key_sin':
        key = st.slider("Key", 0, 11, 0)
        input_data['key_sin'] = np.sin(2 * np.pi * key / 12)
        input_data['key_cos'] = np.cos(2 * np.pi * key / 12)
    elif feature == 'duration_log':
        duration_min = st.slider("Duration (minutes)", 0.0, 10.0, 3.0)
        input_data['duration_log'] = np.log1p(duration_min)
    elif feature == 'loudness_yeo':
        input_data['loudness_yeo'] = pt.transform([[input_data['loudness']]])[0][0]
    elif feature == 'is_instrumental':
        input_data[feature] = 1 if input_data['instrumentalness'] > 0.8 else 0
    elif feature == 'happy_dance':
        input_data[feature] = input_data['valence'] * input_data['danceability']

# Convert input to DataFrame
input_df = pd.DataFrame([input_data])

# Preprocess input
input_df['track_genre'] = le.transform(input_df['track_genre'])
input_scaled = scaler.transform(input_df[features])

# Predict
if st.button("Predict Popularity"):
    prediction = model.predict(input_scaled).flatten()[0]
    st.write(f"Predicted Popularity: {prediction:.2f}")
# base
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Настройка для корректной настройки pipeline
import sklearn
sklearn.set_config(transform_output="pandas")

# Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Preprocessing
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler, OrdinalEncoder, TargetEncoder
from sklearn.model_selection import GridSearchCV, KFold

# for model learning
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import pickle

#models
from catboost import CatBoostRegressor
from catboost import CatBoostClassifier, Pool

#streamlit
import streamlit as st

## Шаг 1. Препроцессинг - предварительная обработка данных

def preprocess_data(data):
    data['Sex'] = data['Sex'].map({'M': 0, 'F': 1})
    data['ExerciseAngina'] = data['ExerciseAngina'].map({'N': 0, 'Y': 1})
    data['RestingECG'] = data['RestingECG'].map({'Normal': 1, 'LVH': 2, 'ST': 0})
    data['ST_Slope'] = data['ST_Slope'].map({'Up': 1, 'Flat': 0, 'Down': -1})

    # Кодирование ChestPainType
    chest_pain_dummies = pd.get_dummies(data['ChestPainType'], prefix='ChestPainType')
    data = pd.concat([data, chest_pain_dummies], axis=1)
    data.drop('ChestPainType', axis=1, inplace=True)
    
    # Проверка на пропущенные значения
    data.fillna(data.median(), inplace=True)
    
    return data, chest_pain_dummies


## Шаг 2. Импорт и подготовка данных для обучения модели

df = pd.read_csv('/Users/maricolada/Downloads/elbrus/ds-phase-1/06-supervised/aux/heart.csv') 
df, chest_pain_dummies = preprocess_data(df)

# разделим данные на features и target

X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


## Шаг 3. Обучение модели

cat_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    cat_features=['Sex', 'RestingECG', 'ExerciseAngina', 'ST_Slope'] + list(chest_pain_dummies.columns),
    verbose=100,
    early_stopping_rounds=50
)

cat_model.fit(X_train, y_train, eval_set=(X_test, y_test))


## Шаг 4. Оценка работы модели

y_pred = cat_model.predict(X_test)
model_accuracy = accuracy_score(y_test, y_pred)

# Сохранение модели
with open('catboost_model.pkl', 'wb') as file:
    pickle.dump(cat_model, file)


## Шаг 5. Создание интерфейса Streamlit

st.title('Heart Disease Detector')
st.write('Here you can get a heart disease risk assessment based on your medical data. Below you need to fill in the fields with the required data.')

# Ввод данных пользователем с примечаниями

age = st.number_input(
    "Your age",
    min_value=0,
    help="Enter your current age."
)

sex = st.selectbox(
    "Choose your gender",
    options=["M", "F"],
    help="Select male (M) or female (F) gender."
)

chest_pain_type = st.selectbox(
    "Chest pain type",
    options=["TA", "ATA", "NAP", "ASY"],
    help="TA: Typical angina, ATA: Atypical angina, "
         "NAP: No angina, ASY: Painless angina."
)

resting_bp = st.number_input(
    "Resting BP Results",
    min_value=0,
    help="Blood pressure at rest."
)

cholesterol = st.number_input(
    "Cholesterol level",
    min_value=0,
    help="Enter your cholesterol level in milligrams per deciliter."
)

fasting_bs = st.selectbox(
    "Blood sugar level > 120 mg/dL?",
    options=["0", "1"],
    help="Select 'Yes' (1) if your blood sugar is above 120 mg/dL, otherwise 'No' (0)."
)

resting_ecg = st.selectbox(
    "Resting ECG Results",
    options=["ST", "Normal", "LVH"],
    help="ST: Pathology, Normal: Normal, LVH: Left chamber enlargement."
)

max_hr = st.number_input(
    "Maximum heart rate",
    min_value=0,
    help="Enter your maximum heart rate."
)

exercise_angina = st.selectbox(
    "Exercise Angina",
    options=["Yes", "No"],
    help="Select 'Yes' if you have exercise angina, otherwise 'No'."
)

oldpeak = st.number_input(
    "Oldpeak",
    format="%.2f",
    help="Enter the old peak value in millimeters."
)

st_slope = st.selectbox(
    "Slope ST",
    options=["Up", "Flat", "Down"],
    help="Up: Increase, Flat: Flat, Down: Decrease."
)

# Кнопка для определения риска

if st.button("Detect risk"):

    # создание датасета с пользовательскими данными
    input_data = {
        "Age": age,
        "Sex": sex,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": int(fasting_bs),
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope,
        "ChestPainType_TA": 1 if chest_pain_type == "TA" else 0,
        "ChestPainType_ATA": 1 if chest_pain_type == "ATA" else 0,
        "ChestPainType_NAP": 1 if chest_pain_type == "NAP" else 0,
        "ChestPainType_ASY": 1 if chest_pain_type == "ASY" else 0
    }

    input_df = pd.DataFrame([input_data])

    cat_features = ['Sex', 'RestingECG', 'ExerciseAngina', 'ST_Slope', 
                    'ChestPainType_TA', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_ASY']

    # Pool для предсказания
    pool = Pool(data=input_df, cat_features=cat_features)

    # Предсказание с использованием обученной модели
    prediction = cat_model.predict(pool)

    # Вывод результата
    result = "Yes" if prediction[0] == 1 else "No"
    st.write(f"Heart Disease Risk: {result}")
    # Вывод точности модели
    st.write(f"Model accuracy: {model_accuracy * 100:.2f}%. Risk detection is not 100% accurate. We recommend confirming your test results with your cardiologist.")





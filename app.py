# 서울 공공자전거 시간별 대여량 예측 - Streamlit 시연 앱
# 실행: streamlit run app.py
# 노트북(seoul_bike_project.ipynb)과 같은 전처리·같은 XGBoost 설정을 사용합니다.

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

RANDOM_STATE = 42

# app.py가 있는 폴더 기준으로 경로를 잡아야 클라우드에서도 파일을 찾을 수 있음
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'dataset', 'SeoulBikeData.csv')
# 노트북 4-7에서 joblib으로 저장한 XGBoost 모델
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'xgb_model.pkl')

FEATURES = ['temperature', 'humidity', 'wind_speed', 'visibility',
            'dew_point_temperature', 'solar_radiation', 'rainfall', 'snowfall',
            'month', 'weekday', 'is_weekend', 'is_holiday',
            'seasons_Spring', 'seasons_Summer', 'seasons_Winter',
            'hour_sin', 'hour_cos']


def add_hour_features(df):
    # 시계 바늘 위치처럼 하루(24시간)를 한 바퀴로 보고 좌표로 변환 (23시와 0시가 가깝도록)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    return df


# @st.cache_data: 화면을 조작할 때마다 다시 읽지 않도록 결과를 기억해 둠
@st.cache_data
def load_data():
    # 기온 단위 기호(°) 때문에 utf-8로 안 읽히는 경우가 있어서 순서대로 시도
    for enc in ['utf-8', 'cp1252', 'cp949']:
        try:
            raw = pd.read_csv(DATA_PATH, encoding=enc)
            break
        except UnicodeDecodeError:
            continue

    raw.columns = ['date', 'rented_bike_count', 'hour', 'temperature', 'humidity',
                   'wind_speed', 'visibility', 'dew_point_temperature',
                   'solar_radiation', 'rainfall', 'snowfall', 'seasons',
                   'holiday', 'functioning_day']
    raw['date'] = pd.to_datetime(raw['date'], format='%d/%m/%Y')

    # 비운영일의 0은 '수요 없음'이 아니라 '기록 없음'이라 학습에서 제외
    data = raw[raw['functioning_day'] == 'Yes'].copy()
    data['month'] = data['date'].dt.month
    data['weekday'] = data['date'].dt.day_of_week          # 월=0 ... 일=6
    data['is_weekend'] = (data['weekday'] >= 5).astype(int)
    data['is_holiday'] = (data['holiday'] == 'Holiday').astype(int)
    data = pd.get_dummies(data, columns=['seasons'], drop_first=True, dtype=int)
    data = add_hour_features(data)
    return data


# @st.cache_resource: 모델 파일은 처음 한 번만 불러오고 계속 재사용
# 앱에서는 학습하지 않고 불러오기만 함 → 빠르게 뜨고 서버 CPU도 적게 씀
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)

    # 노트북 4단계와 같은 방식으로 나눈 테스트 데이터로 성능을 다시 계산해 표시
    data = load_data()
    _, test_df = train_test_split(data, test_size=0.2, random_state=RANDOM_STATE)
    pred = model.predict(test_df[FEATURES])
    y_test = test_df['rented_bike_count']
    scores = {
        'R²': r2_score(y_test, pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, pred)),
        'MAE': mean_absolute_error(y_test, pred),
    }
    return model, scores


def month_to_season(month):
    # 데이터의 계절 구분: 12~2월 겨울, 3~5월 봄, 6~8월 여름, 9~11월 가을
    if month in [12, 1, 2]:
        return 'Winter'
    if month in [3, 4, 5]:
        return 'Spring'
    if month in [6, 7, 8]:
        return 'Summer'
    return 'Autumn'


def make_input(date, hour, is_holiday, weather):
    # 화면에서 받은 값을 학습 때와 똑같은 컬럼 형태로 만듦
    season = month_to_season(date.month)
    row = {
        'hour': hour,
        **weather,
        'month': date.month,
        'weekday': date.weekday(),
        'is_weekend': int(date.weekday() >= 5),
        'is_holiday': int(is_holiday),
        'seasons_Spring': int(season == 'Spring'),
        'seasons_Summer': int(season == 'Summer'),
        'seasons_Winter': int(season == 'Winter'),
    }
    df = add_hour_features(pd.DataFrame([row]))
    return df[FEATURES]


# ---------------- 화면 구성 ----------------
st.set_page_config(page_title='서울 공공자전거 대여량 예측', layout='wide')
st.title('서울 공공자전거 시간별 대여량 예측')
st.caption('데이터: 서울 공공자전거(Seoul Bike Sharing System) 데이터, 2017-12-01 ~ 2018-11-30 / 모델: XGBoost')

for path in [DATA_PATH, MODEL_PATH]:
    if not os.path.exists(path):
        st.error(f'파일이 없습니다: {path}')
        st.stop()

data = load_data()
model, scores = load_model()

# 모델 성능 (노트북 4단계 무작위 분할과 같은 방식으로 계산한 테스트 결과)
c1, c2, c3 = st.columns(3)
c1.metric('테스트 R²', f"{scores['R²']:.3f}")
c2.metric('테스트 RMSE', f"{scores['RMSE']:.1f}대")
c3.metric('테스트 MAE', f"{scores['MAE']:.1f}대")

# 왼쪽 사이드바: 예측 조건 입력 (범위는 실제 데이터의 최솟값~최댓값)
st.sidebar.header('예측 조건 입력')
date = st.sidebar.date_input('날짜', value=pd.Timestamp('2018-06-15'))
hour = st.sidebar.slider('시간(시)', 0, 23, 18)
is_holiday = st.sidebar.checkbox('공휴일')


def num_input(label, col, default):
    return st.sidebar.slider(label, float(data[col].min()), float(data[col].max()), float(default))


weather = {
    'temperature': num_input('기온(°C)', 'temperature', 20.0),
    'humidity': num_input('습도(%)', 'humidity', 50.0),
    'wind_speed': num_input('풍속(m/s)', 'wind_speed', 1.5),
    'visibility': num_input('가시거리(10m)', 'visibility', 2000.0),
    'dew_point_temperature': num_input('이슬점 온도(°C)', 'dew_point_temperature', 9.0),
    'solar_radiation': num_input('일사량(MJ/m²)', 'solar_radiation', 0.5),
    'rainfall': num_input('강수량(mm)', 'rainfall', 0.0),
    'snowfall': num_input('적설량(cm)', 'snowfall', 0.0),
}

# 1) 입력한 한 시간의 예측값
pred = model.predict(make_input(date, hour, is_holiday, weather))[0]
pred = max(pred, 0)   # 대여량은 음수가 될 수 없으므로 0 아래는 0으로
st.subheader(f'{date} {hour}시 예상 대여량: 약 {pred:,.0f}대')
st.caption(f'계절: {month_to_season(date.month)} / 요일: {"월화수목금토일"[date.weekday()]}요일')

# 2) 같은 날씨가 하루 종일 이어진다고 가정한 24시간 예측
hours = list(range(24))
preds_24 = [max(model.predict(make_input(date, h, is_holiday, weather))[0], 0) for h in hours]

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(hours, preds_24, marker='o')
ax.axvline(hour, color='red', linestyle='--', alpha=0.5)
ax.set_xticks(hours)
ax.set_xlabel('Hour')
ax.set_ylabel('Predicted rentals')
ax.set_title('Predicted rentals by hour (same weather all day)')
ax.grid(alpha=0.3)
st.pyplot(fig)
st.caption('빨간 점선은 선택한 시간입니다. 실제로는 시간마다 날씨가 바뀌므로 참고용 그래프입니다.')

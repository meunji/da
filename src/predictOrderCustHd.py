import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
from xgboost import XGBClassifier


def predict(df_cluster, df_ord, df_cust):

    # selected_cluster = 2
    # selected_slitm = 2231282389

    cond_cluster = df_cluster['cluster'] == st.session_state.predict_cluster
    cond_ord = df_ord['CUST_NO'].isin(df_cluster[cond_cluster]['CUST_NO'])

    filtered_orders = df_ord[cond_ord]

    if filtered_orders.empty:
        st.warning("선택한 클러스터에 해당하는 주문이 없습니다.")
        return
    

    # 1.고객 세그먼트 -> 2.타겟 상품에 대한 예측 데이터 준비

    # 해당 상품 주문한(positive) 고객 label 1
    positive = filtered_orders[filtered_orders['SLITM_CD'] == st.session_state.predict_slitm][['CUST_NO']].drop_duplicates()
    positive['label'] = 1

    # 해당 상품 주문 안한(negative) 고객 label 0
    # all_customers = df_cust[['CUST_NO']]
    cluster_customers = df_cluster[df_cluster['cluster'] == st.session_state.predict_cluster][['CUST_NO']]
    labels = cluster_customers.merge(positive, on='CUST_NO', how='left')
    labels['label'] = labels['label'].fillna(0)

    # 3. 학습데이터 구성
    train_data = df_cluster.merge(labels, on='CUST_NO', how='inner')

    # 불필요한 열 제거
    drop_cols = ['AGE', 'first_order', 'last_order', 'label', 'cluster']
    X = train_data.drop(columns=drop_cols, errors='ignore')
    # X = train_data.drop(columns=['AGE','first_order','last_order','label'])
    y = train_data['label']

    # 4. 모델 학습 및 예측

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=0)

    # model = RandomForestClassifier()

    X_train['CUST_NO'] = X_train['CUST_NO'].astype(np.int64)
    X_test['CUST_NO'] = X_test['CUST_NO'].astype(np.int64)

    # model = XGBClassifier(
    #     use_label_encoder=False,  # 경고 방지용
    #     eval_metric='logloss',    # 경고 방지용
    #     scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1])  # 클래스 불균형 조정
    # )

    
    model = lgb.LGBMClassifier(
        class_weight='balanced',  # 클래스 불균형 자동 조정
        random_state=42,

        # 배포환경에서 속도개선, 메모리절약을 위한 파라미터 튜닝 
        n_estimators=40,         # 트리 개수 줄여서 학습/예측 속도 향상
        max_depth=4,             # 깊이 제한으로 모델 복잡도 & 메모리 절감
        num_leaves=15,           # max_depth에 맞춰 적게 설정 (2^5-1=31 이하)
        min_child_samples=30,    # 너무 작은 노드 생성을 줄여 트리 크기 축소
        subsample=0.8,           # 약간 샘플링해 속도 향상, 메모리 절약
        colsample_bytree=0.6,    # 피처 샘플링 비율 낮춰 계산량 감소
        max_bin=63,              # 기본 255보다 낮춰 메모리 사용 줄임
        n_jobs=-1,               # CPU 코어 최대한 활용해 속도 향상
        verbose=-1               # 학습로그 출력줄여서 오버헤드 감소
    )


    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    # st.write(classification_report(y_test, y_pred))

    # 문자열로 된 리포트를 딕셔너리로 변환
    report_dict = classification_report(y_test, y_pred, output_dict=True)

    # DataFrame으로 변환
    report_df = pd.DataFrame(report_dict).transpose()

    # Streamlit에서 보기 좋게 출력
    st.write("")
    st.subheader(f"📊 클러스터 {st.session_state.predict_cluster} 고객 {st.session_state.predict_slitm} 상품 주문 예측 성능 리포트")
    st.dataframe(report_df.style.format("{:.2f}"))

    st.markdown("""
    <span style='font-size: 15px; font-weight: 600;'>📘 지표 설명</span><br>

    - **precision (정밀도)**: 양성이라고 예측한 것 중 실제로 양성인 비율  
    - **recall (재현율)**: 실제 양성 중에서 모델이 정확히 예측한 비율  
    - **f1-score**: 정밀도와 재현율의 조화 평균 (균형 잡힌 평가 지표)  
    - **support**: 각 클래스(예: 0, 1)에 해당하는 실제 샘플 수  
    - **accuracy (정확도)**: 전체 예측 중 맞은 비율  
    <br>
    - **macro avg**: 클래스별 지표의 단순 평균 (클래스 균형이 안 맞을 때 유용)  
    - **weighted avg**: support 수를 가중치로 적용한 평균 (클래스 비율 반영)  
    """, unsafe_allow_html=True)

    st.write("")
    st.write("---")
    st.write("")

    results = X_test.copy()
    results['ACTUAL'] = y_test.values
    results['PREDICT'] = y_pred
    cond = results['PREDICT'] == 1

    results['CUST_NO'] = results['CUST_NO'].astype(str)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"✅ 주문 예측 결과 : {results['CUST_NO'].nunique()}")
        st.dataframe(results[['CUST_NO','ACTUAL','PREDICT']].reset_index(drop=True))
    with col2:
        st.subheader(f"✅ 주문할 것으로 예측된 고객 수 : {cond.sum()}")
        st.dataframe(results[cond][['CUST_NO','ACTUAL','PREDICT']].reset_index(drop=True))


def predict_order_cust(df_cluster, df_ord, df_cust):

    st.write("")

    if "predict_cluster" not in st.session_state:
        st.session_state.predict_cluster = 2

    if "predict_slitm" not in st.session_state:
        st.session_state.predict_slitm = 2231282389

    # 조회조건
    col1, col2, col3, col4 = st.columns([3, 3, 2, 3])

    with col1:
        predict_cluster = st.selectbox("클러스터: ", sorted(df_cluster['cluster'].unique()))
        st.session_state.predict_cluster = predict_cluster

    with col2:
        predict_slitm = st.text_input("상품코드: ", placeholder="예: 2231282389")
        st.session_state.predict_slitm = predict_slitm

    with col3:
        st.write("")
        predict_btn_flag =  st.button("주문 고객 예측")

    with col4:
        st.write("")

    if predict_btn_flag:
        predict(df_cluster, df_ord, df_cust)

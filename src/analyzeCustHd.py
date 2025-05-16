import pandas as pd
from datetime import datetime
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import io

@st.cache_data
def get_data():
    df_ord = pd.read_csv('./file/bfmt_ord.csv', encoding='cp949', low_memory=False)
    df_cust = pd.read_parquet('./file/ord_cust.parquet')
    df_bfmt = pd.read_csv('./file/broad_info.csv', encoding='utf-8', low_memory=False)

    return df_ord, df_cust, df_bfmt

def get_age_group(age):
    if pd.isnull(age):
        return None
    elif age < 20:
        return "10대 이하"
    elif age < 30:
        return "20대"
    elif age < 40:
        return "30대"
    elif age < 50:
        return "40대"
    elif age < 60:
        return "50대"
    elif age < 70:
        return "60대"
    else:
        return "70대 이상"

def preprocess_data(df_ord, df_cust, today):
    # 문자형
    df_ord['BROD_DT'] = df_ord['BROD_DT'].astype(str)
    df_ord['BFMT_NO'] = df_ord['BFMT_NO'].astype(str)
    df_ord['SLITM_CD'] = df_ord['SLITM_CD'].astype(str)
    df_ord['ORD_NO'] = df_ord['ORD_NO'].astype(str)
    df_ord['CUST_NO'] = df_ord['CUST_NO'].astype(str)
    
    df_cust['CUST_NO'] = df_cust['CUST_NO'].astype(str)
    
    # 정수형
    df_ord['INSM_MTHS'] = pd.to_numeric(df_ord['INSM_MTHS'], errors='coerce').astype('Int64')
    
    # 카테고리형
    df_ord['ITEM_GBCD'] = df_ord['ITEM_GBCD'].astype('category')
    df_ord['ITEM_GBNM'] = df_ord['ITEM_GBNM'].astype('category')
    df_ord['INTG_ITEM_GBCD'] = df_ord['INTG_ITEM_GBCD'].astype('category')
    df_ord['INTG_ITEM_GBNM'] = df_ord['INTG_ITEM_GBNM'].astype('category')
    df_ord['LAST_ORD_STAT_GBCD'] = df_ord['LAST_ORD_STAT_GBCD'].astype('category')
    df_ord['LAST_ORD_STAT_GBNM'] = df_ord['LAST_ORD_STAT_GBNM'].astype('category')
    df_ord['ITEM_MDA_GBCD'] = df_ord['ITEM_MDA_GBCD'].astype('category')
    df_ord['ITEM_MDA_GBNM'] = df_ord['ITEM_MDA_GBNM'].astype('category')
    df_ord['ACPT_CH_GBCD'] = df_ord['ACPT_CH_GBCD'].astype('category')
    df_ord['ACPT_CH_GBNM'] = df_ord['ACPT_CH_GBNM'].astype('category')
    df_ord['LAST_STLM_STAT_GBCD'] = df_ord['LAST_STLM_STAT_GBCD'].astype('category')
    df_ord['LAST_STLM_STAT_GBNM'] = df_ord['LAST_STLM_STAT_GBNM'].astype('category')
    df_ord['PAY_WAY_GBCD'] = df_ord['PAY_WAY_GBCD'].astype('category')
    df_ord['PAY_WAY_GBNM'] = df_ord['PAY_WAY_GBNM'].astype('category')
    df_ord['PAY_WAY_GBCD'] = df_ord['PAY_WAY_GBCD'].astype('category')
    df_ord['PAY_WAY_GBCD'] = df_ord['PAY_WAY_GBCD'].astype('category')
    df_ord['PAY_WAY_GBCD'] = df_ord['PAY_WAY_GBCD'].astype('category')
    
    # 날짜형
    df_ord['BROD_STRT_DTM'] = pd.to_datetime(df_ord['BROD_STRT_DTM'], errors='coerce')
    df_ord['BROD_END_DTM'] = pd.to_datetime(df_ord['BROD_END_DTM'], errors='coerce')
    df_ord['PTC_ORD_DTM'] = pd.to_datetime(df_ord['PTC_ORD_DTM'], errors='coerce')
    df_ord['ORD_STAT_PROC_DTM'] = pd.to_datetime(df_ord['ORD_STAT_PROC_DTM'], errors='coerce')

    df_cust['BYMD_DT'] = pd.to_datetime(df_cust['BYMD_DT'], errors='coerce')

    df_cust['AGE'] = (df_cust['BYMD_DT'].apply(
        lambda x: (today - x).days if pd.notnull(x) and x >= pd.Timestamp('1900-01-01') and x <= today else None
    ) // 365)
    df_cust['AGE_GROUP'] = df_cust['AGE'].apply(get_age_group)

    return df_ord, df_cust

def create_cust_cluster(df_ord, df_cust, today):
    
    cust_summary = df_ord.groupby('CUST_NO').agg(
        total_orders = ('ORD_NO', 'nunique'),
        unique_products = ('SLITM_CD', 'nunique'),
        first_order = ('PTC_ORD_DTM', 'min'),
        last_order = ('PTC_ORD_DTM', 'max')
    ).reset_index()

    # 고객별 총 구매 금액(monetary) 계산
    monetary_df = df_ord.groupby('CUST_NO')['LAST_STLM_AMT'].sum().reset_index()
    monetary_df.rename(columns={'LAST_STLM_AMT': 'monetary'}, inplace=True)

    cust_summary['recency'] = (today - cust_summary['last_order']).dt.days
    cust_summary['period'] = (cust_summary['last_order'] - cust_summary['first_order']).dt.days + 1
    cust_summary['frequency'] = cust_summary['total_orders'] / cust_summary['period']

    cust_summary = cust_summary.merge(monetary_df, on='CUST_NO', how='left')
    
    cust_features = cust_summary.merge(df_cust[['CUST_NO','SEX_GBNM','AGE','AGE_GROUP']], on='CUST_NO', how='left', validate='one_to_one')

    # 범주형 -> 숫자형 변수로 변환
    cust_features = pd.get_dummies(cust_features, columns=['SEX_GBNM','AGE_GROUP'], drop_first=True)

    # 클러스터링에 사용할 컬럼만 선택 (원핫인코딩 컬럼도 추가)
    feature_cols = ['total_orders','unique_products','recency','frequency','monetary'] + [col for col in cust_features.columns if col.startswith('SEX_GBNM_') or col.startswith('AGE_GROUP_')]

    # 결측값 처리 : AGE 결측치 0으로 채움
    X_cluster = cust_features[feature_cols].fillna(0)

    # 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    # KMeans 클러스터링
    kmeans = KMeans(n_clusters=5, random_state=0)
    cust_features['cluster'] = kmeans.fit_predict(X_scaled) 

    cluster_summary = cust_features.groupby('cluster')[feature_cols].mean()

    # 차트 그리기
    # 1. 클러스터 크기 확인
    st.write("")
    st.subheader("🧑‍🤝‍🧑 클러스터 크기 확인")
    cluster_size = cust_features['cluster'].value_counts()

    # 사이즈 줄이고 여백 최적화
    fig1, ax1 = plt.subplots(figsize=(4, 3), dpi=150)
    cluster_size.plot(kind='bar', ax=ax1)

    # 폰트 사이즈 줄이기
    ax1.tick_params(axis='x', labelsize=6)
    ax1.tick_params(axis='y', labelsize=6)
    ax1.set_title("클러스터별 고객 수", fontsize=6)
    ax1.set_xlabel("클러스터", fontsize=5)
    ax1.set_ylabel("고객 수", fontsize=5)

    # 여백 줄이고 이미지로 출력
    buf = io.BytesIO()
    plt.tight_layout()
    fig1.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    st.image(buf)

    # 2. 클러스터 특성 확인
    st.write("")
    st.subheader("🧬 클러스터 특성 확인")

    # 폰트 설정
    plt.rcParams['font.family'] = 'Malgun Gothic'

    # 클러스터 요약 데이터를 사용한 히트맵
    cluster_mean = cluster_summary.transpose()
    mean_table = cluster_mean.div(cluster_mean.max(axis=1), axis=0)

    fig2, ax2 = plt.subplots(figsize=(8, 5))

    # 히트맵 생성
    heatmap = sns.heatmap(
        mean_table,
        annot=True,
        fmt='.3f',
        linewidths=0.1,
        annot_kws={'fontsize': 4},
        cmap='RdYlBu_r',
        ax=ax2,
        cbar_kws={'shrink': 0.5}  # 컬러바 자체 크기 조절
    )

    # 컬러바 눈금(label) 폰트 크기 조절
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=5)  # 여기가 핵심

    # 제목과 레이블 폰트 크기 조정
    ax2.set_title('cluster X 변수 mean table', fontsize=6)
    ax2.set_xlabel("클러스터", fontsize=5)
    ax2.set_ylabel("변수", fontsize=5)

    # y축 tick label 폰트 크기 줄이기
    ax2.tick_params(axis='y', labelsize=5)

    # 필요 시 x축도 조절
    ax2.tick_params(axis='x', labelsize=6)

    # 여백 최적화 및 출력
    plt.tight_layout()
    st.pyplot(fig2)

    st.markdown("""
    ##### 📘 고객 주문 행동 지표 설명

    - **recency (최근성)**: 마지막 주문일로부터 오늘까지의 일수  
    → 고객이 **얼마나 오랫동안 주문을 하지 않았는지** 나타냅니다. 낮을수록 최근에 주문한 고객입니다.

    - **period (활동 기간)**: 첫 주문일부터 마지막 주문일까지의 총 기간 (일 단위)  
    → 고객의 **활동 기간**을 보여줍니다. 길수록 오랜 기간 동안 활동한 고객입니다.

    - **frequency (주문 빈도)**: 활동 기간 내 평균 주문 횟수  
    → 고객이 **얼마나 자주 주문했는지** 나타냅니다. 값이 높을수록 충성도가 높을 수 있습니다.

    - **monetary (구매 금액)**: 고객이 주문에 사용한 **총 금액 또는 평균 금액**  
    → 고객의 **경제적 가치**를 나타냅니다. 기업에 대한 수익 기여도를 판단하는 데 사용됩니다.

    <br>
    """, unsafe_allow_html=True)

    # 3. 변수별 클러스터 차이 확인
    st.write("")
    st.subheader("🔍 변수별 클러스터 차이 확인")
    fig3, axes = plt.subplots(nrows=2, ncols=7, figsize=(20, 8))

    sns.barplot(data=cust_features, y='cluster', x='total_orders', orient='h', ax=axes[0, 0])
    sns.barplot(data=cust_features, y='cluster', x='unique_products', orient='h', ax=axes[0, 1])
    sns.barplot(data=cust_features, y='cluster', x='recency', orient='h', ax=axes[0, 2])
    sns.barplot(data=cust_features, y='cluster', x='period', orient='h', ax=axes[0, 3])
    sns.barplot(data=cust_features, y='cluster', x='frequency', orient='h', ax=axes[0, 4])
    sns.barplot(data=cust_features, y='cluster', x='monetary', orient='h', ax=axes[0, 5])
    sns.barplot(data=cust_features, y='cluster', x='SEX_GBNM_여자', orient='h', ax=axes[0, 6])
    
    sns.barplot(data=cust_features, y='cluster', x='AGE_GROUP_20대', orient='h', ax=axes[1, 0])
    sns.barplot(data=cust_features, y='cluster', x='AGE_GROUP_30대', orient='h', ax=axes[1, 1])
    sns.barplot(data=cust_features, y='cluster', x='AGE_GROUP_40대', orient='h', ax=axes[1, 2])
    sns.barplot(data=cust_features, y='cluster', x='AGE_GROUP_50대', orient='h', ax=axes[1, 3])
    sns.barplot(data=cust_features, y='cluster', x='AGE_GROUP_60대', orient='h', ax=axes[1, 4])
    sns.barplot(data=cust_features, y='cluster', x='AGE_GROUP_70대 이상', orient='h', ax=axes[1, 5])
    sns.barplot(data=cust_features, y='cluster', x='SEX_GBNM_남자', orient='h', ax=axes[1, 6])

    plt.tight_layout()
    st.pyplot(fig3) 

    return cust_features

def analyze_cust_cluster(df_cluster, df_ord):
    st.write("")
    st.write("---")
    st.header("🛒 클러스터 고객 주문 내역 분석")

    selected_cluster = st.selectbox("클러스터 선택", [0, 1, 2, 3, 4], index=0)
    st.session_state.selected_cluster = selected_cluster
    filtered_cluster = df_cluster[df_cluster["cluster"] == selected_cluster]

    # 클러스터 필터링
    # if selected_cluster == "전체":
    #     st.write("전체를 선택하는 경우 데이터는 100,000건으로 제한됩니다.")
    #     filtered_cluster = df_cluster.head(100000)
    # else:
    #     filtered_cluster = df_cluster[df_cluster["cluster"] == selected_cluster]

    # bool → str 형변환
    filtered_cluster = filtered_cluster.copy()
    bool_cols = filtered_cluster.select_dtypes(include='bool').columns
    filtered_cluster[bool_cols] = filtered_cluster[bool_cols].astype(str)

    # 주문 정보 필터링
    cond_ord = df_ord['CUST_NO'].isin(filtered_cluster['CUST_NO'])
    filtered_orders = df_ord[cond_ord]

    st.subheader(f"🧾 클러스터 {selected_cluster} 의 고객 주문내역")
    st.dataframe(filtered_orders)



    filtered_orders['주문시간대'] = filtered_orders['PTC_ORD_DTM'].dt.hour
    hourly_order = filtered_orders.groupby('주문시간대')['ORD_NO'].count().reset_index()

    fig_hour = px.bar(hourly_order, x='주문시간대', y='ORD_NO', labels={'ORD_NO':'주문수'}, title="시간대별 주문수")
    st.plotly_chart(fig_hour)

    # 총매출 기준 정렬 후 상위 10개 추출
    # 상품명 + 상품코드 조합 컬럼 생성
    filtered_orders['상품명(코드)'] = filtered_orders['SLITM_NM'] + " (" + filtered_orders['SLITM_CD'].astype(str) + ")"

    # 매출 집계
    cat_sales = (
        filtered_orders.groupby('상품명(코드)')['LAST_STLM_AMT']
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    # 막대그래프 생성
    fig_cat = px.bar(
        cat_sales,
        x='상품명(코드)',
        y='LAST_STLM_AMT',
        labels={'상품명(코드)': '상품명(코드)', 'LAST_STLM_AMT': '총매출'},
        title="상품별 매출 Top 10"
    )

    st.plotly_chart(fig_cat)

@st.cache_data
def load_data():
    today = pd.Timestamp.today().normalize()

    df_ord, df_cust, df_bfmt = get_data()
    df_ord, df_cust = preprocess_data(df_ord, df_cust, today)

    return df_ord, df_cust

def cluster_cust(df_ord, df_cust):
    today = pd.Timestamp.today().normalize()
    df_cluster = create_cust_cluster(df_ord, df_cust, today)

    if "selected_cluster" not in st.session_state:
        st.session_state.selected_cluster = 0

    analyze_cust_cluster(df_cluster, df_ord)

    return df_cluster

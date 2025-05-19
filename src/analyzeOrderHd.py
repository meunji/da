import os
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events


def preprocess_data(raw):
    # 문자형
    raw['BROD_DT'] = raw['BROD_DT'].astype(str)
    raw['BFMT_NO'] = raw['BFMT_NO'].astype(str)
    raw['SLITM_CD'] = raw['SLITM_CD'].astype(str)
    raw['ORD_NO'] = raw['ORD_NO'].astype(str)
    raw['CUST_NO'] = raw['CUST_NO'].astype(str)
    
    # 정수형
    raw['INSM_MTHS'] = pd.to_numeric(raw['INSM_MTHS'], errors='coerce').astype('Int64')
    
    # 카테고리형
    raw['ITEM_GBCD'] = raw['ITEM_GBCD'].astype('category')
    raw['ITEM_GBNM'] = raw['ITEM_GBNM'].astype('category')
    raw['INTG_ITEM_GBCD'] = raw['INTG_ITEM_GBCD'].astype('category')
    raw['INTG_ITEM_GBNM'] = raw['INTG_ITEM_GBNM'].astype('category')
    raw['LAST_ORD_STAT_GBCD'] = raw['LAST_ORD_STAT_GBCD'].astype('category')
    raw['LAST_ORD_STAT_GBNM'] = raw['LAST_ORD_STAT_GBNM'].astype('category')
    raw['ITEM_MDA_GBCD'] = raw['ITEM_MDA_GBCD'].astype('category')
    raw['ITEM_MDA_GBNM'] = raw['ITEM_MDA_GBNM'].astype('category')
    raw['ACPT_CH_GBCD'] = raw['ACPT_CH_GBCD'].astype('category')
    raw['ACPT_CH_GBNM'] = raw['ACPT_CH_GBNM'].astype('category')
    raw['LAST_STLM_STAT_GBCD'] = raw['LAST_STLM_STAT_GBCD'].astype('category')
    raw['LAST_STLM_STAT_GBNM'] = raw['LAST_STLM_STAT_GBNM'].astype('category')
    raw['PAY_WAY_GBCD'] = raw['PAY_WAY_GBCD'].astype('category')
    raw['PAY_WAY_GBNM'] = raw['PAY_WAY_GBNM'].astype('category')
    raw['PAY_WAY_GBCD'] = raw['PAY_WAY_GBCD'].astype('category')
    raw['PAY_WAY_GBCD'] = raw['PAY_WAY_GBCD'].astype('category')
    raw['PAY_WAY_GBCD'] = raw['PAY_WAY_GBCD'].astype('category')
    
    # 날짜형
    raw['BROD_STRT_DTM'] = pd.to_datetime(raw['BROD_STRT_DTM'], errors='coerce')
    raw['BROD_END_DTM'] = pd.to_datetime(raw['BROD_END_DTM'], errors='coerce')
    raw['PTC_ORD_DTM'] = pd.to_datetime(raw['PTC_ORD_DTM'], errors='coerce')
    raw['ORD_STAT_PROC_DTM'] = pd.to_datetime(raw['ORD_STAT_PROC_DTM'], errors='coerce')

    return raw

def analyze_order_trend(df):
    daily_summary = df.groupby(df['PTC_ORD_DTM'].dt.date).agg({'ORD_NO':'count', 'LAST_STLM_AMT':'sum'})
    daily_summary.columns = ['주문수', '총매출']
    daily_summary = daily_summary.sort_index()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=daily_summary.index,
        y=daily_summary['총매출'],
        name='총매출',
        marker_color='indianred'
    ))

    fig.add_trace(go.Scatter(
        x=daily_summary.index,
        y=daily_summary['주문수'],
        name='주문수',
        yaxis='y2',
        mode='lines+markers'
    ))

    fig.update_layout(
        title='일별 총매출 및 주문수 추이',
        xaxis=dict(
            title='주문일자',
            rangeslider=dict(visible=True),
            type='date'
        ),
        yaxis=dict(title='총매출', showgrid=True, gridcolor='lightgray'),
        yaxis2=dict(title='주문수', overlaying='y', side='right', showgrid=False),
        height=850
    )

    fig.add_annotation(
        text='단위: 총매출(B=10억 원), 주문수(K=천건)',
        xref='paper', yref='paper',
        x=0.99, y=1.05,
        align="right",
        showarrow=False,
        font=dict(size=12, color="gray"),
        bgcolor='rgba(255,255,255,0.8)'
    )

    selected_points = plotly_events(
        fig,
        click_event=True,
        hover_event=False,
        select_event=False,
        override_height=850,
        key="order_trend_plot"
    )

    if selected_points:

        try:
            # 다양한 포맷에 대응하도록 robust하게 파싱
            selected_raw = selected_points[0]["x"]
            if isinstance(selected_raw, dict) and "$date" in selected_raw:
                selected_date = pd.to_datetime(selected_raw["$date"]).date()
            else:
                selected_date = pd.to_datetime(selected_raw).date()

            st.session_state.selected_date = selected_date
            st.success(f"선택된 날짜: {selected_date}")
        except Exception as e:
            st.warning(f"날짜 파싱 중 오류 발생: {e}")
            st.session_state.selected_date = None

    # 세션 상태에 날짜가 저장돼 있다면 해당 일자의 데이터 분석
    if "selected_date" in st.session_state and st.session_state.selected_date:
        analyze_order_daily(df, st.session_state.selected_date)


def analyze_order_daily(df, selected_date):
    # 필터링된 데이터 출력
    filtered_df = df[df['PTC_ORD_DTM'].dt.date == selected_date]
    if filtered_df.empty:
        st.warning(f"{selected_date} 에 해당하는 데이터가 없습니다.")
    else:
        st.subheader(f"🧾 {selected_date}의 주문 내역")
        st.dataframe(filtered_df)

    # 시간대별 주문수
    st.subheader("⏰ 시간대별 주문수")
    filtered_df['주문시간대'] = filtered_df['PTC_ORD_DTM'].dt.hour
    hourly_order = filtered_df.groupby('주문시간대')['ORD_NO'].count().reset_index()

    fig_hour = px.bar(hourly_order, x='주문시간대', y='ORD_NO', labels={'ORD_NO':'주문수'})
    st.plotly_chart(fig_hour)

    # 상품별 매출
    st.subheader("📦 상품별 총매출")
    cat_sales = filtered_df.groupby('SLITM_NM')['LAST_STLM_AMT'].sum().sort_values(ascending=False).reset_index()
    fig_cat = px.bar(cat_sales, x='SLITM_NM', y='LAST_STLM_AMT', labels={'SLITM_NM': '상품명', 'LAST_STLM_AMT': '총매출'})
    st.plotly_chart(fig_cat)

    col1, col2 = st.columns(2)

    with col1:
        # 결제수단 비율
        st.subheader("💳 결제수단 비율")
        pay_ratio = filtered_df['PAY_WAY_GBNM'].value_counts().reset_index()
        pay_ratio.columns = ['결제수단', '건수']
        fig_pie = px.pie(pay_ratio, names='결제수단', values='건수')
        st.plotly_chart(fig_pie)
    with col2:
        # 접수채널 비율
        st.subheader("🛎️ 접수채널 비율")
        pay_ratio = filtered_df['ACPT_CH_GBNM'].value_counts().reset_index()
        pay_ratio.columns = ['접수채널', '건수']
        fig_pie = px.pie(pay_ratio, names='접수채널', values='건수')
        st.plotly_chart(fig_pie)


# 주문 추이 분석
def analyze_order_main():
    # raw = pd.read_csv('./file/bfmt_ord.csv', encoding='cp949')
    raw = pd.read_parquet('./file/bfmt_ord.parquet', engine="pyarrow")

    df_raw = preprocess_data(raw)
    analyze_order_trend(df_raw)
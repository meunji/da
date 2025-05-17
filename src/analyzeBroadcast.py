import pandas as pd
import plotly.express as px
import streamlit as st


@st.cache_data
def load_data():
    df = pd.read_csv('./file/broad_info.csv', encoding='utf-8', parse_dates=['date'])

    # df['day_week'] = df['date'].dt.day_name(locale='ko_KR')  # 요일 추가
    
    # 리눅스 로케일 설정 안되어 있는 경우 맵핑
    df['day_week_en'] = df['date'].dt.day_name()

    # 영어 -> 한국어 매핑 딕셔너리
    en_to_ko = {
        'Monday': '월요일',
        'Tuesday': '화요일',
        'Wednesday': '수요일',
        'Thursday': '목요일',
        'Friday': '금요일',
        'Saturday': '토요일',
        'Sunday': '일요일',
    }

    df['day_week'] = df['day_week_en'].map(en_to_ko)


    df['time_slot'] = pd.to_datetime(df['time'], format='%H:%M').dt.hour

    # 데이터방송 company
    t_companies = ['현대홈쇼핑 플러스샵', 'GS홈쇼핑 마이샵', 'CJ온스타일 플러스','롯데원티비', 'NS홈쇼핑 샵플러스', '쇼핑엔티', 'KT알파쇼핑']

    # company_info를 기준으로 방송 유형 분류
    df['company_type'] = df['company_info'].apply(
        lambda x: '데이터방송' if x in t_companies else '라이브방송'
    )

    return df

@st.cache_data
def get_daily_chart(df, channel):
    df_filtered = df[df['company_info'] == channel]
    daily_counts = df_filtered.groupby(['date', 'category']).size().reset_index(name='건수')
    fig = px.bar(daily_counts, x='date', y='건수', color='category', title=f"{channel} - 일자별 방송 상품 분류", )
    return fig

@st.cache_data
def get_weekday_chart(df, channel):
    df_filtered = df[df['company_info'] == channel]
    weekday_counts = df_filtered.groupby(['day_week', 'category']).size().reset_index(name='건수')
    fig = px.bar(weekday_counts, x='day_week', y='건수', color='category', title=f"{channel} - 요일별 방송 상품 분류")
    return fig

@st.cache_data
def get_hourly_chart(df, channel):
    df_filtered = df[df['company_info'] == channel]
    hourly_counts = df_filtered.groupby(['time_slot', 'category']).size().reset_index(name='건수')
    fig = px.bar(hourly_counts, x='time_slot', y='건수', color='category', title=f"{channel} - 시간대별 방송 상품 분류")
    return fig


def get_all_data(df):
    selected_all_data = st.radio(
        "방송 유형 선택 ",
        ["전체", "라이브방송", "데이터방송"],
        horizontal=True
    )

    # 선택된 항목 출력
    # st.write(f"선택된 방송 유형: {selected_all_data}")

    if selected_all_data == "전체":
        filtered_df = df
    else:
        filtered_df = df[df["company_type"] == selected_all_data]

    col1, col2 = st.columns([2, 3])

    with col1:
        # 전체 카테고리 비중
        category_dist = filtered_df['category'].value_counts().reset_index()
        category_dist.columns = ['카테고리', '건수']

        fig = px.pie(category_dist, names='카테고리', values='건수', title='전체 카테고리 분포')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        # 회사별 카테고리 비중
        pivot = filtered_df.groupby(['company_info', 'category']).size().unstack(fill_value=0)
        pivot = pivot.T

        fig = px.imshow(
            pivot,
            labels=dict(x="채널", y="카테고리", color="방송수"),
            title="회사별 카테고리 분포"
        )

        fig.update_layout(
            margin=dict(l=80, r=10, t=100, b=80),  # 좌우 여백 줄이기
            xaxis=dict(tickangle=-45),            # x축 글씨 기울이기
            coloraxis_colorbar=dict(
                x=1.02,        # colorbar의 x축 위치 (1보다 크면 더 오른쪽)
                thickness=10,  # colorbar 너비 줄이기
                len=0.8        # colorbar 높이 줄이기
            )
        )

        st.plotly_chart(fig, use_container_width=False)

    col3, col4 = st.columns([2, 3])

    with col3:
        # 요일별 집중도
        weekday_total = filtered_df['day_week'].value_counts().reset_index()
        weekday_total.columns = ['요일', '방송수']

        fig = px.bar(weekday_total, x='요일', y='방송수', title='요일별 전체 방송 집중도')
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        # 시간대별 방송분포
        hourly_total = filtered_df['time_slot'].value_counts().sort_index().reset_index()
        hourly_total.columns = ['시간대', '방송수']

        fig = px.line(hourly_total, x='시간대', y='방송수', title='시간대별 전체 방송 집중도')
        st.plotly_chart(fig, use_container_width=True)


def get_company_data(df):

    st.write("---")
    st.header("📊 방송사별 편성 데이터 분석")

    # 조회조건 방송유형, 방송사
    col1, col2 = st.columns([2, 3])

    with col1:
        selected_broadcast_type = st.selectbox("방송 유형 선택", ["전체", "라이브방송", "데이터방송"])
        st.session_state.selected_broadcast_type = selected_broadcast_type

        # 선택된 유형에 따라 방송사 목록 필터링
        if st.session_state.selected_broadcast_type == "전체":
            filtered_df = df
        else:
            filtered_df = df[df["company_type"] == st.session_state.selected_broadcast_type]

        company_list = sorted(filtered_df['company_info'].unique())

    with col2:
        selected_company = st.selectbox("채널 선택", company_list, key="selected_company")

    # 방송사별 편성 지표
    if st.session_state.selected_company:
        st.plotly_chart(get_daily_chart(df, selected_company), use_container_width=True)
        st.plotly_chart(get_weekday_chart(df, selected_company), use_container_width=True)
        st.plotly_chart(get_hourly_chart(df, selected_company), use_container_width=True)


def analyze_broad_info():
    df = load_data()
    get_all_data(df)
    get_company_data(df)


# main() 함수 실행
if __name__ == "__main__":
    analyze_broad_info()
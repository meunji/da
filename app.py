import streamlit as st
import pandas as pd
import sys
import os

# import 경로 지정
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from automaticOrder import order_main
from crawlBroadcast import crawl_main
from analyzeBroadcast import analyze_broad_info
from analyzeOrderHd import analyze_order_main
from analyzeCustHd import load_data, cluster_cust
from predictOrderCustHd import predict_order_cust


# session_state key 상수로 정의
TAB1_SELECTED_MENU = "SELECTED_MENU_TAB1"
TAB2_SELECTED_MENU = "SELECTED_MENU_TAB2"
TAB3_SELECTED_MENU = "SELECTED_MENU_TAB3"
TAB4_SELECTED_MENU = "SELECTED_MENU_TAB4"

BROADCAST_ANALYSIS_FLAG = "BROADCAST_ANALYSIS_FLAG"
ORDER_ANALYSIS_FLAG = "ORDER_ANALYSIS_FLAG"
ORDER_PREDICT_FLAG = "ORDER_PREDICT_FLAG"

ORDER_ANALYSIS_DATE = "selected_date"

LOAD_DATA_FLAG = "LOAD_DATA_FLAG"
CLUSTER_CUST_FLAG = "CLUSTER_CUST_FLAG"


def init_session_state():
    if TAB1_SELECTED_MENU not in st.session_state:
        st.session_state[TAB1_SELECTED_MENU] = "새로고침"
    if TAB2_SELECTED_MENU not in st.session_state:
        st.session_state[TAB2_SELECTED_MENU] = "새로고침"
    if BROADCAST_ANALYSIS_FLAG not in st.session_state:
        st.session_state[BROADCAST_ANALYSIS_FLAG] = False
    if TAB3_SELECTED_MENU not in st.session_state:
        st.session_state[TAB3_SELECTED_MENU] = "새로고침"
    if ORDER_ANALYSIS_FLAG not in st.session_state:
        st.session_state[ORDER_ANALYSIS_FLAG] = False
    if TAB4_SELECTED_MENU not in st.session_state:
        st.session_state[TAB4_SELECTED_MENU] = "새로고침"
    if ORDER_PREDICT_FLAG not in st.session_state:
        st.session_state[ORDER_PREDICT_FLAG] = False
    if LOAD_DATA_FLAG not in st.session_state:
        st.session_state[LOAD_DATA_FLAG] = False
    if CLUSTER_CUST_FLAG not in st.session_state:
        st.session_state[CLUSTER_CUST_FLAG] = False
    if ORDER_ANALYSIS_DATE not in st.session_state:
            st.session_state[ORDER_ANALYSIS_DATE] = pd.to_datetime("2025-03-01").date()


def render_tab1():
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("자동주문"):
            st.session_state[TAB1_SELECTED_MENU] = "자동주문"

    with col2:
        if st.session_state.get(TAB1_SELECTED_MENU) == "자동주문":
            st.header("🎮 리모컨 자동주문")
            try:
                with st.spinner("리모컨 자동주문 진행중..."):
                    order_main()
                st.success("리모컨 자동주문이 완료되었습니다!")
            except Exception as e:
                st.error(f"자동주문 중 에러 발생: {e}")
            st.session_state[TAB1_SELECTED_MENU] = "완료"


def render_tab2():
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("새로고침"):
            st.session_state[TAB2_SELECTED_MENU] = "새로고침"
            st.session_state[BROADCAST_ANALYSIS_FLAG] = False

        if st.button("크롤링"):
            st.session_state[TAB2_SELECTED_MENU] = "크롤링"
            st.session_state[BROADCAST_ANALYSIS_FLAG] = False

        if st.button("분석"):
            st.session_state[TAB2_SELECTED_MENU] = "분석"
            st.session_state[BROADCAST_ANALYSIS_FLAG] = True

    with col2:
        selected = st.session_state.get(TAB2_SELECTED_MENU)
        if selected == "크롤링":
            st.header("📡 데이터 크롤링")
            try:
                with st.spinner("크롤링 중..."):
                    crawl_main()
                st.success("크롤링이 완료되었습니다!")
            except Exception as e:
                st.error(f"크롤링 중 에러 발생: {e}")
            st.session_state[TAB2_SELECTED_MENU] = "완료"
            st.session_state[BROADCAST_ANALYSIS_FLAG] = False

        if selected == "분석" and st.session_state.get(BROADCAST_ANALYSIS_FLAG, False):
            st.header("📊 동업계 편성 데이터 현황")
            analyze_broad_info()


def render_tab3():
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("현대홈쇼핑 주문 현황"):
            st.session_state[TAB3_SELECTED_MENU] = "주문현황"
            st.session_state[ORDER_ANALYSIS_FLAG] = True

    with col2:
        if (
            st.session_state.get(TAB3_SELECTED_MENU) == "주문현황"
            and st.session_state.get(ORDER_ANALYSIS_FLAG, False)
        ):
            st.header("📈 현대홈쇼핑 데이터방송 주문 데이터 분석")
            analyze_order_main()


def render_tab4():
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("고객 데이터로딩"):
            st.session_state[TAB4_SELECTED_MENU] = "고객 데이터로딩"
            st.session_state[LOAD_DATA_FLAG] = True
            st.session_state[CLUSTER_CUST_FLAG] = False
            st.session_state[ORDER_PREDICT_FLAG] = False

        if st.button("고객 클러스터링"):
            st.session_state[TAB4_SELECTED_MENU] = "고객 클러스터링"
            st.session_state[LOAD_DATA_FLAG] = False
            st.session_state[CLUSTER_CUST_FLAG] = True
            st.session_state[ORDER_PREDICT_FLAG] = False

        if st.button("고객 주문예측"):
            st.session_state[TAB4_SELECTED_MENU] = "고객 주문예측"
            st.session_state[LOAD_DATA_FLAG] = False
            st.session_state[CLUSTER_CUST_FLAG] = False
            st.session_state[ORDER_PREDICT_FLAG] = True

    with col2:
        selected = st.session_state.get(TAB4_SELECTED_MENU)
        if selected == "고객 데이터로딩":
            st.header("💾 현대홈쇼핑 주문 고객 데이터 로딩")
            with st.spinner(""):
                df_ord, df_cust = load_data()
                st.session_state["df_ord"] = df_ord
                st.session_state["df_cust"] = df_cust
            if st.session_state.get(LOAD_DATA_FLAG, False):
                st.success("데이터 로딩 완료되었습니다!")
            # st.session_state[TAB4_SELECTED_MENU] = "완료"
            st.session_state[LOAD_DATA_FLAG] = False

        elif selected == "고객 클러스터링":
            st.header("🧩 현대홈쇼핑 주문 고객 클러스터링")
            if "df_ord" in st.session_state and "df_cust" in st.session_state:
                if st.session_state.get(CLUSTER_CUST_FLAG, False):
                    df_cluster = cluster_cust(st.session_state["df_ord"], st.session_state["df_cust"])
                    st.session_state["df_cluster"] = df_cluster
            else:
                st.warning("먼저 고객 데이터를 로딩해주세요.")

        elif selected == "고객 주문예측":
            st.header("🎯 현대홈쇼핑 주문 고객 예측")

            if "df_ord" in st.session_state and "df_cust" in st.session_state:
                if "df_cluster" in st.session_state:
                    if st.session_state.get(ORDER_PREDICT_FLAG, False):
                        df_cluster = st.session_state["df_cluster"]
                        df_ord = st.session_state["df_ord"]
                        df_cust = st.session_state["df_cust"]
                        predict_order_cust(df_cluster, df_ord, df_cust)
                else:
                    st.warning("먼저 고객 클러스터링을 진행해주세요.")               
            else:
                st.warning("먼저 고객 데이터를 로딩해주세요.")

            

            # st.session_state[TAB4_SELECTED_MENU] = "완료"


def main():
    st.set_page_config(layout="wide")
    st.title("데이터방송 데이터 분석")

    init_session_state()

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "리모컨 자동주문",
            "동업계 편성정보",
            "현대홈쇼핑 주문현황",
            "현대홈쇼핑 고객정보",
        ]
    )

    with tab1:
        render_tab1()
    with tab2:
        render_tab2()
    with tab3:
        render_tab3()
    with tab4:
        render_tab4()


if __name__ == "__main__":
    main()
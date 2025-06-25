import streamlit as stru
import pandas as pd
import numpy as np
import pickle
import os
import re
import streamlit as st

from collections import Counter
from typing import Dict, Any, Optional

# 1) 기능별 키워드 정의 (속성 사전)
aspect_keywords = {
    '미백': ['미백', '톤업', '하얘짐', '화이트닝', '밝아', '색소침착'],
    '보습': ['보습', '촉촉', '수분', '건조하지', '속당김', '수분감', '촉촉해'],
    '트러블': ['트러블', '여드름', '자극', '진정', '붉어짐', '뾰루지', '민감'],
    '보호': ['자외선', '차단', '보호막', '방어', '먼지차단', '피부보호', '선크림'],
    '노화방지': ['주름', '탄력', '노화', '안티에이징', '처짐', '리프팅']
}

# 2) 감성 키워드 정의
positive_words = ['좋다', '좋아요', '촉촉하다', '만족', '개선', '진정됐다', '괜찮다', '흡수', '효과', '추천''👍','❤️','😁','강추','합격','맛집','재구매'
                  ]
negative_words = ['별로', '자극적', '트러블났다', '건조하다', '따갑다', '효과없다', '불편하다', '뒤집어짐', '실망', '아쉬워','피로감을 느끼다','과하다','여드름']

def label_review(text: str) -> Dict[str, Optional[str]]:
    """리뷰 텍스트를 분석하여 각 속성별 감정을 라벨링"""
    labels: Dict[str, Optional[str]] = {'미백': None, '보습': None, '트러블': None, '보호': None, '노화방지': None}
    
    for aspect, keywords in aspect_keywords.items():
        if any(keyword in str(text) for keyword in keywords):
            if any(pword in str(text) for pword in positive_words):
                labels[aspect] = '긍정'
            elif any(nword in str(text) for nword in negative_words):
                labels[aspect] = '부정'
            else:
                labels[aspect] = '중립'
    
    return labels

def analyze_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """리뷰 데이터 분석 및 벡터화"""
    # 라벨링 적용
    df['라벨'] = df['리뷰내용'].apply(label_review)
    
    # 벡터 변환
    aspects = list(aspect_keywords.keys())
    polarities = ['긍정', '부정', '중립']
    label_columns = [f"{a}_{p}" for a in aspects for p in polarities]
    
    # 벡터 변환 함수
    def convert_label_to_vector(label_series):
        result = []
        for label in label_series:
            vector = {col: 0 for col in label_columns}
            if isinstance(label, dict):
                for aspect, polarity in label.items():
                    if polarity is not None:
                        key = f"{aspect}_{polarity}"
                        if key in vector:
                            vector[key] = 1
            result.append(vector)
        return pd.DataFrame(result)
    
    # 벡터 변환
    vector_df = convert_label_to_vector(df['라벨'])
    df_vectorized = pd.concat([df, vector_df], axis=1)
    
    return df_vectorized

@st.cache_data
def load_and_process_data(file_path: str) -> Optional[pd.DataFrame]:
    """데이터 로드 및 처리 (캐싱 적용)"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        return analyze_reviews(df)
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {str(e)}")
        return None

# Streamlit 앱 메인 함수
def main():
    st.title("✨ 화장품 리뷰 분석 및 추천 시스템")
    st.write("AI 기반 화장품 리뷰 감정 분석 및 맞춤 추천")
    
    # 사이드바: 사용자 입력 컨트롤
    st.sidebar.title("🎯 추천 설정")
    
    # 파일 업로드 또는 경로 입력
    uploaded_file = st.sidebar.file_uploader("CSV 파일 업로드", type=['csv'])
    file_path = st.sidebar.text_input("또는 파일 경로 입력:", value="raw_reviews.csv")
    
    # 데이터 로드
    df = None
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
        df = analyze_reviews(df)
        st.sidebar.success("파일 업로드 완료!")
    elif os.path.exists(file_path):
        df = load_and_process_data(file_path)
        if df is not None:
            st.sidebar.success("데이터 로드 완료!")
    
    if df is None:
        st.error("데이터를 먼저 로드해주세요.")
        return
    
    # 기본 정보 표시
    st.write(f"**총 리뷰 수**: {len(df)}개")
    
    # 피부 타입 선택
    skin_type = st.sidebar.selectbox("피부 타입 선택", ['건성', '민감성', '지성', '중성'])
    
    # 중점 기능 선택
    aspects = list(aspect_keywords.keys())
    selected_aspects = st.sidebar.multiselect("중점 기능 선택", aspects, default=['보습'])
    
    # 임계치 설정
    pos_thresholds = {}
    neg_thresholds = {}
    for asp in selected_aspects:
        pos_thresholds[asp] = st.sidebar.slider(f"{asp} 긍정 최소비율", 0.0, 1.0, 0.5, 0.05)
        neg_thresholds[asp] = st.sidebar.slider(f"{asp} 부정 최대비율", 0.0, 1.0, 0.3, 0.05)
    
    # 피부 타입별 가중치 설정
    weights = {
        '건성': {'보습': 0.4, '보호': 0.2, '미백': 0.1, '트러블': 0.1, '노화방지': 0.2},
        '민감성': {'트러블': 0.4, '보호': 0.2, '보습': 0.2, '미백': 0.1, '노화방지': 0.1},
        '지성': {'트러블': 0.3, '보호': 0.2, '미백': 0.2, '노화방지': 0.2, '보습': 0.1},
        '중성': {'보습': 0.3, '노화방지': 0.3, '미백': 0.2, '보호': 0.1, '트러블': 0.1},
    }
    
    # 제품별 속성 긍/부정 비율 집계
    if '상품명' in df.columns:
        # 숫자 컬럼만 선택 (벡터 컬럼들)
        numeric_cols = [col for col in df.columns if any(f"{aspect}_" in col for aspect in aspects)]
        if numeric_cols:
            agg = df.groupby('상품명')[numeric_cols].mean().reset_index()
        else:
            st.error("분석 결과가 없습니다. 데이터를 확인해주세요.")
            return
    else:
        # 상품명이 없는 경우 카테고리별로 집계
        if '카테고리' in df.columns:
            numeric_cols = [col for col in df.columns if any(f"{aspect}_" in col for aspect in aspects)]
            if numeric_cols:
                agg = df.groupby('카테고리')[numeric_cols].mean().reset_index()
                agg = agg.rename(columns={'카테고리': '상품명'})
            else:
                st.error("분석 결과가 없습니다. 데이터를 확인해주세요.")
                return
        else:
            st.error("상품명 또는 카테고리 컬럼이 필요합니다.")
            return
    
    # 필터링: 선택된 기능별 임계치 적용
    mask = pd.Series(True, index=agg.index)
    for asp in selected_aspects:
        pos_col = f"{asp}_긍정"
        neg_col = f"{asp}_부정"
        if pos_col in agg.columns and neg_col in agg.columns:
            mask &= (agg[pos_col] >= pos_thresholds[asp]) & (agg[neg_col] <= neg_thresholds[asp])
    
    filtered = agg[mask].copy()
    
    if len(filtered) == 0:
        st.warning("조건에 맞는 제품이 없습니다. 임계치를 조정해보세요.")
        return
    
    # 개인화 점수 계산: 가중치 반영
    wt = weights[skin_type]
    filtered['추천점수'] = filtered.apply(
        lambda row: sum([wt[asp] * row.get(f"{asp}_긍정", 0) - wt[asp] * row.get(f"{asp}_부정", 0) for asp in wt.keys()]),
        axis=1
    )
    recommended = filtered.sort_values(by='추천점수', ascending=False)
    
    # 메인 페이지 출력
    st.header("🎯 맞춤 화장품 추천")
    st.write(f"**피부 타입**: {skin_type}")
    st.write(f"**중점 기능**: {', '.join(selected_aspects)}")
    
    # 추천 목록
    st.subheader("📋 추천 목록")
    display_cols = ['상품명', '추천점수']
    for asp in selected_aspects:
        display_cols.extend([f"{asp}_긍정", f"{asp}_부정"])
    
    available_cols = [col for col in display_cols if col in recommended.columns]
    st.dataframe(recommended[available_cols].head(10))
    
    # 추천점수 차트
    st.subheader("📊 추천점수 Top 10")
    if len(recommended) > 0:
        chart_data = recommended.set_index('상품명')['추천점수'].head(10)
        st.bar_chart(chart_data)
    
    # 상세 분석
    st.subheader("📈 상세 긍/부정 비율")
    if len(recommended) > 0:
        detail_cols = []
        for asp in selected_aspects:
            detail_cols.extend([f"{asp}_긍정", f"{asp}_부정"])
        
        available_detail_cols = [col for col in detail_cols if col in recommended.columns]
        if available_detail_cols:
            st.dataframe(recommended.set_index('상품명')[available_detail_cols].head(10))
    
    # 전체 통계
    st.subheader("📊 전체 분석 통계")
    
    # 감정 분포
    if '라벨' in df.columns:
        all_labels = []
        for label in df['라벨']:
            if isinstance(label, dict):
                for aspect, polarity in label.items():
                    if polarity is not None:
                        all_labels.append(f"{aspect}_{polarity}")
        
        if all_labels:
            label_counts = Counter(all_labels)
            st.write("**속성별 감정 분포:**")
            for label, count in label_counts.most_common():
                percentage = (count / len(all_labels)) * 100
                st.write(f"- {label}: {count}회 ({percentage:.1f}%)")
    
    # 결과 다운로드
    if len(recommended) > 0:
        csv = recommended.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 추천 결과 다운로드 (CSV)",
            data=csv,
            file_name="cosmetic_recommendations.csv",
            mime="text/csv"
        )
    
    st.markdown("---")
    st.markdown("*AI 기반 화장품 리뷰 분석 및 맞춤 추천 시스템*")

if __name__ == "__main__":
    main()
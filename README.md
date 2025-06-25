# OlivFeeling
모델학습 및 추천 
## 목차
1. [프로젝트 개요](#프로젝트-개요)  
2. [ABSA 라벨링 전략](#ABSA-라벨링-전략)    
3. [모델 학습 (LSTM & Bi-LSTM+Attention)](#모델-학습-LSTM--Bi-LSTMAttention)  
4. [실시간 추천 대시보드 (Streamlit)](#실시간-추천-대시보드-Streamlit)
   

## 프로젝트 개요
- **목표**: 리뷰에서 `미백`, `보습`, `트러블 완화`, `피부 보호`, `노화 방지` 등 **5개 기능**에 대한 감성(긍정/부정/중립)을 자동 추출하고, 이를 바탕으로 AI 추천 엔진을 구현  
- **파이프라인**:  
  1. 속성 기반 자동 라벨링  
  2. 멀티 레이블 벡터화  
  3. LSTM 계열 모델 학습 (Bi-LSTM + Attention 포함)  
  4. Streamlit 대시보드로 개인화 추천  

## ABSA 라벨링 전략
1. **속성 사전 정의**  
   - 분석 대상 기능: `미백`, `보습`, `트러블`, `보호`, `노화방지`  
   - 각 속성과 연관된 키워드 매핑  
   ```python
   aspect_keywords = {
     '미백': ['미백','톤업','화이트닝','하얘짐'],
     '보습': ['보습','촉촉','수분','속당김','건조하지'],
     '트러블': ['트러블','여드름','자극','진정','붉어짐'],
     '보호': ['자외선','차단','보호막','방어'],
     '노화방지': ['주름','탄력','노화','안티에이징']
   }

2. 감성 키워드 정의
```python
positive_words = ['좋다', '좋아요', '촉촉하다', '만족', '개선', '진정됐다', '괜찮다', '흡수', '효과', '추천''👍','❤️','😁','강추','합격','맛집','재구매']
negative_words = ['별로', '자극적', '트러블났다', '건조하다', '따갑다', '효과없다', '불편하다', '뒤집어짐', '실망', '아쉬워','피로감을 느끼다','과하다','여드름']
```

### 임계치(Threshold)

모델이 `0.0`~`1.0` 사이의 점수를 출력할 때, 이를 **긍정/부정/중립** 같은 확실한 태그로 바꿀 기준값을 **임계치**라고 함

#### 1. 기본 개념
- 모델 출력(예: `prob = 0.7`)이 **임계치** 이상이면 긍정, 미만이면 부정으로 분류  
- 중간 구간(예: `0.4 ≤ prob < 0.6`)을 **중립**으로 처리해 애매한 사례를 걸러낼 수 있음  

#### 2. 속성별 임계치 활용
화장품 리뷰처럼 여러 속성(미백, 보습, 트러블 등)을 동시에 분석할 때,  
각 속성별로 다른 임계치를 설정하면 더 세밀한 라벨링이 가능

```python
# 속성별 임계치 예시
thresholds = {
    '미백': {'pos': 0.6, 'neg': 0.5},
    '보습': {'pos': 0.5, 'neg': 0.5},
    '트러블': {'pos': 0.4, 'neg': 0.5},
}

def label_aspect(aspect, prob_pos, prob_neg):
    thr = thresholds[aspect]
    if prob_pos >= thr['pos']:
        return f"{aspect} 긍정"
    elif prob_neg >= thr['neg']:
        return f"{aspect} 부정"
    else:
        return f"{aspect} 중립"
```

1. 속성 언급 시 주변에 긍정 키워드→ 라벨 = `'긍정'`
2. 부정 키워드 주변에 부정 키워드 → 라벨 = `'부정'`
3. 기타 언급만 있을 경우→ 라벨 = `'중립'`


# 4) 실제 변환해서 원본에 붙이기(백터화)
vector_df = fast_convert_label_to_vector(df_all['라벨'])
df_vectorized = pd.concat([df_all, vector_df], axis=1)
5개 속성 × 3극성 → 15차원 이진 벡터

```python
# 3) 벡터 변환 함수 정의
def fast_convert_label_to_vector(label_series):
    result = []
    for label in label_series:
        # 기본값 0으로 채워진 딕셔너리
        vector = {col: 0 for col in label_columns}
        # 딕셔너리 형태(label_review 반환값)라면
        if isinstance(label, dict):
            for aspect, polarity in label.items():
                key = f"{aspect}_{polarity}"
                if key in vector:
                    vector[key] = 1
        result.append(vector)
    return pd.DataFrame(result)
```



# 모델 학습
## 모델 아키텍처

### 1) 기본 LSTM 모델
#### 단방향 빠른 학습
####  단방향 RNN 구조로 이전 시점 정보만 활용. 학습 속도가 빠르지만 문장 뒷부분 맥락 반영에 한계가 있음.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense

model = Sequential([
  Embedding(vocab_size, 128, input_length=max_len),
  LSTM(64),
  Dropout(0.5),
  Dense(64, activation='relu'),
  Dropout(0.5),
  Dense(15, activation='sigmoid')
])
model.compile(
  optimizer='adam',
  loss='binary_crossentropy',
  metrics=['accuracy']
)
```
###  2) Bi-LSTM+Attention
####  양방향 문맥과 핵심 토큰 강조
####  순방향과 역방향 LSTM을 결합해 과거·미래 양쪽 문맥을 모두 포착. 특히 긴 문장에서 앞·뒤 맥락을 균형 있게 학습
#### 핵심 감성 표현(예: "촉촉", "자극")에 집중시켜 성능을 높여줌  



```python
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout, Layer
from tensorflow.keras.models import Model

# 간단한 Attention 레이어 정의
class Attention(Layer):
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer='glorot_uniform',
                                 trainable=True)
    def call(self, x):
        e = tf.matmul(x, self.W)           # (batch, seq_len, 1)
        a = tf.nn.softmax(e, axis=1)       # (batch, seq_len, 1)
        return tf.reduce_sum(x * a, axis=1)  # (batch, hidden_size)

inputs = Input(shape=(max_len,))
x = Embedding(vocab_size, 128)(inputs)
x = Bidirectional(LSTM(64, return_sequences=True))(x)
context = Attention()(x)
outputs = Dense(15, activation='sigmoid')(context)

model = Model(inputs, outputs)
model.compile(
  optimizer='adam',
  loss='binary_crossentropy',
  metrics=['accuracy']
)


```
![image](https://github.com/user-attachments/assets/4566d14e-895b-462f-b2a6-80b4edf5650b)
#### 성능이 56%에 머무른다는 것은, 이제 데이터를 모델에 넣기까지의 과정, 즉 '전처리(Preprocessing)'나 학습 파라미터에 문제
#### 리뷰 텍스트(문장)를 숫자 시퀀스로 변환하고, 모든 시퀀스의 길이를 동일하게 맞춰주는 과정
#### 공개된 한국어 Word2Vec, FastText, 또는 KoBERT, KR-BERT 같은 모델의 임베딩을 가져와서, 모델의 첫 번째 Embedding 레이어에 적용이 필요하다는 것을 느낌

## 실시간 대시보드 
### 추천 로직
![250624 시연](https://github.com/user-attachments/assets/02a85b57-83e3-4783-8937-b21649b68fd3)

1. **다중 필터**: 조건 만족 제품 선별 (피부타입, 중점기능(미백,보습,트러블,보호,노화방지))
2. **개인화 점수**: Σ(긍정×가중치) − Σ(부정×가중치)
   
![image](https://github.com/user-attachments/assets/0ca6b86f-b4cb-44fd-b855-c1917c5cdfca)

![image](https://github.com/user-attachments/assets/8a410e31-699b-47b2-802e-b1e6cd2893f7)
![image](https://github.com/user-attachments/assets/8d5bfe17-3ddb-492c-8725-50e6ed61a351)


![image](https://github.com/user-attachments/assets/bc57087e-e801-47ee-9f48-a6053d7288ca)

### 파일을 업로드를 하면 피부타입(건성,민감성,지성,중성), 중점기능(미백,보습,트러블,보호,노화방지)에 
### 맞는 제품을 추천해주고, 상세 긍 부정 비율을 확인할 수 있음

### 업로드를 해서 분석하는것이 아닌, 사용자가 피부타입과 고민을 선택하면 추천 제품으로 나오는 것으로 변경이 필요함

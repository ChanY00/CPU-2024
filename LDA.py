import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords

# NLTK 불용어 다운로드
nltk.download('stopwords')
stop_words = stopwords.words('english')

# 데이터 로드
file_path = 'C:\\workstation\\CPU\\CPU_data_big.xlsx'  # 파일 경로를 적절히 수정하세요
df = pd.read_excel(file_path, engine='openpyxl')

# 데이터 전처리
df['출원일'] = pd.to_datetime(df['출원일'], format='%Y.%m.%d', errors='coerce')
df['출원연도'] = df['출원일'].dt.year

# 연도별 대분류 특허 출원 동향 분석
yearly_trend = df.groupby(['출원연도', '대분류']).size().unstack().fillna(0)

# 연도별 대분류 특허 출원 동향 시각화
plt.figure(figsize=(12, 8))
yearly_trend.plot(kind='line')
plt.title('연도별 대분류 특허 출원 동향')
plt.xlabel('출원연도')
plt.ylabel('특허 건수')
plt.legend(title='대분류')
plt.grid(True)
plt.show()

# 국가별 대분류 특허 출원 동향 분석
country_trend = df.groupby(['국가코드', '대분류']).size().unstack().fillna(0)

# 국가별 대분류 특허 출원 동향 시각화
plt.figure(figsize=(12, 8))
country_trend.plot(kind='bar', stacked=True)
plt.title('국가별 대분류 특허 출원 동향')
plt.xlabel('국가코드')
plt.ylabel('특허 건수')
plt.legend(title='대분류')
plt.grid(True)
plt.show()

# 주요 출원인 분석
top_applicants = df['대표출원인 국적'].value_counts().head(10)
print("상위 10개 출원인 국적:")
print(top_applicants)

# 특허 인용 수와 비특허 인용 수 분포 분석
df[['특허인용 수', '비특허인용 수']].plot(kind='hist', bins=50, alpha=0.7, figsize=(12, 8))
plt.title('특허 인용 수와 비특허 인용 수 분포')
plt.xlabel('인용 수')
plt.ylabel('빈도')
plt.grid(True)
plt.show()

# 키워드 분석을 위한 텍스트 데이터 전처리 (예: 요약 또는 청구항 텍스트 사용)
texts = df['요약'].fillna('')

# 벡터화 (불용어 제거 포함)
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words=stop_words)
dtm = vectorizer.fit_transform(texts)

# 키워드 추출
sum_words = dtm.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

# 상위 20개 키워드 출력
top_keywords = words_freq[:20]
print("상위 20개 키워드:")
for word, freq in top_keywords:
    print(f"{word}: {freq}")

# 상위 키워드 시각화
words = [word for word, freq in top_keywords]
freqs = [freq for word, freq in top_keywords]

plt.figure(figsize=(12, 8))
plt.barh(words, freqs, color='skyblue')
plt.xlabel('Frequency')
plt.title('Top 20 Keywords')
plt.gca().invert_yaxis()
plt.show()

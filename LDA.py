import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords

# NLTK stopwords download
nltk.download('stopwords')
stop_words = stopwords.words('english')

# 사용자 정의 불용어 추가
custom_stop_words = ['first', 'control', 'system', 'second', 'one', 'includes', 'least', 'power', 'assembly', 'body', 'main', 'connected', 'axis']
stop_words.extend(custom_stop_words)

# 데이터 로드
file_path = 'C:\\workstation\\CPU\\CPU_data_big.xlsx'  # Adjust file path as needed
df = pd.read_excel(file_path, engine='openpyxl')

# 데이터 전처리
df['출원일'] = pd.to_datetime(df['출원일'], format='%Y.%m.%d', errors='coerce')
df['출원연도'] = df['출원일'].dt.year

# 국가별 대분류 특허 출원 동향 분석
country_trend = df.groupby(['대표출원인 국적', '대분류']).size().unstack().fillna(0)

# 빈도 범위별 데이터 분류
country_trend_0_100 = country_trend[country_trend.sum(axis=1).between(0, 100)]
country_trend_100_200 = country_trend[country_trend.sum(axis=1).between(100, 200)]
country_trend_200_400 = country_trend[country_trend.sum(axis=1).between(200, 400)]
country_trend_400_1000 = country_trend[country_trend.sum(axis=1).between(400, 1000)]
country_trend_1000_plus = country_trend[country_trend.sum(axis=1) > 1000]

# 그래프 함수 정의
def plot_country_trend(data, title):
    plt.figure(figsize=(12, 8))
    data.plot(kind='bar', stacked=True)
    plt.title(title)
    plt.xlabel('Country Code')
    plt.ylabel('Number of Patents')
    plt.legend(title='Major Classification')
    plt.grid(True)
    plt.show()

# 그래프 그리기
plot_country_trend(country_trend_0_100, 'Trends in Patent Applications by Country and Major Classification (0-100)')
plot_country_trend(country_trend_100_200, 'Trends in Patent Applications by Country and Major Classification (100-200)')
plot_country_trend(country_trend_200_400, 'Trends in Patent Applications by Country and Major Classification (200-400)')
plot_country_trend(country_trend_400_1000, 'Trends in Patent Applications by Country and Major Classification (400-1000)')
plot_country_trend(country_trend_1000_plus, 'Trends in Patent Applications by Country and Major Classification (1000+)')

# 주요 출원인 분석
top_applicants = df['대표출원인 국적'].value_counts().head(10)
print("Top 10 Applicant Nationalities:")
print(top_applicants)

# 특허 인용 수와 비특허 인용 수 분포 분석
df[['특허인용 수', '비특허인용 수']].plot(kind='hist', bins=100, alpha=0.7, figsize=(12, 8))
plt.title('Distribution of Patent and Non-Patent Citations')
plt.xlabel('Number of Citations')
plt.ylabel('Frequency')
plt.xticks(range(0, 401, 50))  # x축 눈금을 0, 50, 100, 150, 200, 250, 300, 350, 400 으로 설정
plt.grid(True)
plt.show()

# 국가코드별 연도별 출원 동향 분석
country_yearly_trend = df.groupby(['출원연도', '국가코드']).size().unstack().fillna(0)

# 국가코드별 연도별 출원 동향 시각화
plt.figure(figsize=(12, 8))
country_yearly_trend.plot(kind='line')
plt.title('Trends in Patent Applications by Year and Country Code')
plt.xlabel('Application Year')
plt.ylabel('Number of Patents')
plt.legend(title='Country Code')
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
print("Top 20 Keywords:")
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

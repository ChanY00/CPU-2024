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
file_path = "C:\\workstation\\CPU\\CPU_data_big.xlsx"  # Adjust file path as needed

try:
    df = pd.read_excel(file_path, engine='openpyxl')
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    raise
except Exception as e:
    print(f"Error loading file: {e}")
    raise

# 데이터 전처리
df['출원일'] = pd.to_datetime(df['출원일'], format='%Y.%m.%d', errors='coerce')
df['출원연도'] = df['출원일'].dt.year

# 주요 출원인 분석 (상위 5개국)
top_5_countries = df['대표출원인 국적'].value_counts().head(5).index

# 상위 5개국 필터링
df_top_5 = df[df['대표출원인 국적'].isin(top_5_countries)]

# 국가별 대분류 특허 출원 동향 분석 (상위 5개국)
country_trend_top_5 = df_top_5.groupby(['대표출원인 국적', '대분류']).size().unstack().fillna(0)

# 그래프 함수 정의
def plot_country_trend(data, title):
    if data.empty:
        print(f"Warning: The data for '{title}' is empty.")
        return
    plt.figure(figsize=(12, 8))
    data.plot(kind='bar', stacked=True)
    plt.title(title)
    plt.xlabel('Country Code')
    plt.ylabel('Number of Patents')
    plt.legend(title='Major Classification')
    plt.grid(True)
    plt.show()

# 상위 5개국의 국가별 및 주요 분류별 특허 출원 경향 그래프 그리기
plot_country_trend(country_trend_top_5, 'Trends in Patent Applications by Top 5 Countries and Major Classification')

# 주요 출원인 분석
top_applicants = df['대표출원인 국적'].value_counts().head(10)
print("Top 10 Applicant Nationalities:")
print(top_applicants)

# 대표 출원인 국적 상위 10개 항목 바 그래프 시각화
plt.figure(figsize=(12, 8))
top_applicants.plot(kind='bar', color='skyblue')
plt.title('Top 10 Applicant Nationalities')
plt.xlabel('Nationality')
plt.ylabel('Number of Patents')
plt.grid(True)
plt.show()

# 국가코드별 연도별 출원 동향 분석 (상위 5개국)
country_yearly_trend_top_5 = df_top_5.groupby(['출원연도', '대표출원인 국적']).size().unstack().fillna(0)

# 국가코드별 연도별 출원 동향 시각화 (상위 5개국)
plt.figure(figsize=(12, 8))
country_yearly_trend_top_5.plot(kind='line')
plt.title('Trends in Patent Applications by Year and Country Code (Top 5 Countries)')
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

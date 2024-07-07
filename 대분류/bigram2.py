import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import bigrams
from collections import Counter
import spacy
from openpyxl import Workbook

# SpaCy 모델 로드
nlp = spacy.load("en_core_web_sm")

# 엑셀 파일 불러오기
file_path = 'C:\\workstation\\CPU\\CPU_data_big.xlsx'  # 파일 경로 지정
df = pd.read_excel(file_path)

# 필요한 열에 대한 전처리 수행
df['발명의 명칭'] = df['발명의 명칭'].astype(str)
df['발명의 명칭'] = df['발명의 명칭'].apply(lambda x: x.upper())  # 대문자 변환 혹은 다른 필요한 처리


# 함수 정의: 특수 문자 제거
def remove_special_characters(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text


# '발명의 명칭' 열에 적용하여 특수 문자 제거
df['발명의 명칭'] = df['발명의 명칭'].astype(str).apply(remove_special_characters)

# 불용어 제거 코드
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
custom_stopwords = ['METHOD', 'APPARATUS', 'USING', 'BASED', 'BY', 'OR', 'AND', 'OF', 'A', 'AND', 'FOR', 'IN', 'SOME',
                    'TO', 'WHICH', 'OR', 'OF', 'THE', 'WITH', 'MORE', 'IS', 'AN', 'AT', 'FIRST', 'FROM', 'AS', 'ON',
                    'TO', 'THAT', 'BY'
                                  'ONE', 'MORE', 'IS', 'AN', 'AT', 'FIRST', 'FROM', 'AS', 'BE', 'CAN', 'SECOND', 'EACH',
                    'MAY', 'DURING', 'ALSO', 'INTO', 'SUCH', 'INPUT', 'NEW', 'USED', 'THAN', 'HAVING', 'TAKEN', 'TRUE',
                    'WITHIN', 'THEIR', 'BEING', 'OVER'
    , 'FULL', 'ONE', 'ARE', 'THEREBY', 'I', 'THIS', 'IF', 'WHOM', 'ITS', 'THEN', 'END', 'JUST', 'N', 'THEREOF',
                    'PROVIDE', 'SET', 'CREATE', 'OTHER', 'LEAST', 'ASSIGN', 'INCLUDE', 'ALLOW', 'LELATE', 'CONTENT',
                    'STEP',
                    'DISCLOSE', 'TRANSLATED']  # 추가할 불용어 리스트
for word in custom_stopwords:
    stop_words.add(word)


# 불용어 제거 함수 정의
def remove_stopwords(text):
    word_tokens = word_tokenize(text)
    filtered_text = ' '.join([word for word in word_tokens if word.upper() not in stop_words])
    return filtered_text


# 불용어 제거 및 대체 작업 적용
df['발명의 명칭'] = df['발명의 명칭'].astype(str).apply(remove_stopwords)


# 명사의 원형 추출 함수 정의
def extract_noun_words(text):
    doc = nlp(text)
    noun_words = [token.text for token in doc if token.pos_ == "NOUN"]
    return ' '.join(noun_words).strip()


# DataFrame의 '발명의 명칭' 열에 적용하여 명사만 추출
df['발명의 명칭'] = df['발명의 명칭'].astype(str).apply(extract_noun_words)

# '출원일' 열의 형식 변환
df['출원일'] = pd.to_datetime(df['출원일'], format='%Y.%m.%d', errors='coerce')

# NaT 값을 제거하는 것도 고려할 수 있습니다.
df = df.dropna(subset=['출원일'])
df['출원연도'] = df['출원일'].dt.year

# 출원연도가 2000년 이후인 데이터만 선택
df_2000_onwards = df[df['출원연도'] >= 2000]


# bigram 생성 함수 정의
def create_bigrams(text):
    word_tokens = word_tokenize(text)
    bigram_list = ['_'.join(bigram) for bigram in bigrams(word_tokens)]
    return bigram_list


# 발명의 명칭 열에 적용하여 bigram 생성 및 할당
df_2000_onwards = df_2000_onwards.copy()
df_2000_onwards['발명의 명칭_bigrams'] = df_2000_onwards['발명의 명칭'].astype(str).apply(create_bigrams)


# 년도별 그룹화 및 빈도 계산
def create_bigram_trend_df(df):
    all_bigrams = set()
    yearly_bigram_freq = df.groupby('출원연도')['발명의 명칭_bigrams'].apply(
        lambda x: [item for sublist in x for item in sublist]).apply(Counter)

    for year, bigrams_counter in yearly_bigram_freq.items():
        all_bigrams.update(bigrams_counter.keys())

    bigram_trend_data = []
    for bigram in all_bigrams:
        trend = {'Bigram': bigram}
        for year in sorted(yearly_bigram_freq.keys()):
            trend[year] = yearly_bigram_freq[year][bigram] if bigram in yearly_bigram_freq[year] else 0

        # 중분류에서의 빈도 계산
        for category in df['중분류'].unique():
            category_bigrams = df[df['중분류'] == category]['발명의 명칭_bigrams'].sum()
            trend[category] = category_bigrams.count(bigram)

        trend['Total'] = sum(trend[year] for year in sorted(yearly_bigram_freq.keys()))
        bigram_trend_data.append(trend)

    bigram_trend_df = pd.DataFrame(bigram_trend_data)
    bigram_trend_df = bigram_trend_df.sort_values(by='Total', ascending=False)

    return bigram_trend_df


# 바이그램 추세 데이터프레임 생성
bigram_trend_df = create_bigram_trend_df(df_2000_onwards)


# 바이그램 추세 데이터프레임 변환 함수 정의
def transform_bigram_trend_df(bigram_trend_df):
    rows = []
    for _, row in bigram_trend_df.iterrows():
        bigram = row['Bigram']
        total = row['Total']
        years = sorted([col for col in bigram_trend_df.columns if isinstance(col, int)])
        categories = [col for col in bigram_trend_df.columns if col not in ['Bigram', 'Total'] + years]
        rows.append([bigram, 'Total', '', '', total])
        for year in years:
            year_row = [bigram, year] + [row[category] if row[category] != 0 else '' for category in categories] + [
                row[year]]
            rows.append(year_row)
    columns = ['Bigram', 'Year'] + categories + ['Frequency']
    transformed_df = pd.DataFrame(rows, columns=columns)
    return transformed_df


# 변환된 데이터프레임 생성
transformed_bigram_trend_df = transform_bigram_trend_df(bigram_trend_df)


# 엑셀 파일로 저장하는 함수 정의
def save_bigram_trends_to_excel(file_path, transformed_bigram_trend_df):
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        transformed_bigram_trend_df.to_excel(writer, index=False, sheet_name='Bigram Trends')


# 결과를 엑셀 파일로 저장
save_bigram_trends_to_excel('C:\\workstation\\CPU\\CPU-2024\\대분류\\CPU_data_big_entire.xlsx')

print("Excel 파일로 저장 완료!")

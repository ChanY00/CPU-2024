import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
from nltk import bigrams

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

# '요약' 열에 적용하여 특수 문자 제거
df['발명의 명칭'] = df['발명의 명칭'].astype(str).apply(remove_special_characters)

# 불용어 제거 코드
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
custom_stopwords = ['METHOD', 'APPARATUS', 'USING', 'BASED', 'BY', 'OR', 'AND', 'OF', 'A', 'AND', 'FOR', 'IN', 'SOME', 'TO', 'WHICH', 'OR', 'OF', 'THE', 'WITH', 'MORE', 'IS', 'AN', 'AT', 'FIRST', 'FROM', 'AS', 'ON', 'TO', 'THAT', 'BY'
                    'ONE', 'MORE', 'IS', 'AN', 'AT', 'FIRST', 'FROM', 'AS', 'BE', 'CAN', 'SECOND', 'EACH', 'MAY', 'DURING', 'ALSO', 'INTO', 'SUCH', 'INPUT', 'NEW', 'USED', 'THAN', 'HAVING', 'TAKEN', 'TRUE', 'WITHIN', 'THEIR', 'BEING', 'OVER'
                    , 'FULL', 'ONE', 'ARE', 'THEREBY', 'I', 'THIS', 'IF', 'WHOM', 'ITS', 'THEN', 'END', 'JUST', 'N', 'THEREOF', 'PROVIDE', 'SET', 'CREATE', 'OTHER', 'LEAST', 'ASSIGN', 'INCLUDE', 'ALLOW', 'LELATE', 'CONTENT', 'STEP',
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

# 표제어 추출을 위한 WordNet 데이터 다운로드
nltk.download('wordnet')

# WordNetLemmatizer 객체 생성
lemmatizer = WordNetLemmatizer()

# 표제어 추출 함수
def lemmatize_text(text):
    doc = nlp(text)
    lemmatized_text = ' '.join([token.lemma_ for token in doc])
    return lemmatized_text

# DataFrame의 요약 컬럼에 적용
df['발명의 명칭'] = df['발명의 명칭'].astype(str).apply(lemmatize_text)

# 명사의 원형 추출 함수 정의
def extract_noun_words(text):
    doc = nlp(text)
    noun_words = [token.text for token in doc if token.pos_ == "NOUN"]
    return ' '.join(noun_words).strip()

# DataFrame의 '요약' 열에 적용하여 명사만 추출
df['발명의 명칭'] = df['발명의 명칭'].astype(str).apply(extract_noun_words)

# '출원일' 열의 형식 변환
df['출원일'] = pd.to_datetime(df['출원일'], format='%Y.%m.%d', errors='coerce')

# NaT 값을 제거하는 것도 고려할 수 있습니다.
df = df.dropna(subset=['출원일'])
df['출원연도'] = df['출원일'].dt.year

# 출원연도가 2019년 이후인 데이터만 선택
df_2019_onwards = df[df['출원연도'] >= 2019]

# bigram 생성 함수 정의
def create_bigrams(text):
    word_tokens = word_tokenize(text)
    bigram_list = ['_'.join(bigram) for bigram in bigrams(word_tokens)]
    return ' '.join(bigram_list)

# 발명의 명칭 열에 적용하여 bigram 생성 및 할당
df_2019_onwards['발명의 명칭_bigrams'] = df_2019_onwards['발명의 명칭'].astype(str).apply(create_bigrams)

# 연도별, 기술별로 그룹화하고 키워드 빈도분석
keyword_freq = df_2019_onwards.groupby(['중분류', df_2019_onwards['출원일'].dt.year])['발명의 명칭'].apply(lambda x: ' '.join(x)).apply(lambda x: pd.Series(nltk.FreqDist(word_tokenize(x.upper())))).unstack().fillna(0)

# bigram에 대한 빈도분석
keyword_freq_bigrams = df_2019_onwards.groupby(['중분류', df_2019_onwards['출원일'].dt.year])['발명의 명칭_bigrams'].apply(lambda x: ' '.join(x)).apply(lambda x: pd.Series(nltk.FreqDist(word_tokenize(x.upper())))).unstack().fillna(0)

# 전치하여 행과 열을 바꾸고 CSV 파일로 저장
keyword_freq_transposed = keyword_freq.T
keyword_freq_bigrams_transposed = keyword_freq_bigrams.T

# 두 DataFrame을 합치기
merged_df = pd.concat([keyword_freq_transposed, keyword_freq_bigrams_transposed], axis=1)

# 합친 DataFrame을 EXCEL 파일로 저장
merged_df.to_excel("C:\\workstation\\CPU\\CPU_data_big_merge_2019_onwards.xlsx")

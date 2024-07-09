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
file_path = 'C:\\workstation\\CPU\\CPU-2024\\CPU_data_big_.xlsx'  # 파일 경로 지정
df = pd.read_excel(file_path)

# 필요한 열에 대한 전처리 수행
df['발명의 명칭'] = df['발명의 명칭'].astype(str)
df['발명의 명칭'] = df['발명의 명칭'].apply(lambda x: x.upper())


# 함수 정의: 특수 문자 제거
def remove_special_characters(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# '발명의 명칭' 열에 적용하여 특수 문자 제거
df['발명의 명칭'] = df['발명의 명칭'].astype(str).apply(remove_special_characters)
df['발명의 명칭'] = df['발명의 명칭'].apply(lambda x: x.upper())


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
    lemmatized_text = ','.join([token.lemma_ for token in doc])
    return lemmatized_text

# DataFrame의 발명의 명칭 열에 적용
df['발명의 명칭'] = df['발명의 명칭'].astype(str).apply(lemmatize_text)
df['발명의 명칭'] = df['발명의 명칭'].apply(lambda x: x.upper())


# 명사의 원형 추출 함수 정의
def extract_noun_words(text):
    doc = nlp(text)
    noun_words = [token.text for token in doc if token.pos_ == "NOUN"]
    return ' '.join(noun_words).strip()


# '출원일' 열의 형식 변환
df['출원일'] = pd.to_datetime(df['출원일'], format='%Y.%m.%d', errors='coerce')

# NaT 값을 제거하는 것도 고려할 수 있습니다.
df = df.dropna(subset=['출원일'])
df['출원연도'] = df['출원일'].dt.year

# 결과 데이터프레임에 필요한 열 선택
result_df = df[['중분류', '출원연도', '발명의 명칭']]

# 결과를 EXCEL 파일로 저장
result_df.to_excel("C:\\workstation\\CPU\\CPU-2024\\네트워크분석전처리.xlsx", index=False)

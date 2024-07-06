import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
from nltk import bigrams

# SpaCy 모델 로드
nlp = spacy.load("en_core_web_sm")

# 불용어 제거 함수 정의
def remove_stopwords(text):
    word_tokens = word_tokenize(text)
    return ' '.join([word for word in word_tokens if word.upper() not in stop_words])

# 표제어 추출 함수 정의
def lemmatize_text(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])

# 명사의 원형 추출 함수 정의
def extract_noun_words(text):
    doc = nlp(text)
    noun_words = [token.text for token in doc if token.pos_ == "NOUN"]
    return ' '.join(noun_words).strip()

# bigram 생성 함수 정의
def create_bigrams(text):
    word_tokens = word_tokenize(text)
    bigram_list = ['_'.join(bigram) for bigram in bigrams(word_tokens)]
    return ' '.join(bigram_list)

# 큰 데이터를 저장하기 위한 파일 분할 함수
def save_to_excel_in_chunks(df, file_prefix, chunk_size=100000):
    num_chunks = (len(df) // chunk_size) + 1
    for i in range(num_chunks):
        start_row = i * chunk_size
        end_row = start_row + chunk_size
        chunk = df.iloc[start_row:end_row]
        chunk.to_excel(f"{file_prefix}_part_{i+1}.xlsx", index=False)

# 엑셀 파일 불러오기
file_path = 'C:\\Users\\kseon\\CPU\\캠퍼스 유니버시아드\\코드\\CPU_data_big.xlsx'  # 파일 경로 지정
df = pd.read_excel(file_path)
print("엑셀 파일 불러오기 완료")

# 필요한 열에 대한 전처리 수행
df['발명의 명칭'] = df['발명의 명칭'].astype(str)
df['발명의 명칭'] = df['발명의 명칭'].apply(lambda x: x.upper())  # 대문자 변환
print("발명의 명칭 전처리 완료")

# 불용어 제거 코드
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
custom_stopwords = ['METHOD', 'APPARATUS', 'USING', 'BASED', 'BY', 'OR', 'AND', 'OF', 'A', 'FOR', 'IN', 'SOME', 'TO', 'WHICH', 'THE', 'WITH', 'MORE', 'IS', 'AN', 'AT', 'FIRST', 'FROM', 'AS', 'ON', 'THAT', 'BY', 'ONE', 'BE', 'CAN', 'SECOND', 'EACH', 'MAY', 'DURING', 'ALSO', 'INTO', 'SUCH', 'INPUT', 'NEW', 'USED', 'THAN', 'HAVING', 'TAKEN', 'TRUE', 'WITHIN', 'THEIR', 'BEING', 'OVER', 'FULL', 'THEREBY', 'I', 'THIS', 'IF', 'WHOM', 'ITS', 'THEN', 'END', 'JUST', 'N', 'THEREOF', 'PROVIDE', 'SET', 'CREATE', 'OTHER', 'LEAST', 'ASSIGN', 'INCLUDE', 'ALLOW', 'LELATE', 'CONTENT', 'STEP', 'DISCLOSE', 'TRANSLATED']
stop_words.update(custom_stopwords)

df['발명의 명칭'] = df['발명의 명칭'].apply(remove_stopwords)
print("불용어 제거 완료")

# 표제어 추출을 위한 WordNet 데이터 다운로드
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

df['발명의 명칭'] = df['발명의 명칭'].apply(lemmatize_text)
print("표제어 추출 완료")

# 명사의 원형 추출 적용
df['발명의 명칭'] = df['발명의 명칭'].apply(extract_noun_words)
print("명사의 원형 추출 완료")

# bigram 생성 및 중복 제거 적용
df['발명의 명칭_bigrams'] = df['발명의 명칭'].apply(create_bigrams)
df['발명의 명칭_bigrams'] = df['발명의 명칭_bigrams'].apply(lambda x: ' '.join(pd.Series(x.split()).drop_duplicates()))
print("바이그램 생성 및 중복 제거 완료")

# 제1출원인 빈도수 계산 및 상위 10개 필터링
applicant_freq = df['제1출원인'].value_counts().head(10)
top_applicants = applicant_freq.index.tolist()
df_top = df[df['제1출원인'].isin(top_applicants)]
print("상위 10개 제1출원인 필터링 완료")

# 연도별, 기술별로 그룹화하고 키워드 빈도분석
keyword_freq = df_top.groupby(['제1출원인', '중분류'])['발명의 명칭'].apply(lambda x: ' '.join(x)).apply(lambda x: pd.Series(nltk.FreqDist(word_tokenize(x.upper())))).unstack().fillna(0)
print("키워드 빈도 분석 완료")
keyword_freq_bigrams = df_top.groupby(['제1출원인', '중분류'])['발명의 명칭_bigrams'].apply(lambda x: ' '.join(x)).apply(lambda x: pd.Series(nltk.FreqDist(word_tokenize(x.upper())))).unstack().fillna(0)
print("바이그램 빈도 분석 완료")

# 전치하여 행과 열을 바꾸기
keyword_freq_transposed = keyword_freq.T
keyword_freq_bigrams_transposed = keyword_freq_bigrams.T

# '빈도' 열 추가
keyword_freq_transposed['빈도'] = keyword_freq_transposed.sum(axis=1)
keyword_freq_bigrams_transposed['빈도'] = keyword_freq_bigrams_transposed.sum(axis=1)

# 키워드 빈도 분석 결과를 여러 개의 EXCEL 파일로 저장
save_to_excel_in_chunks(keyword_freq_transposed, "C:\\Users\\kseon\\CPU\\캠퍼스 유니버시아드\\코드\\제1출원인\\CPU_person2_freq")
print("키워드 빈도 분석 결과 저장 완료")

# 바이그램 빈도 분석 결과를 여러 개의 EXCEL 파일로 저장
save_to_excel_in_chunks(keyword_freq_bigrams_transposed, "C:\\Users\\kseon\\CPU\\캠퍼스 유니버시아드\\코드\\제1출원인\\CPU_person2_bigram_freq")
print("바이그램 빈도 분석 결과 저장 완료")

# 제1출원인 빈도수 결과를 EXCEL 파일로 저장
applicant_freq.to_excel("C:\\Users\\kseon\\CPU\\캠퍼스 유니버시아드\\코드\\제1출원인\\CPU_applicant2_freq.xlsx")
print("제1출원인 빈도수 결과 저장 완료")
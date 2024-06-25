import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
from nltk import bigrams

# Load the Excel file
file_path = 'C:/Users/dnjsr/Desktop/특허 데이터 분석 참고 자료/코드/데이터/신약개발_완.xlsx'
df = pd.read_excel(file_path)

# 필요한 열에 대한 전처리 수행
df['Abstract'] = df['Abstract'].astype(str)
df['Abstract'] = df['Abstract'].apply(lambda x: x.upper())  # 대문자 변환 혹은 다른 필요한 처리

# 함수 정의: 특수 문자 제거
def remove_special_characters(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# 'Abstract' 열에 적용하여 특수 문자 제거
df['Abstract'] = df['Abstract'].astype(str).apply(remove_special_characters)


# 불용어 제거 코드
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
custom_stopwords = ['METHOD','APPARATUS', 'USING', 'BASED','BY','OR','AND', 'OF','A','AND','FOR','IN','SOME', 'TO', 'WHICH', 'OR', 'OF', 'THE', 'WITH','MORE', 'IS', 'AN', 'AT','FIRST', 'FROM','AS', 'ON', 'TO', 'THAT', 'BY'
                    'ONE','MORE', 'IS', 'AN', 'AT','FIRST', 'FROM','AS','BE','CAN','SECOND','EACH','MAY','DURING','ALSO', 'INTO','SUCH','INPUT','NEW','USED','THAN','HAVING','TAKEN','TRUE','WITHIN','THEIR','BEING','OVER'
                    , 'FULL','ONE','ARE','THEREBY','I','THIS','IF','WHOM','ITS','THEN','END','JUST','N','THEREOF', 'PROVIDE', 'SET', 'CREATE', 'OTHER', 'LEAST', 'ASSIGN', 'INCLUDE', 'ALLOW', 'LELATE', 'CONTENT', 'STEP', 
                    'DISCLOSE', 'TRANSLATED']  # 본문에 사용하시는 불용어 리스트
for word in custom_stopwords:
    stop_words.add(word)

# 불용어 제거 함수 정의
def remove_stopwords(text):
    word_tokens = word_tokenize(text)
    filtered_text = ' '.join([word for word in word_tokens if word.upper() not in stop_words])
    return filtered_text

# 불용어 제거 및 대체 작업 적용
df['Abstract'] = df['Abstract'].astype(str).apply(remove_stopwords)

#표제어추출
# WordNet 데이터 다운로드 (표제어 추출을 위해 필요)
nltk.download('wordnet')

# WordNetLemmatizer 객체 생성
lemmatizer = WordNetLemmatizer()

# 여러분의 DataFrame('df')을 가정하고 있는데, 여기서 'Abstract' 컬럼을 가정합니다.
# 해당 컬럼의 각 단어를 lemmatize하여 다시 컬럼에 할당하는 예시입니다.
df['Abstract'] = df['Abstract'].astype(str).apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

# SpaCy 모델 로드
nlp = spacy.load("en_core_web_sm")

# 표제어 추출 함수
def lemmatize_text(text):
    doc = nlp(text)
    lemmatized_text = ' '.join([token.lemma_ for token in doc])
    return lemmatized_text

# DataFrame의 Abstract 컬럼에 적용
df['Abstract'] = df['Abstract'].astype(str).apply(lemmatize_text)

# 명사의 원형 추출 함수 정의
def extract_noun_words(text):
    doc = nlp(text)
    noun_words = [token.text for token in doc if token.pos_ == "NOUN"]
    return ' '.join(noun_words).strip()

# DataFrame의 'Abstract' 열에 적용하여 명사만 추출
df['Abstract'] = df['Abstract'].astype(str).apply(extract_noun_words)



# 대체어 작성 코드
replacements = {
    "REALTITH": "REALTITY",
    "REALITY TECHNOLOGY": "REALITY-TECHNOLOGY",
    "INFORMATION TECHNOLOGY": "INFORMATION-TECHNOLOGY",
    "NEW TECHNOLOGY": "NEW-TECHNOLOGY",
    "ARTIFICIAL INTELLIGENCE": "ARTIFICIAL-INTELLIGENCE",
    "DEEP LEARNING": "DEEP-LEARNING",
    "NEURAL NETWORK": "NEURAL-NETWORK",
    "MACHINE LEARNING": "MACHINE-LEARNING",
    "HEAD MOUNTED GLASSES": "HEAD-MOUNTED-GLASSES",
    "HEAD MOUNTED DISPLAY": "HEAD-MOUNTED-DISPLAY",
    "HMD": "HEAD-MOUNTED-DISPLAY",
    "BIG DATA": "BIGDATA",
    "REAL WORLD": "REAL-WORLD",
    "PHYSICAL WORLD": "PHYSICAL-WORLD",
    "3D VIRTUAL": "3D-VIRTUAL",
    "USER EXPERIENCE": "USER-EXPERIENCE",
    "DIGITAL SPACE": "DIGITAL-SPACE",
    "DIGITAL TECHNOLOGIES": "DIGITAL-TECHNOLOGY",
    "VIRTUAL REALITY": "VIRTUAL-REALITY",
    "DIGITAL TWIN": "DIGITAL-TWIN",
    "DIGITAL TRANSFORMATION": "DIGITAL-TRANSFORMATION",
    "VIRTUAL SPACE": "VIRTUAL-SPACE",
    "VIRTUAL WORLD": "VIRTUAL-WORLDS",
    "AUGMENTED REALITY": "AUGMENTED-REALITY",
    "VIRTUAL REALITY": "VIRTUAL-REALITY",
    "MIXED REALITY": "MIXED-REALITY",
    "EXTENDED REALITY": "EXTENED-REALITY",
    "FIELD D": "FIELD-D",
    "P VALUE": "",
    "3 D": "3-D",
    "2 D": "2-D",
    "METHODS": "METHOD",
    "PATIENTS": "PATIENT",
    "SYSTEMS": "SYSTEM",
    "PROVIDED": "PROVIDE",
    "PROTOCOLS": "PROTOCOL",
    "THERAPEUTICS": "THERAPEUTIC",
    "INCLUDES": "INCLUDE",
    "DETERMINED": "DETERMINE",
    "DIAGNOSTICS": "DIAGNOSTIC",
    "SYMPTOMS": "SYMPTOM",
    "INTERVENTIONS": "INTERVENTION",
    "QUESTIONS": "QUESTION",
    "AMOUNTS": "AMOUNT",
    "DISORDERS": "DISORDER",
    "PROGRAMS": "PROGRAM",
    "SURVIVORS": "SURVIVOR",
    "IMPROVED": "IMPROVE",
    "PREDICTING": "PREDICT",
    "PERSONALIZED": "PERSONALIZE",
    "CONSIDERED": "CONSIDER",
    "DISCLOSED": "DISCLOSE",
    "INCLUDING": "INCLUDE",
    "INDIVIDUALIZED":"INDIVIDUAL",
    "DETERMINATION": "DETERMINE",
    "SPECIALIZED": "SPECIAL",
    "GROUPS": "GROUP",
    "MANAGING": "MANAGE",
    "DEVICES": "DEVICE",
    "STORES": "STORE",
    "DEVICESPECIFIC": "DEVICE-SPECIFIC",
    "ALTERS": "ALTER",
    "ALTERED": "ALTER",
    "ASSIGNED":"ASSIGN",
    "CAUSES": "CAUSE",
    "SELFSELECTING": "SELF-SELECTING",
    "RECIPES": "RECIPE",
    "ASSOCIATED": "ASSOCIATE",
    "INPUTS": "INPUT",
    "MARKERS": "MARKER",
    "BIOMARKERS": "BIOMARKER",
    "RECOMMENDATION": "RECOMMEND",
    "KNOWING": "KNOW",
    "CONTAINS": "CONTAIN",
    "UTILIZED": "UTILIZE",
    "COMPLEXITY": "COMPLEX",
    "MANAGED": "MANAGE",
    "ENABLES": "ENABLE",
    "NEEDS": "NEED",
    "PROCESSING": "PROCESS",
    "PREPROCESSING": "PREPROCESS",
    "HELPS": "HELP",
    "LARGER": "LARGE",
    "ANALYZES": "ANALYZE",
    "TRACKING": "TRACK",
    "UNDESIRED": "UNDESIRE",
    "EFFECTS": "EFFECT",
    "MEDICATIONS": "MEDICATION",
    "RECEIVES": "RECEIVE",
    "PRESCRIBED": "PRESCRIBE",
    "SELECTS": "SELECT",
    "INTERACTIONS": "INTERACTION",
    "DIRECTED": "DIRECTE",
    "MODELS": "MODEL",
    "VALIDATED": "VALIDATE",
    "SUBSETS": "SUBSET",
    "VERSIONS": "VERSION",
    "OUTCOMES": "OUTCOME",
    "CONSUMED": "CONSUME",
    "PROCESSED": "PROCESS",
    "TRANSFORMED": "TRANSFORM",
    "RESULTS": "RESULT",
    "RELATED": "RELATE",
    "MAPPED": "MAP",
    "DIMENSIONS": "DIMENSION",
    "MAPPING": "MAP",
    "UNSTRUCTURED": "UNSTRUCTURE",
    "CORRELATIONS": "CORRELATION",
    "STEPS": "STEP",
    "COMPRISING": "COMPRISE",
    "USERS": "USER",
    "ENCODED": "ENCODE",
    "COMPUTERSTORAGE": "COMPUTER-STORAGE",
    "IMPLEMENTATIONS": "IMPLEMENTATION",
    "ACCESSED": "ACCESSE",
    "TRIGGERS": "TRIGGER",
    "DETERMINING": "DETERMINE",
    "SATISFIED": "SATISFY",
    "DETECTING": "DETECT",
    "INDIVIDUALS": "INDIVIDUAL",
    "SELECTED": "SELECTE",
    "CBTCOGNITIVE": "CBT COGNITIVE",
    "APPARATUS": "APPARATU",
    "TREATING": "TREAT",
    "ADMINISTERING": "ADMINISTER",
    "FACTORS": "FACTOR",
    "AFFIRMATIONACHIEVEMENT": "AFFIRMATION-ACHIEVEMENT",
    "MODULES": "MODULE",
    "INSTRUCTIONS": "INSTRUCTION",
    "FNIRSFUNCTIONAL": "FNIRS-FUNCTIONAL",
    "NEARINFRARED": "NEAR-INFRARED",
    "FNIR": "FNIRS",
    "ADMINISTRATION": "ADMINISTER",
    "CORRELATED": "CORRELATE",
    "TRAINING": "TRAIN",
    "TRAINED": "TRAIN",
    "ANALYZED": "ANALYZE",
    "OUTPUTS": "OUTPUT",
    "CULTURES": "CULTURE",
    "TRANSFORMS": "TRANSFORM",
    "PATIENTUSER": "PATIENT-USER",
    "SUFFERING": "SUFFER",
    "ALLOWS": "ALLOW",
    "GUIDED": "GUIDE",
    "STRUCTURED": "STRUCTURE",
    "NONBEHAVIORAL": "NON-BEHAVIORAL",
    "THERAPIES": "THERARY",
    "TECHNOLOGIES": "TECHNOLOGIE",
    "TECHNIQUES": "TECHNIQUE",
    "DISEASES": "DISEASE",
    "CONDITIONS": "CONDITION",
    "THERAPIE": "THERAPY",
    "PHARMACEUTICALS": "PHARMACEUTICAL",
    "COMPOSITIONS": "COMPOSITION",
    "REGIMENS": "REGIMEN",
    "SKILLS": "SKILL",
    "DEVELOPING": "DEVELOP",
    "FEATURES": "FEATURE",
    "USERCREATED": "USER-CREATED",
    "TASKS": "TASK",
    "RECOMMENDER": "RECOMMEND",
    "GENERATING": "GENERATE",
    "CONTENTBASED": "CONTENT-BASED",
    "AREAS": "AREA",
    "CORRESPONDING":"CORRESPOND",
    "PERFORMS": "PERFORM",
    "EMBODIMENTS": "EMBODIMENT",
    "CONFIGURED": "CONFIGURE",
    "PATTERNS": "PATTERN",
    "SEQUENCES": "SEQUENCE",
    "TIMESERIES": "TIME-SERIES",
    "LEARNS": "LEARN",
    "LEARNING": "LEARN",
    "HEALTHCARE": "HEALTH-CARE",
    "CATEGORIZING": "CATEGORIZE",
    "PROVIDER": "PROVIDE",
    "COMPUTERIZED": "COMPUTING",
    "COUPLED": "COUPLE",
    "PROVIDER": "PROVIDE",
    "PREVIOUSLY": "PREVIOUS",
    "RELATIONSHIPS": "RELATIONSHIP",
    "DISPLAYED": "DISPLAY",
    "DIGITALTWIN": "DIGITAL-TWIN",
    "INDIVIDUALIZATION": "INDIVIDUAL",
    "PREDICTIVE": "PREDICT",
    "DERIVED": "DERIVE",
    "ESTIMATING": "ESTIMATE",
    "OPTIMIZED": "OPTIMIZE",
    "REPORTING": "REPORT",
    "INDICATIVE": "INDICATE",
    "RETRIEVED": "RETRIEVE",
    "CONTROLLED": "CONTROL",
    "ISSUED": "ISSUE",
    "ENHANCED": "ENHANCE",
    "REQUESTOR": "REQUEST",
    "ACCESSES": "ACCESSE",
    "RECORDED": "RECORD",
    "EXPERIENCED": "EXPERIENCE",
    "GENERATES": "GENERATE",
    "PATIENTPECIFIC": "PATIENT-SPECIFIC",
    "OUTLINING": "OUTLINE",
    "PROVIDES": "PROVIDE",
    "SENSED": "SENSE",
    "BONES": "BONE",
    "CONTINUOUS": "CONTINUE",
    "RETURNTOWORK": "RETURN-TO-WORK",
    "PROPOSED": "PROPOSE",
    "OBTAINING": "OBTAIN",
    "ATTRIBUTES": "ATTRIBUTE",
    "PROVIDING": "PROVIDE",
    "RECOMMENDED": "RECOMMEND",
    "TRANSMITS": "TRANSMIT",
    "ENCODES": "ENCODE",
    "BIOSIGNALS": "BIOSIGNAL",
    "LABELS": "LABEL",
    "COMPARES": "COMPARE",
    "DETERMINES": "DETERMINE",
    "FOODS": "FOOD",
    "OBJECTIVES": "OBJECTIVE",
    "PHYSICIANS": "PHYSICIAN",
    "SALES": "SALE",
    "INTERACTED": "INTERACTE",
    "PROVIDERS": "PROVIDER",
    "SERVICES": "SERVICE",
    "DEIDENTIFIED": "DE-IDENTIFIED",
    "ENSURES": "ENSURE",
    "STORED": "STORE",
    "DRUGS": "DRUG",
    "SIMULATING": "SIMULATE",
    "ENRICHED": "ENRICH",
    "BEHAVIORS": "BEHAVIOR",
    "TRANSMITTING": "TRANSMIT",
    "PRACTICES":"PRACTICE",
    "RECORDS": "RECORD",
    "PLANNED": "PLAN",
    "COMPARISON": "COMPARE",
    "SCORES": "SCORE",
    "CLASSIFYING": "CLASSIFY",
    "CATEGORIZED": "CATEGORIZE",
    "DISPENSING": "DISPENSE",
    "INCONSISTENCIES": "INCONSISTENCIE",
    "NAMES": "NAME",
    "CODES": "CODE",
    "PARAMETERS": "PARAMETER",
    "BIOSAMPLES": "BIOSAMPLE",
    "ANALYSING": "ANALYSIS",
    "ANALYTE": "ANALYSIS",
    "ANALYSING": "ANALYSIS",
    "VALUES": "VALUE",
    "ACTIONS": "ACTION",
    "BASES": "BASE",
    "COMPOUNDS": "COMPOUND",
    "INCLUDED": "INCLUDE",
    "IMAGES": "IMAGE",
    "SELECTING": "SELECT",
    "SELECTION": "SELECT",
    "LOGICBASED": "LOGICBASE",
    "DIAGNOSISSPECIFIC": "DIAGNOSIS-SPECIFIC",
    "REDUCED": "REDUCE",


}
# 단어 대체 적용
for key, value in replacements.items():
    df['Abstract'] = df['Abstract'].str.replace(key, value, case=False)

# 'Publication Year' 열을 정수형으로 변환
df['Publication Year'] = df['Publication Year'].astype(int)


# NaT 값을 제거하는 것도 고려할 수 있습니다.
df = df.dropna(subset=['Publication Year'])


# bigram 생성 함수 정의
def create_bigrams(text):
    word_tokens = word_tokenize(text)
    bigram_list = ['_'.join(bigram) for bigram in bigrams(word_tokens)]
    return ' '.join(bigram_list)

# 'Abstract' 열에 적용하여 bigram 생성
df['Abstract_bigrams'] = df['Abstract'].astype(str).apply(create_bigrams)

# 연도별, 기술별로 그룹화하고 키워드 빈도분석
keyword_freq = df.groupby(['Name', df['Publication Year']])['Abstract'].apply(lambda x: ' '.join(x)).apply(lambda x: pd.Series(nltk.FreqDist(word_tokenize(x.upper())))).unstack().fillna(0)

# bigram에 대한 빈도분석
keyword_freq_bigrams = df.groupby(['Name', df['Publication Year']])['Abstract_bigrams'].apply(lambda x: ' '.join(x)).apply(lambda x: pd.Series(nltk.FreqDist(word_tokenize(x.upper())))).unstack().fillna(0)


# 전치하여 행과 열을 바꾸고 CSV 파일로 저장
keyword_freq_transposed = keyword_freq.T
keyword_freq_bigrams_transposed = keyword_freq_bigrams.T

# 두 DataFrame을 합치기
merged_df = pd.concat([keyword_freq_transposed, keyword_freq_bigrams_transposed], axis=1)

# 합친 DataFrame을 CSV 파일로 저장
# merged_df.to_csv("C:/Users/JongMin/Desktop/biagram_키워드_빈도분석.csv")
merged_df.to_csv("C:/Users/dnjsr/Desktop/특허 데이터 분석 참고 자료/코드/논문_키워드_빈도분석.csv")

# 합친 DataFrame을 Excel 파일로 저장
# merged_df.to_excel("C:/Users/JongMin/Desktop/biagram_키워드_빈도분석.xlsx")
# merged_df.to_excel("C:/Users/dnjsr/Desktop/특허 데이터 분석 참고 자료/코드/논문_키워드_빈도분석.xlsx", index_label="Keyword")

import spacy
from spacy.cli import download

# 모델 설치
download("en_core_web_sm")

# 모델 로드
nlp = spacy.load("en_core_web_sm")
print("Model loaded successfully!")
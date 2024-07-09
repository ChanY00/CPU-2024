import pandas as pd
import time
from deep_translator import GoogleTranslator

# 엑셀 파일 불러오기
file_path = 'C:\\workstation\\CPU\\출원인.xlsx'  # 파일 경로 지정
df = pd.read_excel(file_path)

# 번역 함수 정의
def translate_text(text, retry=3):
    for _ in range(retry):
        try:
            return GoogleTranslator(source='auto', target='en').translate(text)
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            time.sleep(5)
    return text

# 발명의 명칭 열 번역
df['제1출원인'] = df['제1출원인'].apply(translate_text)

# 번역 결과를 확인하기 위해 데이터프레임 일부 출력
print(df[['제1출원인']].head())

# 번역된 데이터를 엑셀 파일로 저장
df.to_excel("C:\\workstation\\CPU\\출원인.xlsx", index=False)

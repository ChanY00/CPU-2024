import pandas as pd
import networkx as nx
import operator
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.font_manager as fm

# 엑셀 파일 읽기
file_path = "C:\\workstation\\CPU\\CPU-2024\\대분류\\big_네트워크분석_전처리.xlsx"
dataset = pd.read_excel(file_path)

# 필요한 열에 대한 전처리 수행
dataset['발명의 명칭'] = dataset['발명의 명칭'].astype(str).apply(lambda x: x.upper())

# 2023년도 데이터만 필터링
year_data = dataset[dataset['출원연도'] == 2020]

# 키워드 빈도 계산
all_keywords = []
for _, row in year_data.iterrows():
    keywords = row['발명의 명칭'].split(',')
    all_keywords.extend(keywords)

keyword_counts = Counter(all_keywords)
filtered_keywords = {k for k, v in keyword_counts.items() if v >= 8}

# 네트워크 생성
G_central = nx.Graph()

# 네트워크 구축 (빈도가 3번 이상인 키워드만 사용)
for _, row in year_data.iterrows():
    keywords = row['발명의 명칭'].split(',')
    filtered_row_keywords = [keyword for keyword in keywords if keyword in filtered_keywords]
    for i in range(len(filtered_row_keywords)):
        for j in range(i + 1, len(filtered_row_keywords)):
            G_central.add_edge(filtered_row_keywords[i], filtered_row_keywords[j], weight=1)

# 중심성 척도 계산
pgr = nx.pagerank(G_central)

# 그래프 생성
G = nx.Graph()
for key, value in pgr.items():
    G.add_node(key, nodesize=value*10000)  # 노드 크기를 더 확실히 보이게 조정

for _, row in year_data.iterrows():
    keywords = row['발명의 명칭'].split(',')
    filtered_row_keywords = [keyword for keyword in keywords if keyword in filtered_keywords]
    for i in range(len(filtered_row_keywords)):
        for j in range(i + 1, len(filtered_row_keywords)):
            G.add_edge(filtered_row_keywords[i], filtered_row_keywords[j])

# 레이아웃 설정
pos = nx.spring_layout(G)

# 폰트 설정
font_fname = "c:/Windows/Fonts/malgun.ttf"
fontprop = fm.FontProperties(fname=font_fname, size=12)

# 그래프 그리기
plt.figure(figsize=(12, 12))
nx.draw_networkx_nodes(G, pos, node_size=[data['nodesize'] for data in G.nodes.values()], node_color='red')
nx.draw_networkx_edges(G, pos, width=1)
nx.draw_networkx_labels(G, pos, font_family=fontprop.get_name(), font_size=12)
plt.title('Network Analysis for 2023')
plt.axis('off')
plt.show()

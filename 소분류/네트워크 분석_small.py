import pandas as pd
import networkx as nx
import operator
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from collections import Counter

# 엑셀 파일 읽기
file_path = 'C:\\Users\\dnjsr\\Desktop\\캠퍼스 유니버시아드\\코드\\CPU_keywords_2019_onwards.xlsx'  # 파일 경로 지정
dataset = pd.read_excel(file_path)

# 필요한 열에 대한 전처리 수행
dataset['발명의 명칭'] = dataset['발명의 명칭'].astype(str).apply(lambda x: x.upper())

# 2019년도 데이터만 필터링
year_data = dataset[dataset['출원연도'] == 2019]

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
dgr = nx.degree_centrality(G_central)
btw = nx.betweenness_centrality(G_central)
cls = nx.closeness_centrality(G_central)
egv = nx.eigenvector_centrality(G_central, max_iter=1000)
pgr = nx.pagerank(G_central)

# 중심성이 큰 순서대로 정렬
sorted_dgr = sorted(dgr.items(), key=operator.itemgetter(1), reverse=True)
sorted_btw = sorted(btw.items(), key=operator.itemgetter(1), reverse=True)
sorted_cls = sorted(cls.items(), key=operator.itemgetter(1), reverse=True)
sorted_egv = sorted(egv.items(), key=operator.itemgetter(1), reverse=True)
sorted_pgr = sorted(pgr.items(), key=operator.itemgetter(1), reverse=True)

# 단어 네트워크를 그려줄 Graph 선언
G = nx.Graph()

# 페이지 랭크에 따라 두 노드 사이의 연관성을 결정 (단어 연관성)
# 연결 중심성으로 계산한 척도에 따라 노드의 크기가 결정됨 (단어의 등장 빈도수)
for i in range(len(sorted_pgr)):
    G.add_node(sorted_pgr[i][0], nodesize=sorted_pgr[i][1])

for _, row in year_data.iterrows():
    keywords = row['발명의 명칭'].split(',')
    filtered_row_keywords = [keyword for keyword in keywords if keyword in filtered_keywords]
    for i in range(len(filtered_row_keywords)):
        for j in range(i + 1, len(filtered_row_keywords)):
            G.add_edge(filtered_row_keywords[i], filtered_row_keywords[j], weight=1)

# 노드의 색 지정
node_colors = ['grey' for node in G.nodes()]

# 노드 크기 조정
sizes = [G.nodes[node]['nodesize'] * 10000 for node in G]

# 폰트 설정
font_fname = "c:/Windows/Fonts/malgun.ttf"
fontprop = fm.FontProperties(fname=font_fname, size=18)

# 그리기 옵션 설정
options = {
    'edge_color': '#FFDEA2',
    'width': 1,
    'with_labels': True,
    'font_weight': 'regular',
    'font_family': fontprop.get_name(),
    'node_color': node_colors  # 노드 색 설정
}

# 그래프 그리기
plt.figure(figsize=(12, 12))
nx.draw(G, node_size=sizes, pos=nx.spring_layout(G, k=0.5, iterations=100), **options)
ax = plt.gca()
ax.collections[0].set_edgecolor('#555555')
plt.title('Network Analysis for 2019')
plt.show()

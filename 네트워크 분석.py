import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import operator
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.font_manager as fm
from adjustText import adjust_text

# 엑셀 파일 읽기
file_path = 'C:\\Users\\dnjsr\\Desktop\\캠퍼스 유니버시아드\\코드\\net_big.xlsx'
dataset = pd.read_excel(file_path)

# 필요한 열에 대한 전처리 수행
dataset['발명의 명칭'] = dataset['발명의 명칭'].astype(str).apply(lambda x: x.upper())

# 불용어 제거 코드
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
custom_stopwords = ['', '']
for word in custom_stopwords:
    stop_words.add(word)

def remove_stopwords(text):
    word_tokens = word_tokenize(text)
    filtered_text = ' '.join([word for word in word_tokens if not word.upper() in stop_words])
    return filtered_text

dataset['발명의 명칭'] = dataset['발명의 명칭'].apply(remove_stopwords)

# 2019년도 데이터만 필터링
year_data = dataset[dataset['출원연도'] == 2022]

# 키워드 빈도 계산
all_keywords = []
for _, row in year_data.iterrows():
    keywords = [keyword.strip() for keyword in row['발명의 명칭'].split(',')]
    all_keywords.extend(keywords)

keyword_counts = Counter(all_keywords)
filtered_keywords = {k for k, v in keyword_counts.items() if v >=15}

# 네트워크 생성 (방향성 있는 그래프 사용)
G = nx.DiGraph()
for _, row in year_data.iterrows():
    keywords = [keyword.strip() for keyword in row['발명의 명칭'].split(',')]
    filtered_row_keywords = [keyword for keyword in keywords if keyword in filtered_keywords]
    for i in range(len(filtered_row_keywords)):
        for j in range(i + 1, len(filtered_row_keywords)):
            if filtered_row_keywords[i] != filtered_row_keywords[j]:
                if G.has_edge(filtered_row_keywords[i], filtered_row_keywords[j]):
                    G[filtered_row_keywords[i]][filtered_row_keywords[j]]['weight'] += 1
                else:
                    G.add_edge(filtered_row_keywords[i], filtered_row_keywords[j], weight=1)

# 중심성 척도 계산
centrality_measures = {
    'Degree': nx.degree_centrality(G),
    'Betweenness': nx.betweenness_centrality(G),
    'Closeness': nx.closeness_centrality(G),
    'Eigenvector': nx.eigenvector_centrality(G, max_iter=1000),
    'PageRank': nx.pagerank(G)
}

# 그래프 그리기
plt.figure(figsize=(16, 16))
pos = nx.spring_layout(G, k=3.5, iterations=100)
ax = plt.gca()

# 노드 그리기 - 모든 노드 크기 고정
fixed_node_size = 250
nx.draw_networkx_nodes(G, pos, node_size=fixed_node_size, node_color='red', alpha=0.7)

# 엣지 그리기 - 화살표로
nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=10, width=1, edge_color='gray', alpha=0.5)

# 레이블 추가 및 겹침 방지
fontprop = fm.FontProperties(fname="c:/Windows/Fonts/malgun.ttf", size=12)
texts = [plt.text(pos[node][0], pos[node][1], node, fontproperties=fontprop) for node in G.nodes()]
adjust_text(texts, arrowprops=dict(arrowstyle='-', color='blue', lw=1))

plt.title('Network Analysis for Main Category 2022', fontsize=20)
plt.show()

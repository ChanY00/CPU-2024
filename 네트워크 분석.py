import pandas as pd
import networkx as nx
import operator
import numpy as np
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Excel 파일 읽기
    dataset = pd.read_excel('C:\workstation\CPU\특허유니버시아드_특허데이터_한국특허전략개발원')

    # 중심성 척도 계산을 위한 그래프 생성
    G_central = nx.Graph()

    # 빈도 5 이상인 데이터만 사용
    filtered_data = dataset[dataset['빈도'] >= 8]

    for _, row in filtered_data.iterrows():
        # 출원일과 디지털치료제 사이의 연결 추가
        G_central.add_edge(row['상세기술'], row['기술'], weight=int(row['빈도']))

    # 중심성 척도 계산
    dgr = nx.degree_centrality(G_central)
    btw = nx.betweenness_centrality(G_central)
    cls = nx.closeness_centrality(G_central)
    egv = nx.eigenvector_centrality(G_central)
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

    for _, row in filtered_data.iterrows():
        # 상세기술과 기술 사이의 연결 추가
        G.add_edge(row['상세기술'], row['기술'], weight=int(row['빈도']))


    # 노드의 색 지정
    node_colors = []
    for node in G.nodes():
        if node == 'thesis':
            node_colors.append('lightcoral')  # new drug development 노드는 lightcoral
        elif node == 'Patent':
            node_colors.append('cornflowerblue')  # digital therapy 노드는 cornflowerblue
        else:
            connected_to_new_drug_dev = False
            connected_to_digital_therapy = False
            for neighbor in G.neighbors(node):
                if neighbor == 'thesis':
                    connected_to_new_drug_dev = True
                elif neighbor == 'Patent':
                    connected_to_digital_therapy = True
            if connected_to_new_drug_dev and connected_to_digital_therapy:
                node_colors.append('orchid')  # 둘 다 연결된 노드는 orchid
            elif connected_to_new_drug_dev:
                node_colors.append('lightcoral')  # new drug development와 연결된 노드는 lightcoral
            elif connected_to_digital_therapy:
                node_colors.append('cornflowerblue')  # digital therapy와 연결된 노드는 cornflowerblue
            else:
                node_colors.append('grey')  # 둘 다와 연결되지 않은 노드는 회색

    # 노드 크기 조정
    sizes = [G.nodes[node]['nodesize'] * 10000 if node in ['thesis', 'Patent'] else G.nodes[node]['nodesize'] * 10000 for node in G]

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
    nx.draw(G, node_size=sizes, pos=nx.spring_layout(G, k=0.5, iterations=100), **options)
    ax = plt.gca()
    ax.collections[0].set_edgecolor('#555555')
    plt.show()

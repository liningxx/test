# -- coding: utf-8 --
# -- coding: utf-8 --
# -- coding: utf-8 --
# -- coding: utf-8 --
# f_r=open("data/Flickr_adj.txt","r")
# f_w=open("data/Flickr.edge","w")
# for i in range(195):
#     lines=f_r.readline()
#     print(lines)
#     adj=lines.split("\t")
#     for j in range(195):
#         if adj[j]=='1':
#             f_w.writelines(str(i)+"\t"+str(j))
#             f_w.writelines("\n")

# f_r= open("data/Flickr_feature.txt", "r")
# f_w = open("data/Flickr.attributes", "w")
# for i in range(195):
#     lines=f_r.readline()
#     print(len(lines.split("\t")))
#     f_w.writelines(str(i)+"\t"+lines)

# -- coding: utf-8 --
# -- coding: utf-8 --
# -- coding: utf-8 --
import networkx as nx

from util.graph_helper import load_graph
#
#
# def graph(data):
#     edge_file = open("data/{}.edge".format(data), 'r')
#     edges = edge_file.readlines()
#     # edge_num = int(edges[1].split('\t')[1].strip())
#     edges.pop(0)
#     edges.pop(0)
#
#     G = load_graph(edges)
#     return G
#
# def getStructureCentrality(data):
#     G=graph(data)
#
#     print(nx.shortest_path_length(G,314,377))
#
#
#
# if __name__ =='__main__':
#     getStructureCentrality('Flickr')
#
# import xlsxwriter
#
#
# # 创建sheet工作表
#
# workbook = xlsxwriter.Workbook('label.xlsx')
# worksheet = workbook.add_worksheet()
#
# f_r=open("data/Flickr.label","r")
# label=[0]*3490
# print(label)
# for i in range(3490):
#     lines=f_r.readline().strip("\n")
#     idx=int(lines[0])
#     label[i]=idx
# for i in range(3490):
#     worksheet.write(i,0,label[i]+1)
#
# workbook.close()


# -- coding: utf-8 --
# f_r=open("data\clusters.txt",'r')
# f_w=open("data\Flickr.label",'w')
# label=[0]*3491
# for i in range(10):
#     lines=f_r.readline()
#     label_idx=lines.strip("\n").split("\t")
#     for j in range(len(label_idx)):
#         idx=int(label_idx[j])
#         if idx==0:
#             print(i)
#         label[idx]=i+1
# for i in range(1,len(label)):
#     f_w.writelines(str(label[i])+"\n")

# f_r=open("data/content.txt","r")
# f_w=open("data/Flickr.attributes","w")
# for i in range(3490):
#     lines=f_r.readline()
#     f_w.writelines(str(i)+"\t"+lines)

# f_r=open("data/edge.txt","r")
# f_w=open("data/Flickr.edge","w")
# for i in range(30822):
#     lines=f_r.readline().strip("\n").split("\t")
#     print(lines)
#     source=int(lines[0])-1
#     target=int(lines[1])-1
#     f_w.writelines(str(source)+"\t"+str(target)+"\n")


# -- coding: utf-8 --
# -- coding: utf-8 --
# -- coding: utf-8 --
# -- coding: utf-8 --
import math
import sys
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from util.graph_helper import load_graph
from sklearn import metrics
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import accuracy_score, f1_score

np.set_printoptions(threshold=np.inf)


# 得到属性相关的矩阵
def get_attr_matrix(dataset):
    attribute_file = open("data/{}.attributes".format(dataset), 'r')
    attributes = attribute_file.readlines()
    node_num = int(attributes[0].split('\t')[1].strip())
    print(node_num)
    attribute_number = int(attributes[1].split('\t')[1].strip())
    attributes.pop(0)
    attributes.pop(0)

    attribute_matrix = [[0] * attribute_number for row in range(node_num)]
    for line in attributes:
        attr_line = line.strip('\n').split('\t')
        node_idx = int(attr_line[0])
        for i in range(attribute_number):
            attribute_matrix[node_idx][i] = int(attr_line[i + 1])
    return attribute_matrix


# 得到属性间的余弦距离
def getAttributeDistance(attr_matrix):
    min_distance, max_distance = sys.float_info.max, 0.0
    num = len(attr_matrix)
    attribute_distance = {}
    for i in range(len(attr_matrix)):
        for j in range(i + 1, len(attr_matrix)):
            try:
                attr_distance = np.dot(attr_matrix[i], attr_matrix[j]) / (
                        np.linalg.norm(attr_matrix[i]) * np.linalg.norm(attr_matrix[j]))  # 余弦距离
            # attr_distance = np.linalg.norm(np.array(attr_matrix[i]) - np.array(attr_matrix[j]))  # 欧式距离
            except RuntimeWarning:
                attr_distance=0
            attribute_distance[(i, j)] = attr_distance
            attribute_distance[(j, i)] = attr_distance
            min_distance = min(min_distance, attr_distance)
            max_distance = max(max_distance, attr_distance)

    for i in range(0, num):
        attribute_distance[(i, i)] = 0.0
    return attribute_distance, num, max_distance, min_distance


def auto_select_dc(distance, num, max_dis, min_dis):
    '''
    Auto select the dc so that the average number of neighbors is around 1 to 2 percent
    of the total number of points in the data set
    '''
    dc = (max_dis + min_dis) / 2
    while True:
        neighbor_percent = sum([1 for value in distance.values() if value < dc]) / num ** 2
        if neighbor_percent >= 0.01 and neighbor_percent <= 0.02:
            break
        if neighbor_percent < 0.01:
            min_dis = dc
        else:
            max_dis = dc
        dc = (max_dis + min_dis) / 2
        if max_dis - min_dis < 0.0001:
            break

    return dc


def local_density(distance, num, dc, gauss=False, cutoff=True):
    '''
    Compute all points' local density
    Return: local density vector of points that index from 1
    '''
    assert gauss and cutoff == False and gauss or cutoff == True
    gauss_func = lambda dij, dc: math.exp(- (dij / dc) ** 2)
    cutoff_func = lambda dij, dc: 1 if dij < dc else 0
    func = gauss_func if gauss else cutoff_func
    rho = [0] * num
    for i in range(num - 1):
        for j in range(i + 1, num):
            rho[i] += func(distance[(i, j)], dc)
            rho[j] += func(distance[(j, i)], dc)

    return rho


def min_distance(distance, num, max_dis, rho):
    '''
    Compute all points' min distance to a higher local density point
    Return: min distance vector, nearest neighbor vector
    '''
    sorted_rho_idx = np.argsort(-np.array(rho))
    delta = [max_dis] * num
    nearest_neighbor = [0] * num
    delta[sorted_rho_idx[0]] = -1.0
    for i in range(num):
        idx_i = sorted_rho_idx[i]
        for j in range(0, i):
            idx_j = sorted_rho_idx[j]
            if distance[(idx_i, idx_j)] < delta[idx_i]:
                delta[idx_i] = distance[(idx_i, idx_j)]
                nearest_neighbor[idx_i] = idx_j

    delta[sorted_rho_idx[0]] = max(delta)
    return delta


def density_and_distance(dataset, dc=None):
    attribute_matrix = get_attr_matrix(dataset)
    att_distance, num, max_dis, min_dis = getAttributeDistance(attribute_matrix)

    if dc == None:
        dc = auto_select_dc(att_distance, num, max_dis, min_dis)
    rho = local_density(att_distance, num, dc)
    delta = min_distance(att_distance, num, max_dis, rho)
    return rho, delta


def rho_delta(rho, delta):
    dpc = [1] * len(rho)
    for i in range(len(dpc)):
        dpc[i] = rho[i] * delta[i]

    attr_max = max(dpc)
    attr_min = min(dpc)

    for i in range(len(dpc)):
        dpc[i] = (dpc[i] - attr_min) / (attr_max - attr_min)
    return np.array(dpc, np.float32)


def getAttrPoints(dataset):
    rho, delta = density_and_distance(dataset)
    dpc_points = rho_delta(rho, delta)
    return np.array(dpc_points, np.int)


def graph(dataset):
    edge_file = open("data/{}.edge".format(dataset), 'r')
    edges = edge_file.readlines()
    node_num = int(edges[0].split('\t')[1].strip())
    edges.pop(0)
    edges.pop(0)

    G = load_graph(edges)
    return G, node_num


def getStructureCentrality(dataset):
    G, node_num = graph(dataset)

    # 获取pagerank的最大值和最小值
    pagerank = nx.pagerank(G)
    pagerank_max = max(pagerank.values())
    pagerank_min = min(pagerank.values())
    # pagerank归一化
    for key in pagerank.keys():
        pagerank[key] = (pagerank[key] - pagerank_min) / (pagerank_max - pagerank_min)

    # #获取eigenvector_centrality的最大值和最小值
    eigenvector = nx.eigenvector_centrality(G)
    eigenvector_max = max(eigenvector.values())
    eigenvector_min = min(eigenvector.values())
    # eigenvector归一化
    for key in eigenvector.keys():
        eigenvector[key] = (eigenvector[key] - eigenvector_min) / (eigenvector_max - eigenvector_min)

    # 求degree最大值
    degree = nx.degree_centrality(G)
    degree_max = max(degree.values())
    degree_min = min(degree.values())
    # degree归一化
    for key in degree.keys():
        degree[key] = (degree[key] - degree_min) / (degree_max - degree_min)

    stru_Centrality = [0] * node_num
    for i in range(node_num):
        try:
            stru_Centrality[i] = (1 / 3) * (pagerank[i] + eigenvector[i] + degree[i])
        except KeyError:
            stru_Centrality[i] = 0
    return stru_Centrality


def getRepresentativePoint(dataset, reprePoints_num,attribute_ratio):
    attr_dpc = getAttrPoints(dataset)
    print(len(attr_dpc))
    stru_Centrality = getStructureCentrality(dataset)
    print(len(stru_Centrality))

    stru_attr = {}
    for i in range(len(attr_dpc)):
        stru_attr[i] = (attribute_ratio) * attr_dpc[i] + (1-attribute_ratio)*stru_Centrality[i]

    value_order = sorted(stru_attr.items(), key=lambda x: x[1], reverse=True)
    representivePoints = [0] * reprePoints_num

    for i in range(reprePoints_num):
        representivePoints[i] = value_order[i][0]
    print(representivePoints)

    return np.array(representivePoints, np.int)


def get_str_attr_Similarity(dataset, n_clusters,repre_ratio,attribute_ratio):
    attr_matrix = get_attr_matrix(dataset)
    node_num = len(attr_matrix)
    reprePoints_num=int(repre_ratio*node_num)
    repre_points = getRepresentativePoint(dataset, reprePoints_num,attribute_ratio)
    print(repre_points)
    attr_Similarity = [[0.0 for i in range(len(repre_points))] for j in range(node_num)]

    for i in range(node_num):
        for j in range(len(repre_points)):
            # attr_Similarity[i][j] = np.linalg.norm(np.array(attr_matrix[i]) - np.array(attr_matrix[j]))  # 欧式距离
            attr_Similarity[i][j] = np.dot(attr_matrix[i], attr_matrix[j]) / (
                    np.linalg.norm(attr_matrix[i]) * np.linalg.norm(attr_matrix[j]))
    str_Similarity = [[0 for i in range(len(repre_points))] for j in range(node_num)]
    min_short_distance, max_short_distance = sys.float_info.max, 0.0
    G, num = graph(dataset)
    for i in range(node_num):
        for j in range(len(repre_points)):
            try:
                n = nx.shortest_path_length(G, i, repre_points[j])
            except nx.NetworkXNoPath:
                n = 15
            except nx.exception.NodeNotFound:
                n = 15
            str_Similarity[i][j] = n
            min_short_distance = min(min_short_distance, str_Similarity[i][j])
            max_short_distance = max(max_short_distance, str_Similarity[i][j])
    for i in range(node_num):
        for j in range(len(repre_points)):
            str_Similarity[i][j] = (str_Similarity[i][j] - min_short_distance) / (
                        max_short_distance - min_short_distance)

    attr_stru_Similarity = [[0.0 for i in range(len(repre_points))] for j in range(node_num)]

    for i in range(node_num):
        for j in range(len(repre_points)):

            attr_stru_Similarity[i][j] =(1-attribute_ratio)*str_Similarity[i][j]+(attribute_ratio) *attr_Similarity[i][j]
    X = np.array(attr_stru_Similarity)
    clustering1 = SpectralClustering(n_clusters, assign_labels="discretize", random_state=0).fit(X)

    label_file = open("data/{}.label".format(dataset), 'r')

    true_label = [] * node_num
    for line in label_file:
        true_label.append(int(line.strip('\n')))
    print("谱聚类：")
    print("NMI:", metrics.normalized_mutual_info_score(true_label, clustering1.labels_))
    print("ARI：", metrics.adjusted_rand_score(true_label, clustering1.labels_))
    # print('ACC',metrics.roc_curve(true_label, clustering1.labels_))



if __name__ == '__main__':
    print('0.1')
    get_str_attr_Similarity('Flickr',9 , 0.1, 0.5)
    print('0.2')
    get_str_attr_Similarity('Flickr',9 , 0.2, 0.5)
    print('0.3')
    # get_str_attr_Similarity('Flickr',9 , 0.3, 0.5)
    # print('0.4')
    # get_str_attr_Similarity('Flickr',9 , 0.4, 0.5)
    # print('0.5')
    # get_str_attr_Similarity('Flickr',9 , 0.5, 0.5)
    # print('0.6')
    # get_str_attr_Similarity('Flickr',9 , 0.6, 0.5)
    # print('0.7')
    # get_str_attr_Similarity('Flickr',9 , 0.7, 0.5)
    # print('0.8')
    # get_str_attr_Similarity('Flickr',9 , 0.8, 0.5)
    # print('0.9')
    # get_str_attr_Similarity('Flickr',9 , 0.9, 0.5)
    # print('1')
    # get_str_attr_Similarity('Flickr',9 , 1, 0.5)


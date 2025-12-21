import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import normalized_mutual_info_score,adjusted_mutual_info_score

def DataChoose(x):
    if x == '1':
        return 'klein.rds'
    elif x == '2':
        return 'lake.rds'
    elif x == '3':
        return 'romanov.rds'
    elif x == '4':
        return 'xin.rds'
    elif x == '5':
        return 'zeisel.rds'

def LabelChoose(x):
    if x == '1':
        return 'klein.rds_label'
    elif x == '2':
        return 'lake.rds_label'
    elif x == '3':
        return 'romanov.rds_label'
    elif x == '4':
        return 'xin.rds_label'
    elif x == '5':
        return 'zeisel.rds_label'


choose = '0'
while choose != 'q':
    choose = input("Waiting for selection:\n1.klein\n2.lake\n3.romanov\n4.xin\n5.zeisel\nq/Q to quit\n")
    if choose == 'q' or choose == 'Q':
        break
    data = pd.read_csv(DataChoose(choose), sep='\t')
    label = pd.read_csv(LabelChoose(choose), sep='\t')

    #数据降维
    data_T = data.T
    pca = PCA(0.95)
    pca.fit(data.T)
    X = pca.transform(data.T)

    #进行聚类
    km = KMeans(n_clusters=9,random_state=1)
    km.fit(X)
    centers = km.cluster_centers_
    # 预测
    y_pred = km.predict(X)

    #对比
    NMI = lambda x, y: normalized_mutual_info_score(x, y, average_method='arithmetic')
    AMI = lambda x, y: adjusted_mutual_info_score(x, y, average_method='arithmetic')

    label_1 = np.asarray(label)
    label_true = label_1[:,0]

    print(f"NMI={NMI(label_true, y_pred)}")
    print(f"AMI={AMI(label_true, y_pred)}")

    #可视化
    # 预测为同一簇的样本同颜色
    plt.scatter(X[:,0],X[:,1],c=y_pred)
    #标记聚类中心
    plt.scatter(centers[:,0], centers[:,1],color='white',marker='x', label='centers')
    plt.show()

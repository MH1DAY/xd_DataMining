from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

#决策树
def DecisionTree(x_train, x_test, y_train, y_test):
    #训练模型
    clf = DecisionTreeClassifier(
        criterion='gini',
        max_depth=5,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    )
    clf.fit(x_train, y_train)

    #预测准确率
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"决策树模型准确率: {accuracy:.2f}")

    #绘制决策树
    plt.figure(figsize=(12, 8))
    data.target_names = ['0','1','2']
    tree.plot_tree(clf,
                   feature_names=data.feature_names,
                   class_names=data.target_names,
                   filled=True)
    plt.show()


#KNN
def Knn(x_train, x_test, y_train, y_test):
    # 训练模型并预测
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train, y_train)
    #预测
    y_pred = knn.predict(x_test)
    print(f"KNN模型预测值:{y_pred} ")
    #准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"KNN模型准确率: {accuracy:.2f}")

#GNB
def Gnb(x_train, x_test, y_train, y_test):
    # 训练模型
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    # 预测
    y_pred = gnb.predict(x_test)
    print(f"GNB模型预测值:{y_pred} ")
    #准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"GNB模型准确率: {accuracy:.2f}")


#数据导入与划分
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data,
                                                    data.target,
                                                    test_size=0.3,
                                                    random_state=42,
                                                    stratify=data.target)

print(f"测试值:{y_test}")
Knn(X_train, X_test, y_train, y_test)
Gnb(X_train, X_test, y_train, y_test)
DecisionTree(X_train, X_test, y_train, y_test)

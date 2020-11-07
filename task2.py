import numpy as np
import pandas as pd
#from sklearn.pipeline import make_pipeline
#from collections import Counter
from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score ,roc_curve,auc
from sklearn.model_selection import GridSearchCV

"""
Task 2
실습 데이터 Wafer_data.csv를 이용하여 아래와 같이 분류 모형을 학습하고 결과를 비교하시오.
1) 성능 지표는 accuracy와 recall을 사용하시오.
2) 데이터의 순서를 무작위로 섞은 후, 5개의 부분으로 분할하는 K-fold 교차 검증을 통해 분
류 성능을 평가하시오.
3) 분류 모델은 Logistic regression을 이용하고, K-fold 교차 검증을 통해 얻은 평균 accuracy
와 recall 값에 대해 해석 하시오.
"""

def load_data(file_name):
    file_name_path = "./"+ file_name
    data = pd.read_csv(file_name_path, header=0)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return X, y, data

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False, random_state=0)
    print('train data shape', X_train.shape,'test data shape', X_test.shape)
    return X_train, X_test, y_train, y_test

def data_visualization(data):
    """
        데이터의 특성을 보기 위해 여러가지 정보를 print 및 plotting한다.
    :param data: 전체 데이터
    """
    print(data.describe())
    print(data.info())
    print(data.isnull().sum())

    describe_data = data.describe()
    describe_data.to_csv('wefer_des.csv')

    plt.title('X1-X5')
    aX1 = data['X1']
    aX2 = data['X2']
    aX3 = data['X3']
    aX4 = data['X4']
    aX5 = data['X5']
    plt.plot(aX1, label="X1")
    plt.plot(aX2, label="X2")
    plt.plot(aX3, label="X3")
    plt.plot(aX4, label="X4")
    plt.plot(aX5, label="X5")

    plt.legend()
    plt.show()

    plt.title('X51-X55')
    aX6 = data['X51']
    aX9 = data['X52']
    aX10 = data['X53']
    aX11 = data['X54']
    aX12 = data['X55']
    plt.plot(aX6, label="X51")
    plt.plot(aX9, label="X52")
    plt.plot(aX10, label="X53")
    plt.plot(aX11, label="X54")
    plt.plot(aX12, label="X55")
    plt.legend()
    plt.show()

    plt.title('X101-X105')
    aX13 = data['X101']
    aX15 = data['X102']
    aX16 = data['X103']
    aX17 = data['X104']
    aX18 = data['X105']
    plt.plot(aX13, label="X101")
    plt.plot(aX15, label="X102")
    plt.plot(aX16, label="X103")
    plt.plot(aX17, label="X104")
    plt.plot(aX18, label="X105")
    plt.legend()
    plt.show()

    plt.title('X148-X152')
    aX19 = data['X148']
    aX20 = data['X149']
    aX21 = data['X150']
    aX22 = data['X151']
    aX23 = data['X152']
    plt.plot(aX19, label="X148")
    plt.plot(aX20, label="X149")
    plt.plot(aX21, label="X150")
    plt.plot(aX22, label="X151")
    plt.plot(aX23, label="X152")
    plt.legend()
    plt.show()

    plt.title('Y')
    aY1 = data['Y']
    plt.scatter(aY1.index, aY1.values, label="Y")
    plt.legend()
    plt.show()

    Var_Corr = data.corr(method='pearson')
    sns.heatmap(Var_Corr)#, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns)#), annot=True)
    corr_with_d = data.corrwith(data['Y'])
    corr_with_d.sort_values(ascending=False, inplace=True)
    print(corr_with_d)
    print(corr_with_d[abs(corr_with_d) > 0.20].index)

def print_confusion_matrix(confusion_matrix, axes, class_label, class_names, fontsize=14):
    """
        confusion matrix를 입력으로 받아 heatmap을 그린다.
    :param axes: figure의 axis의 위치
    :param class_label: plot title로 사용
    :param class_names: class name
    :return:
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )

    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    axes.set_xlabel('True label')
    axes.set_ylabel('Predicted label')
    axes.set_title(class_label)

def buid_logistic(X_train, y_train, X_test, y_test ):
    """
        logistic regression모델을 만들고 acc, recall, f1점수를 출력한다.
    """
    clf = LogisticRegression(random_state=0, max_iter=500)
    clf.fit(X_train, y_train)
    clf_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, clf_pred)
    recall = recall_score(y_test, clf_pred)
    f1 = f1_score(y_test, clf_pred)
    print(" No KFold Acc: ", acc, "\t Recall: ", recall, "\t F1: ", f1)
    return clf

def build_grid_search_logsitic(X_train, y_train):
    """
        grid search를 통해 최상의 logistic regression모델을 만든다.
    """
    param_grid = {'C': np.arange(1e-05, 3, 0.1)}
    scoring = {'Accuracy': 'accuracy', 'AUC': 'roc_auc', 'Log_loss': 'neg_log_loss'}

    gs = GridSearchCV(LogisticRegression(max_iter=500), return_train_score=True,
                      param_grid=param_grid, scoring=scoring, cv=5, refit='Accuracy')

    gs.fit(X_train, y_train)
    results = gs.cv_results_

    print('=' * 20)
    print("best params: " + str(gs.best_estimator_))
    print("best params: " + str(gs.best_params_))
    print('best score:', gs.best_score_)
    print('=' * 20)
    return gs

def buid_logistic_kfold(X_train, y_train):
    """
        kfold를 사용하여 logistic regression 모델을 훈련한다.
    """
    kfold = KFold(n_splits=5, random_state=0, shuffle=True)

    clf_fold = LogisticRegression(random_state=0, max_iter=500)
    X = np.array(X_train)
    y = np.array(y_train)

    i = 0
    for train_index, validate_index in kfold.split(X):
        X_k_train, X_validate = X[train_index], X[validate_index]
        y_k_train, y_validate = y[train_index], y[validate_index]
        clf_fold.fit(X_k_train, y_k_train)


        clf_pred = clf_fold.predict(X_validate)
        acc = accuracy_score(y_validate, clf_pred)
        recall = recall_score(y_validate, clf_pred)
        f1 = f1_score(y_validate, clf_pred)
        print("Acc: ", acc, "\t Recall: ", recall, "\t F1: ", f1)
        cfs_matrix = confusion_matrix(y_validate, clf_fold.predict(X_validate))
        if i == 0:
            ax_v = ax[0, 1]
        elif i == 1:
            ax_v = ax[0, 2]
        elif i == 2:
            ax_v = ax[0, 3]
        elif i == 3:
            ax_v = ax[1, 0]
        elif i == 4:
            ax_v = ax[1, 1]
        print_confusion_matrix(cfs_matrix, ax_v, "KFOLD {0}".format(i + 1), ["0", "1"])
        i += 1
    return clf_fold


if __name__ == "__main__":
    X, y, data = load_data("Wafer_data.csv")

    data_visualization(data)
    X_train, X_test, y_train, y_test = split_data(X, y)

    fig, ax = plt.subplots(2, 4, figsize=(12, 7))

    # Data를 split하여 logistic regression을 훈련한다.
    clf = buid_logistic(X_train, y_train, X_test, y_test )
    cfs_matrix = confusion_matrix(y_test, clf.predict(X_test))
    print_confusion_matrix(cfs_matrix, ax[0,0], "Split Data", ["0", "1"])

    # Data를 kfold로 나누어서 logistic regression을 훈련한다.
    clf_fold = buid_logistic_kfold(X_train, y_train)
    clf_pred = clf_fold.predict(X_test)
    acc = accuracy_score(y_test, clf_pred)
    recall = recall_score(y_test, clf_pred)
    f1 = f1_score(y_test, clf_pred)
    print(" KFold un seen Acc: ", acc, "\t Recall: ", recall, "\t F1: ", f1)
    cfs_matrix = confusion_matrix(y_test, clf_fold.predict(X_test))
    print_confusion_matrix(cfs_matrix, ax[1,2], "KFOLD(UNSEEN)", ["0", "1"])

    # grid search를 통해 좋은 logistic regression 모델을 찾는다.
    gs = build_grid_search_logsitic(X_train, y_train)
    clf_pred = gs.best_estimator_.predict(X_test)
    acc = accuracy_score(y_test, clf_pred)
    recall = recall_score(y_test, clf_pred)
    f1 = f1_score(y_test, clf_pred)
    print(" KFold with grid un seen Acc: ", acc, "\t Recall: ", recall, "\t F1: ", f1)

    # 상관성계수가 높은 변수를 뽑아 logistic regrssion을 훈련한다.
    X, y, data = load_data("Wafer_data.csv")
    corr_value = [0.2, 0.15, 0.1, 0.05]
    corr_with_d = data.corrwith(data['Y'])
    corr_with_d.sort_values(ascending=False, inplace=True)
    for c_val in corr_value:
        column_index = list(corr_with_d[abs(corr_with_d) > c_val].index)
        if 'Y' in column_index:
            column_index.remove('Y')
        X_s = X[column_index]
        X_train, X_test, y_train, y_test = split_data(X_s, y)
        clf_fe = LogisticRegression(random_state=0, max_iter=500)
        clf_fe.fit(X_train, y_train)
        clf_pred = clf_fe.predict(X_test)
        acc = accuracy_score(y_test, clf_pred)
        recall = recall_score(y_test, clf_pred)
        f1 = f1_score(y_test, clf_pred)
        print(" corellation :",c_val, "column cnt", len(column_index)," Acc: ", acc, "\t Recall: ", recall, "\t F1: ", f1)


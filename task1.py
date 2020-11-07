import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import seaborn as sns

"""
Task 1
실습 데이터 Chem_data.csv를 이용하여 아래와 같이 회귀 모형을 학습하고 결과를 비교하시오.
1) 성능 지표는 MSE (Mean Squared Error)를 사용하시오.
2) 데이터의 순서를 유지하여 train / test set을 7:3으로 분할하시오.
3) Train set을 이용하여 다중선형회귀를 모형을 학습하고, test data를 이용하여 모델 성능을
계산하시오.
4) 동일한 방법으로 Ridge와 Lasso를 학습한 후 성능을 비교하시오. 학습 시 모델 파라미터
alpha는 임의의 다른 3개의 값을 사용하고 (예: 1, 0.1, 0.01), 모델 파라미터에 따른 모델
성능을 비교하시오.
"""

def load_data(file_name):
    file_name_path = "./"+ file_name
    data = pd.read_csv(file_name_path, header=0)
    X = data.iloc[:, :-2]
    y1 = data.iloc[:, -2]
    y2 = data.iloc[:, -1]
    return X, y1, y2, data

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False, random_state=0)
    print('train data shape', X_train.shape,'test data shape', X_test.shape)
    return X_train, X_test, y_train, y_test

def build_linear_regression( X_train, X_test, y_train, y_test):
    """
        linear regression모델을 만들고, 주어진 데이터에 따라 훈련한다.
        테스트데이터로  r2, mse를 구한
    :param X_train: x 훈련데이터
    :param X_test:  x 데이스데이터
    :param y_train: y 훈련데이터
    :param y_test: y 테스트 데이
    :return:
        Plotting하기 위해 정보를 넘긴다.
        linear_pred : prdiction result
        linear : linear model
        mse : mse
    """
    linear = LinearRegression()
    linear.fit(X_train, y_train)
    linear_pred = linear.predict(X_test)
    r2 = r2_score(y_test, linear_pred)
    mse = mean_squared_error(y_test, linear_pred)
    print("Linear regresstion R2: ",r2, "\t MSE: ", mse)
    return linear_pred, linear, mse

def build_ridge(X_train, X_test, y_train, y_test, alpha):
    """
         Ridge regression모델을 만들고, 주어진 데이터에 따라 훈련한다.
         테스트데이터로  r2, mse를 구한다
    """
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)
    r2 = r2_score(y_test, ridge_pred)
    mse = mean_squared_error(y_test, ridge_pred)
    print("ridge R2 alpha (" + str(alpha) + ") : ", r2, "\t MSE: ", mse)
    return ridge_pred, ridge, mse

def build_lasso(X_train, X_test, y_train, y_test, alpha):
    """
         lasso regression모델을 만들고, 주어진 데이터에 따라 훈련한다.
         테스트데이터로  r2, mse를 구한다
    """
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    lasso_pred = lasso.predict(X_test)
    r2 = r2_score(y_test, lasso_pred)
    mse = mean_squared_error(y_test, lasso_pred)
    print("Lasso R2 alpha (" + str(alpha) + ") : ", r2, "\t MSE: ", mse)
    return lasso_pred, lasso, mse

def plot_result(linear_pred, ridge_pred, lasso_pred ):
    """
        linear, ridge lasso모델의 결과를 받아 plotting을 한다

    :param linear_pred: linear prediction 결과
    :param ridge_pred: ridge prediction 결과
    :param lasso_pred: lasso prediction 결과
    :return:
    """
    f, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(21, 6))
    ax1.plot(range(len(y_test)), y_test, label='true')
    ax1.plot(range(len(y_test)), linear_pred, label='prediction')
    ax1.set_title('Linear Model')
    ax1.legend()
    ax2.plot(range(len(y_test)), y_test, label='true')
    ax2.plot(range(len(y_test)), ridge_pred[0], label='prediction')
    ax2.set_title('Ridge Model alpha(1)')
    ax2.legend()
    ax3.plot(range(len(y_test)), y_test, label='true')
    ax3.plot(range(len(y_test)), ridge_pred[1], label='prediction')
    ax3.set_title('Ridge Model alpha(0.1)')
    ax3.legend()
    ax4.plot(range(len(y_test)), y_test, label='true')
    ax4.plot(range(len(y_test)), ridge_pred[2], label='prediction')
    ax4.set_title('Ridge Model alpha(0.01)')
    ax4.legend()
    ax5.plot(range(len(y_test)), y_test, label='true')
    ax5.plot(range(len(y_test)), lasso_pred[0], label='prediction')
    ax5.set_title('Lasso Model alpha (1)')
    ax5.legend()
    ax6.plot(range(len(y_test)), y_test, label='true')
    ax6.plot(range(len(y_test)), lasso_pred[1], label='prediction')
    ax6.set_title('Lasso Model alpha(0.1)')
    ax6.legend()
    ax7.plot(range(len(y_test)), y_test, label='true')
    ax7.plot(range(len(y_test)), lasso_pred[2], label='prediction')
    ax7.set_title('Lasso Model alpha(0.01)')
    ax7.legend()
    ax8.plot(range(len(y_test)), y_test, label='true')
    ax8.plot(range(len(y_test)), lasso_pred[3], label='prediction')
    ax8.set_title('Lasso Model alpha(0.001)')
    ax8.legend()
    plt.show()

def data_visualization(data):
    """
        데이터의 특성을 보기 위해 여러가지 정보를 print 및 plotting한다.
    :param data: 전체 데이터
    """
    print(data.describe())
    print(data.info())
    print(data.isnull().sum())

    describe_data = data.describe()
    describe_data.to_csv('chem_des.csv')

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

    plt.title('X6-X12')
    aX6 = data['X6']
    aX9 = data['X9']
    aX10 = data['X10']
    aX11 = data['X11']
    aX12 = data['X12']
    plt.plot(aX6, label="X6")
    plt.plot(aX9, label="X9")
    plt.plot(aX10, label="X10")
    plt.plot(aX11, label="X11")
    plt.plot(aX12, label="X12")
    plt.legend()
    plt.show()

    plt.title('X13-X18')
    aX13 = data['X13']
    aX15 = data['X15']
    aX16 = data['X16']
    aX17 = data['X17']
    aX18 = data['X18']
    plt.plot(aX13, label="X13")
    plt.plot(aX15, label="X15")
    plt.plot(aX16, label="X16")
    plt.plot(aX17, label="X17")
    plt.plot(aX18, label="X18")
    plt.legend()
    plt.show()

    plt.title('X19-X23')
    aX19 = data['X19']
    aX20 = data['X20']
    aX21 = data['X21']
    aX22 = data['X22']
    aX23 = data['X23']
    plt.plot(aX19, label="X19")
    plt.plot(aX20, label="X20")
    plt.plot(aX21, label="X21")
    plt.plot(aX22, label="X22")
    plt.plot(aX23, label="X23")
    plt.legend()
    plt.show()

    plt.title('X24-X27')
    aX24 = data['X24']
    aX25 = data['X25']
    aX26 = data['X26']
    aX27 = data['X27']
    plt.plot(aX24, label="X24")
    plt.plot(aX25, label="X25")
    plt.plot(aX26, label="X26")
    plt.plot(aX27, label="X27")
    plt.legend()
    plt.show()

    plt.title('Y1-Y2')
    aY1 = data['Y1']
    aY2 = data['Y2']
    plt.plot(aY1, label="Y1")
    plt.plot(aY2, label="Y2")

    plt.legend()
    plt.show()

    Var_Corr = data.corr(method='pearson')
    sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns)#), annot=True)


def coeff(lm, name):
    """
        모델 훈련후 나온 상관계수를 높은순서대로 출력한다.
    """
    X_Cols = ['X1','X2','X3','X4','X5','X6','X9','X10','X11','X12','X13','X15','X16','X17'
        ,'X18','X19','X20','X21','X22','X23','X24','X25','X26','X27','Y1','Y2']

    print(X_Cols)
    coefs = pd.DataFrame(zip(X_Cols, lm.coef_), columns=['features', 'coefficients'])

    print(name)
    print(coefs.reindex(coefs.coefficients.abs().sort_values(ascending=False).index))

if __name__ == "__main__":
    X, y1, y2, data = load_data("Chem_data.csv")

    data_visualization(data)

    X_train, X_test, y_train, y_test = split_data(X, y1)
    linear_pred, lm_l, mse_l = build_linear_regression(X_train, X_test, y_train, y_test)
    alpha = [1, 0.1, 0.01, 0.001, 0.0001]

    #alpha별 모델을 훈련해서 정보를 저장한다
    ridge_all = [(build_ridge(X_train, X_test, y_train, y_test, a)) for a in alpha]
    lasso_all = [(build_lasso(X_train, X_test, y_train, y_test, a)) for a in alpha]
    ridge_mse = [mse[2] for mse in ridge_all]
    lasso_mse = [mse[2] for mse in lasso_all]
    ridge_pred = [mse[0] for mse in ridge_all]
    lasso_pred = [mse[0] for mse in lasso_all]
    ridge_m = [mse[1] for mse in ridge_all]

    # alpha별 mse변화를 그린다.
    fig, ax1  = plt.subplots(1, 1)
    alpha_x = [str(a) for a in alpha]
    linear_mse = [mse_l for _ in alpha]
    ax1.plot(alpha_x , ridge_mse, '--', label="Ridge", linewidth=4)
    ax1.plot(alpha_x , lasso_mse, label="Lasso")
    ax1.plot(alpha_x , linear_mse, label="linear")
    ax1.set_ylabel("mse")
    ax1.set_xlabel("alpha")
    plt.legend(labels=['Ridge', 'Lasso', 'linear'])
    fig.tight_layout()
    plt.show()

    plot_result(linear_pred, ridge_pred, lasso_pred )
    coeff(lm_l, "Linear")
    coeff(ridge_m[0], "ridge(1)")
    coeff(ridge_m[1], "ridge(0.1)")
    coeff(ridge_m[2], "ridge(0.01)")

    # 상관계수 높은것만 뽑아서 훈
    X, y1, y2, data = load_data("Chem_data.csv")
    corr_value = [0.3, 0.4, 0.5, 0.6, 0.8]
    corr_with_d = data.corrwith(data['Y1'])
    corr_with_d.sort_values(ascending=False, inplace=True)
    for c_val in corr_value:
        #lasso_all = [(build_lasso(X_train, X_test, y_train, y_test, a)) for a in alpha]
        column_index = list(corr_with_d[abs(corr_with_d) > c_val].index)
        if 'Y1' in column_index:
            column_index.remove('Y1')
        if 'Y2' in column_index:
            column_index.remove('Y2')
        X_s = X[column_index]
        X_train, X_test, y_train, y_test = split_data(X_s, y1)
        print( "## correllation (> {0}, [{1}])".format(c_val, len(column_index)))
        ridge_all = [(build_ridge(X_train, X_test, y_train, y_test, a)) for a in alpha]




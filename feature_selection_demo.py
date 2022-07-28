import pandas as pd
from scipy.stats import pearsonr,spearmanr
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

def var_thr():
    data=pd.read_csv("./data/factor_returns.csv")
    # print(data)
    transfer=VarianceThreshold(threshold=10)    
    trans_data=transfer.fit_transform(data.iloc[:,1:10])
    print("\n",data.iloc[:,1:10].shape)
    print("\n",trans_data.shape)
    print(trans_data)
    pass


def pca_demo():
    data=[[2,8,4,5], [6,3,0,8], [5,4,9,1]]
    transfer=PCA(n_components=2)
    data1=transfer.fit_transform(data)
    print(data1)
    transfer2=PCA(n_components=0.3)
    data2=transfer2.fit_transform(data)
    print(data2)
    pass

if __name__ == '__main__':
    #var_thr()

    # x1 = [12.5, 15.3, 23.2, 26.4, 33.5, 34.4, 39.4, 45.2, 55.4, 60.9]
    # x2 = [21.2, 23.9, 32.9, 34.1, 42.5, 43.2, 49.0, 52.8, 59.4, 63.5]
    # print(pearsonr(x1,x2))
    # print(spearmanr(x1,x2))
    pca_demo()
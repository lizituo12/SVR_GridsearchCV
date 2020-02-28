import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.externals import joblib

''''导入数据集'''
def df_data(filepath1,filepath2):
    df2=pd.read_excel(filepath1,index_col=0)#导入自己的数据即可。方式可能有变化，具体查询pandas.
    df1=pd.read_excel(filepath2,index_col=0)
    df=pd.concat([df2,df1],axis=0)
    #df=pd.read_excel('d:/xinxi/fubu3.xlsx')
    return df
   
#''''''''''''''''''''''''
#'''PCA主成分分析'''
#''''''''''''''''''''''''
def pca_data(df,n):
    x=df.iloc[:,:-1]
    pca=PCA(n_components=n)
    x=pd.DataFrame(pca.fit_transform(x))
    return x
''''''''''''''''''''''''
'''检测相关性'''
''''''''''''''''''''''''
def corr(df,thre):
    corr=[]
    corr1=[]
    for i in range(df.shape[1]-1):
        corr.append(np.corrcoef(df.iloc[:,i],df.iloc[:,-1])[1][0])

    for i in corr:
        if abs(i)>thre:
            corr1.append(corr.index(i))
    return corr1
''''''''''''''''''''''''
'''合并数据集'''
''''''''''''''''''''''''
def concat(df):
    x=df.iloc[:,corr1]
    y=df.iloc[:,-1].tolist()
    y=[y[i]/17.1for i in range(len(y))]
    y=pd.DataFrame(y)
    x=x.reset_index(drop=True)
    y=y.reset_index(drop=True)
    X=pd.concat([x,y],axis=1)
    return X#觉得train_test_split 函数需要先固定好数据，因此用concat重新组合一下。
''''''''''''''''''''''''
'''设置训练集测试集'''
''''''''''''''''''''''''
def Train_test_split(X)
    train,test=train_test_split(X,test_size=0.3)
    train_x=train.iloc[:,:-1]
    train_y=train.iloc[:,-1]
    test_x=test.iloc[:,:-1]
    test_y=test.iloc[:,-1].to_list()
    return train_x,train_y,test_x,test_y

'''''''''''''''model select'''''''''''''''
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
'''''**************'''''''''
'''SVR GridSearchCV'''

'''''***************'''''''''
model=SVR()
param_grid=[{
            'kernel':['rbf','linear','poly','precomputed'],
            'degree':[3,4,5],
            'C':[1.0,1.5,2.0],
            'epsilon':[0.1,0.2,0.23,0.25],
            'shrinking':[True,False],
            'verbose':[True,False]
            }]
print('SVR start search!')

'''''**********************'''''''''
'''RandomForestRegression GridSearchCV'''

'''''**********************'''''''''
#param_grid=[{'n_estimators':[200,300,400],
#            'max_features':['auto',"sqrt","log2"],
#             'verbose':[0,1,2,3],'criterion':['mse','mae'],
#             'oob_score':['True','False'],
#             'warm_start':['True','False']
#             }]
#print('start search!')
#model=RandomForestRegressor(n_jobs=-1)


'''''''''''''''''start search'''''''''''''''''''''
grid=GridSearchCV(model,param_grid=param_grid,cv=4)
grid.fit(train_x,train_y)
print('grid_best_params:',  grid.best_params_)

print('grid.best_score_:', grid.best_score_)
'''''''''''''''''''model trainng'''''''''''''''''''
#model=RandomForestRegressor(n_estimators=300,oob_score=True,warm_start=False,criterion='mae',max_features='auto')
#model.fit(train_x,train_y)
#pre=model.predict(test_x)
#from matplotlib import pyplot as plt
#plt.scatter(test_y,pre)  
#print('test_x,test_y accuracy:',model.score(test_x,test_y))
#print('train_x,train_y accuracy:',model.score(train_x,train_y))
#print('all data accuracy',model.score(x,y))
#result.append(model.score(test_x,test_y))

#result=[]
#for i in range(10):
#    model.fit(train_x,train_y)
#    pre=model.predict(test_x)
##    plt.scatter(test_y,pre)  
#    print('test_x,test_y accuracy:',model.score(test_x,test_y))
#    print('train_x,train_y accuracy:',model.score(train_x,train_y))
#    print('all data accuracy',model.score(x,y))

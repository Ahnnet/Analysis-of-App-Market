#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler, RobustScaler,StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


# In[2]:


def preprocessing():
    df = pd.read_csv("E:\VScodeWorkSheet\datasets\Google-Playstore.csv")
    df=df.sample(frac=0.05,replace=True,random_state=1)
    #print(df.head())

    for col in df.columns:
        col1 = col.replace(' ','')
        df = df.rename(columns={col:col1})
    #print(df.columns)

    #print("Dropping the following columns - AppId, DeveloperWebsite, DeveloperEmail, PrivacyPolicy, Currency, DeveloperId, ScrapedTime, MinimumAndroid")
    df = df.drop(['AppId','DeveloperWebsite','DeveloperEmail','PrivacyPolicy','Currency','DeveloperId','ScrapedTime','MinimumAndroid'],axis=1)
    #print(df.head())

    #print("Dataset information",df.info())

    # In[9]
    cols = df.columns[df.isnull().any()].to_list()
    #print("Columns having null values are :",cols)

    #for c in cols:
    #    print(c,type(c),": ",df[c].isnull().sum())

    df.dropna(subset=['Size','MinimumInstalls','Installs','AppName'],inplace=True)
    df.dropna(subset=['AppName'],inplace=True)

    # In[11]
    df['Rating']  = df['Rating'].astype(float)
    avg = round(df['Rating'].mean(),1)
    df['Rating'].fillna(avg,inplace=True)

    df['RatingCount']  = df['RatingCount'].astype(float)
    avg = round(df['RatingCount'].mean(),1)
    df['RatingCount'].fillna(avg,inplace=True)

    #print(df['ContentRating'].value_counts())


    # In[13]
    df['ContentRating'] = df['ContentRating'].replace('Unrated',"Everyone")

    #Cleaning other values just to include Everyone, Teens and Adult 

    df['ContentRating'] = df['ContentRating'].replace('Mature 17+',"Adults")
    df['ContentRating'] = df['ContentRating'].replace('Adults only 18+',"Adults")
    df['ContentRating'] = df['ContentRating'].replace('Everyone 10+',"Everyone")

    # In[14]
    # CLeaning the Installs column so as to convert it into numeric
    df.Installs = df.Installs.str.replace(',','')
    df.Installs = df.Installs.str.replace('+','')
    df.Installs = df.Installs.str.replace('Free','0')
    df['Installs'] = pd.to_numeric(df['Installs'])

    # In[15]
    df['PriceRange'] = pd.cut(df['Price'],bins=[0,0.19,9.99,29.99,410],labels=['Free','Low','Mid','High'],include_lowest=True)
    #dummies = pd.get_dummies(df['PriceRange'],prefix='Price')
    #df = df.join(dummies)
    #print(df['PriceRange'].value_counts())

    # In[16]
    #print(df.Free.value_counts())
    #print("Apps that have Price = 0, have Free column True")
    df.loc[(df.Price==0) & (df.Free==False),'Free'] = True
    #print(df.Free.value_counts())

    df['Type'] = np.where(df['Free'] == True,'Free','Paid')
    df.drop(['Free'],inplace=True,axis=1)

    df['RatingType'] = 'NoRating'
    df.loc[(df['RatingCount'] > 0) & (df['RatingCount'] <= 10000.0),'RatingType'] = 'Less than 10K'
    df.loc[(df['RatingCount'] > 10000) & (df['RatingCount'] <= 500000.0),'RatingType'] = 'Between 10K and 500K'
    df.loc[(df['RatingCount'] > 500000) & (df['RatingCount'] <= 138557570.0),'RatingType'] = 'More than 500K'
    #print(df.RatingType.value_counts())

    columns = ['Category', 'Type', 'ContentRating', 'AdSupported', 'EditorsChoice', 'InAppPurchases', 'RatingType', 'PriceRange']
    df = encoding(df, columns)

    return df


# In[3]:


def encoding(df, columns):
    data = df.copy()
    labelEncoder = LabelEncoder()

    for col in columns:
        labelEncoder.fit(data[col])
        data[col] = labelEncoder.transform(data[col])

    return data


# In[4]:


def scailing(X):
    standard_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()
    robust_scaler = RobustScaler()

    X_std = standard_scaler.fit_transform(X)
    X_mm = minmax_scaler.fit_transform(X)
    X_rb = robust_scaler.fit_transform(X)

    return X_std, X_mm, X_rb


# In[5]:


df = preprocessing()


# In[6]:


X = df.drop(['AppName','Size', 'MinimumInstalls', 'Released','RatingCount' ,'Type','MaximumInstalls','Price','LastUpdated','Rating','RatingType'],axis=1)
y = df['RatingType'].values


# In[8]:


def linearRegression(X_train, X_test, y_train, y_test):
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    reg_acc = round(reg.score(X_test, y_test)*100, 3)

    return y_pred, reg_acc


# In[9]:


def decisionTree(X_train, X_test, y_train, y_test):
    dcTree = DecisionTreeClassifier()
    dcTree.fit(X_train, y_train)
    y_pred = dcTree.predict(X_test)
    dcTree_acc = round(dcTree.score(X_test, y_test)*100, 3)

    return y_pred, dcTree_acc


# In[10]:


def modeling(X_train, X_test, y_train, y_test):
    lin = list()
    tree = list()

    lin.append(linearRegression(X_train, X_test, y_train, y_test))
    tree.append(decisionTree(X_train, X_test, y_train, y_test))

    return lin, tree


# In[11]:


def func_kfold(X, y, k):
    n = list()
    for i in range(0, k):
        n.append(i)
    i=0
    
    standard_scaler = StandardScaler()
    X = standard_scaler.fit_transform(X)

    # prepare cross validation
    kf = KFold(n_splits=k, shuffle=True, random_state=1)

    pred_n_acc = dict()
    # enumerate splits
    for train, test in kf.split(X):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]

        lin, tree = modeling(X_train, X_test, y_train, y_test)
        lin = {'lin':[{'y_pred':lin[0][0]}, {'acc':lin[0][1]}, {'y_test':y_test}]}
        tree = {'tree':[{'y_pred':tree[0][0]}, {'acc':tree[0][1]}, {'y_test':y_test}]}
        
        pred_n_acc[n[i]] = [dict(lin), dict(tree)]
        i += 1
    return pred_n_acc


# In[12]:


def best_score():
    case = dict()
    scores = list()
    y_preds = list()
    y_tests = list()
    models = ['lin', 'tree']
    nk = [3, 5, 10]

    for k in nk:
        case['k='+str(k)]=func_kfold(np.array(X), y, k)
        for i in range(0, k):
            for n in range(0, len(case['k='+str(k)][i])):
                for model in models:
                    if model in case['k='+str(k)][i][n]:
                        y_preds.append(case['k='+str(k)][i][n][model][0]['y_pred'])
                        scores.append(case['k='+str(k)][i][n][model][1]['acc'])
                        y_tests.append(case['k='+str(k)][i][n][model][2]['y_test'])
                    elif model in case['k='+str(k)][i][n]:
                        y_preds.append(case['k='+str(k)][i][n][model][0]['y_pred'])
                        scores.append(case['k='+str(k)][i][n][model][1]['acc'])
                        y_tests.append(case['k='+str(k)][i][n][model][2]['y_test'])

    bestScore = max(scores)
    index = scores.index(bestScore)
    bestPredict = y_preds[index]
    y_test = y_tests[index]
    model = str()
    k = int()

    if index % 2 == 0:
        model = 'Linear Regression'
    elif index % 2 == 1:
        model = 'DecisionTreeClassifier'

    if len(scores) - len(models)*nk[0] <= 0:
        kNumber = 'k=3'
    elif len(scores) - len(models)*nk[0] - len(models)*nk[1] <= 0:
        kNumber = 'k=5'
    elif len(scores) - len(models)*nk[0] - len(models)*nk[1] - len(models)*nk[2]<= 0:
        kNumber = 'k=10'

    return bestScore, bestPredict, y_test, model, kNumber


# In[13]:


feature_name2 = ['Category', 'Type', 'ContentRating', 'AdSupported', 'EditorsChoice', 'InAppPurchases', 'PriceRange']
class_name2 = ['NoRating', 'Less than 10K', 'Between 10K and 500K', 'More than 500K']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=20)
dcTree = DecisionTreeClassifier()
dcTree.fit(X_train, y_train)
y_pred = dcTree.predict(X_test)
dcTree_acc = round(dcTree.score(X_test, y_test)*100, 3)


# In[14]:


from sklearn.tree import export_graphviz
export_graphviz(dcTree, out_file="tree.dot",class_names=class_name2, feature_names=feature_name2, impurity=True, filled=True)


# In[15]:


import graphviz
with open("tree.dot") as f:
    dot_graph=f.read()
src=graphviz.Source(dot_graph)
src


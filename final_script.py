import numpy as np
from numpy.core.fromnumeric import std
import pandas as pd
import csv
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, calinski_harabasz_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import RidgeCV
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import preprocessing


def getUsers1(final):
    compidx = [i for i in range(len(final[3])) if final[3][i] >= 0.95]
    userVal = []
    rep = 0
    
    for ind1 in compidx:
        user = final[0][ind1]
        for ind2 in compidx:
            if (user == final[0][ind2]):
                rep += 1
        if(rep >= 5):
            if user not in userVal:
                userVal.append(user)
        rep = 0
    return userVal

def getData(filename):
    
    token = open(filename)
    lines = token.readlines()[1:]
    token.close
    final = [[x.split()[i] for x in lines] for i in range(0,12)]

    for i in range(1,12):
        for j in range(len(final[i])):
            final[i][j] = float(final[i][j])

    userEntries = {}
    for i in range(len(final[0])):
        if (final[0][i] not in userEntries.keys()):
            userEntries[final[0][i]] = []
        templist = []
        for j in range(2,12):
            templist.append(final[j][i])
        userEntries[final[0][i]].append(templist)
    return userEntries


def validEntries1(entries):
    newDict = {}
    counter = 0
    for key, val in entries.items():
        for i in range(len(val)):
            if(val[i][2] >= 0.98):
                counter += 1
        if (counter >= 5):
            newDict[key] = val
        counter = 0
    
    return newDict

def getMatrix1(entries):
    matrix = np.empty((len(entries),7),dtype=float)
    i = 0
    for key,val in entries.items():
        tempArray = np.array(val)
        tempArray = tempArray[:,[0,1,3,4,5,7,8]]
        tempMeans = np.mean(tempArray,axis=0)
        matrix[i] = tempMeans
        i += 1
    return matrix
    
def kMeans(X):
    distortions = []
    CHscores = []
    for i in range(2,10):
        kMeansObj = KMeans(n_clusters=i,random_state=1)
        kMeansObj.fit(X)
        distortions.append(kMeansObj.inertia_)
        CHscores.append(calinski_harabasz_score(X,kMeansObj.labels_))

    plt.plot(range(2,10),CHscores, 'bx-')
    plt.title('Calinski-Harabasz Scores')
    plt.xlabel('number of clusters')
    plt.ylabel('Calinski-Harabasz Index')
    plt.show()

    plt.plot(range(2,10),distortions, 'bx-')
    plt.title('Distortions')
    plt.xlabel('number of clusters')
    plt.ylabel('sum of square distances')
    plt.show()

    return distortions
    

def validEntries2(entries):
    newDict = {}
    for key,val in entries.items():
        if(len(entries[key])>46):
            newDict[key] = val
    return newDict

def getMatrix2(entries):
    tempkey = next(iter(entries))
    matrix = np.empty((len(entries),8),dtype=float)
    i = 0
    for key,val in entries.items():
        tempArray = np.array(val)
        tempArray = tempArray[:,[0,1,3,4,5,7,8,9]]
        tempMeans = np.mean(tempArray,axis=0)
        matrix[i] = tempMeans
        i += 1
    
    return matrix


def bestModel(X, y):
    a = np.logspace(start=1, stop=2, num=100) 
    ridge_model = RidgeCV(alphas=a, store_cv_values=True, normalize=True)
    ridge_model.fit(X, y)
    print(f'score for ridge regression model: {ridge_model.score(X, y)}')
    print(f'coefficients for ridge regression model{ridge_model.coef_}')
    print(f'intercept for ridge regression model {ridge_model.intercept_}')
    
    return ridge_model

def getMatrix3(entries):
    matrixList = []
    for key, val in entries.items():
        for i in range(len(val)):
            matrixList.append(val[i])
        
    matrix = np.array(matrixList)
    matrix = matrix[:,[0,1,3,4,5,7,8,9]]
    return matrix

def bestLogisticModel(X,y):
    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    model = LogisticRegressionCV(max_iter=100)
    model.fit(X_scaled,y.ravel())
    score = model.score(X_scaled,y.ravel())
    print(f'highest accuracy for logistic regression: {model.scores_[1.0].max()}')
    print(f'coefficients for logistic regression model{model.coef_}')

if __name__ == '__main__':
    entries = getData('behavior-performance.txt')
    entries1 = validEntries1(entries)
    matrix1 = getMatrix1(entries1)
    distortion = kMeans(matrix1)
    

    entries2 = validEntries2(entries1)
    matrix2 = getMatrix2(entries2)
    X, y = np.split(matrix2,[-1],axis=1)
    regressionModel = bestModel(X,y)
    
    matrix3 = getMatrix3(entries)
    X, y = np.split(matrix3,[-1],axis=1)
    logisticModel = bestLogisticModel(X,y)


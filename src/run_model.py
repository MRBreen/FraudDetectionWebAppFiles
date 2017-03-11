import pandas as pd
import numpy as np
import cPickle as pickle
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from MykoModel import MykoModel

if __name__=='__main__':
    datafile = '../data/subset.json'
    myko = MykoModel(GradientBoostingClassifier())
    X,y = myko.get_data(datafile)
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    myko.fit(X_train, y_train)
    with open('model.pkl', 'w') as f:
        pickle.dump(myko, f)

'''
Created on Sep 11, 2019

@author: aurea
'''

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from utils.Metrics import Metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

class RandomForestOtherTest(object):
    '''
    classdocs
    '''


    def __init__(self, _features_matrix, _target, _attributes_names, _title, _test_matrix):
        # defining feature matrix(X) and response vector(y)
        self.X = _features_matrix;
        self.X_test = _test_matrix;
        self.y = _target;
        self.attributes_names = _attributes_names;
        self.title = _title;
        self.predicted_y = [];
        self.prediction();

    def prediction(self):
        print('****************Random Forest****************');

        # create regression object
        reg = RandomForestRegressor(random_state=42,n_estimators=100)

        # train the model using the training sets
        reg.fit(self.X, self.y);


        self.predicted_y = reg.predict(self.X_test);
        

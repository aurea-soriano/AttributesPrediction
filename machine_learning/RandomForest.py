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

class RandomForest(object):
    '''
    classdocs
    '''


    def __init__(self, _features_matrix, _target, _attributes_names, _title):
        # defining feature matrix(X) and response vector(y)
        self.X = _features_matrix;
        self.y = _target;
        self.attributes_names = _attributes_names;
        self.title = _title;
        self.predicted_y = [];
        self.prediction();

    def prediction(self):
        print('****************Random Forest****************');


        # splitting X and y into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3)


        # create regression object
        reg = RandomForestRegressor(random_state=42,n_estimators=100)

        # train the model using the training sets
        reg.fit(X_train, y_train);


        # variance score: 1 means perfect prediction
        print('Variance score: {}'.format(reg.score(X_test, y_test)));


        y_train_pred = reg.predict(X_train);
        y_test_pred = reg.predict(X_test);

        self.predicted_y = reg.predict(self.X);
        # standard error
        std_error = Metrics.std_error(y_train, y_train_pred);
        r2_score = metrics.r2_score(y_train, y_train_pred);
        print('Std error: \n',std_error);
        #R2 corresponds to the squared correlation between the observed
        # outcome values and the predicted values by the model.
        # The Higher the R-squared, the better the model.
        print('R2: \n',r2_score);



        # Plot
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        plt.title('Random Forest - '+self.title);
        plt.xlabel('Real Y');
        plt.ylabel('Predicted Y');
        ax1.scatter(np.array(self.predicted_y), np.array(self.y), s=20, c='b', alpha=0.5,  label='StdError: '+ str(round(std_error,4)) + ' R2: ' + str(round(r2_score,4)));
        plt.legend(loc='upper left');
        plt.show()

        print("Features sorted by their score:");
        print(sorted(zip(map(lambda x: round(x, 4), reg.feature_importances_), self.attributes_names),
             reverse=True));

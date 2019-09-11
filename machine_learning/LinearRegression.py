'''
Created on Sep 7, 2019

@author: aurea
'''
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn import linear_model, metrics 
from sklearn.model_selection import train_test_split 
from utils.Metrics import Metrics

class LinearRegression(object):
    '''
    classdocs
    '''


    def __init__(self, _features_matrix, _target):
        # defining feature matrix(X) and response vector(y) 
        self.X = _features_matrix;
        self.y = _target;
        self.prediction();
        
    def prediction(self):
        print('****************Linear Regression****************');
        # splitting X and y into training and testing sets 
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.4) 
        
        train_num = X_train.shape[0];
        test_num = X_test.shape[0];
        
        # create linear regression object 
        reg = linear_model.LinearRegression();
  
        # train the model using the training sets 
        reg.fit(X_train, y_train);
  
        # regression coefficients 
        print('Coefficients: \n', reg.coef_); 
  
        # variance score: 1 means perfect prediction 
        print('Variance score: {}'.format(reg.score(X_test, y_test))); 
  
        
        y_train_pred = reg.predict(X_train);
        y_test_pred = reg.predict(X_test);
        
        
        # standard error 
        std_train_error = Metrics.std_error(y_train, y_train_pred);
        r2_score_train = metrics.r2_score(y_train, y_train_pred)
        print('Train error: \n',std_train_error);
        #R2 corresponds to the squared correlation between the observed 
        # outcome values and the predicted values by the model. 
        # The Higher the R-squared, the better the model.
        print('R2: \n',r2_score_train);
        
        std_test_error = Metrics.std_error(y_test, y_test_pred);
        r2_score_test = metrics.r2_score(y_test, y_test_pred)
        print('Test error: \n',std_test_error);
        #R2 corresponds to the squared correlation between the observed 
        # outcome values and the predicted values by the model. 
        # The Higher the R-squared, the better the model.
        print('R2: \n',r2_score_test);
        
        
        # Plot
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        plt.title('Linear Regression');
        plt.xlabel('Real Y');
        plt.ylabel('Predicted Y');
        ax1.scatter(np.array(y_train_pred), np.array(y_train), s=20, c='b', alpha=0.5,  label='Train '+ str(round(std_train_error,4)));
        #ax1.scatter(np.array(y_test_pred), np.array(y_test), s=10, c='g', edgecolors='b', alpha=0.5, label='Test '+ str(round(std_test_error,4)));
        plt.xlim([-0.4,0.4]);
        plt.ylim([-0.4,0.4]);
        plt.legend(loc='upper left');
        plt.show()

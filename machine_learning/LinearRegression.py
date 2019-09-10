'''
Created on Sep 7, 2019

@author: aurea
'''
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn import linear_model, metrics 
from sklearn.model_selection import train_test_split 
from utils.Utils import Utils

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
        # splitting X and y into training and testing sets 
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.4, 
                                                    random_state=1) 
        
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
        std_train_error = Utils.std_error(y_train, y_train_pred);
        std_test_error =  Utils.std_error(y_test, y_test_pred);
        
        print('Train error: \n',std_train_error);
        print('Test error: \n',std_test_error);
        
        # Plot
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        plt.title('Linear Regression');
        plt.xlabel('Real Y');
        plt.ylabel('Predicted Y');
        
        ax1.scatter(np.array(y_train_pred.T), np.array(y_train.T), s=10, c='b', alpha=0.5,  label='Train '+ str(round(std_train_error,4)));
        ax1.scatter(np.array(y_test_pred.T), np.array(y_test.T), s=10, c='g', alpha=0.5, label='Test '+ str(round(std_test_error,4)));
        plt.legend(loc='upper left');
        
        
        plt.show()

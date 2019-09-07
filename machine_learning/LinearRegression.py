'''
Created on Sep 7, 2019

@author: aurea
'''
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn import linear_model, metrics 
from sklearn.cross_validation import train_test_split 

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
  
        # create linear regression object 
        reg = linear_model.LinearRegression() 
  
        # train the model using the training sets 
        reg.fit(X_train, y_train) 
  
        # regression coefficients 
        print('Coefficients: \n', reg.coef_) 
  
        # variance score: 1 means perfect prediction 
        print('Variance score: {}'.format(reg.score(X_test, y_test))) 
  
        # plot for residual error 
  
        # ## setting plot style 
        plt.style.use('fivethirtyeight') 
  
        ## plotting residual errors in training data 
        plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train, 
                    color = "red", s = 10, label = 'Train data') 
          
        ## plotting residual errors in test data 
        plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test, 
                    color = "blue", s = 10, label = 'Test data') 
          
        ## plotting line for zero residual error 
        plt.hlines(y = 0, xmin = 0, xmax = 0.04, linewidth = 2) 
          
        ## plotting legend 
        plt.legend(loc = 'upper right') 
          
        ## plot title 
        plt.title("Residual errors") 
          
        ## function to show plot 
        plt.show() 
        
    
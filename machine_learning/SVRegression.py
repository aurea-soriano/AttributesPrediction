'''
Created on Sep 7, 2019

@author: aurea
'''
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn import linear_model, metrics 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVR
from utils.Metrics import Metrics

class SVRegression(object):
    '''
    classdocs
    '''


    def __init__(self, _features_matrix, _target):
        # defining feature matrix(X) and response vector(y) 
        self.X = _features_matrix;
        self.y = _target;
        self.prediction();
        
    def prediction(self):
        
        print('****************SVR Regression****************');
        # splitting X and y into training and testing sets 
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, 
                                                    random_state=1) 
        train_num = X_train.shape[0];
        test_num = X_test.shape[0];
        

        # create support vector regression object 
        svr_rbf = SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=1.5,
                kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False); 
        svr_rbf.fit(X_train, y_train);
        print('Variance score(RBF): {}'.format(svr_rbf.score(X_test, y_test))) 
        y_train_pred = svr_rbf.predict(X_train);
        y_test_pred = svr_rbf.predict(X_test);
        
        
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
        #RBF
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        plt.title('Support Vector Regression - RBF');
        plt.xlabel('Real Y');
        plt.ylabel('Predicted Y');
        #plt.xlim([-0.4,0.4]);
        #plt.ylim([-0.4,0.4]);
        ax1.scatter(np.array(y_train_pred), np.array(y_train), s=20, c='b', alpha=0.5,  label='StdError: '+ str(round(std_train_error,4)) + ' R2: ' + str(round(r2_score_train,4)));
        #ax1.scatter(np.array(y_test_pred_rbf.T), np.array(y_test.T), s=10, c='g', alpha=0.5, label='Test '+ str(round(std_test_error_rbf,4)));
        plt.legend(loc='upper left');
        
        plt.show()
        
        
               
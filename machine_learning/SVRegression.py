'''
Created on Sep 7, 2019

@author: aurea
'''
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn import linear_model, metrics 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVR
from utils.Utils import Utils

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
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.4, 
                                                    random_state=1) 
        train_num = X_train.shape[0];
        test_num = X_test.shape[0];
        

        # create support vector regression object 
        #svr = SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma='scale',
        #        kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False);
        svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
        svr_lin = SVR(kernel='linear', C=100, gamma='auto')
        svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
               coef0=1)
        lw = 2
        svrs = [svr_rbf, svr_lin, svr_poly]
        kernel_label = ['RBF', 'Linear', 'Polynomial']
        model_color = ['m', 'c', 'g']

        # train the model using the training sets 
        #svr.fit(X_train, y_train);
        svr_rbf.fit(X_train, y_train);
        svr_lin.fit(X_train, y_train);
        svr_poly.fit(X_train, y_train);
        
        # variance score: 1 means perfect prediction 
        print('Variance score(RBF): {}'.format(svr_rbf.score(X_test, y_test))) 
        print('Variance score(Lin): {}'.format(svr_lin.score(X_test, y_test))) 
        print('Variance score(Poly): {}'.format(svr_poly.score(X_test, y_test))) 
        
        
        y_train_pred_rbf = svr_rbf.predict(X_train);
        y_test_pred_rbf = svr_rbf.predict(X_test);
        
        y_train_pred_lin = svr_lin.predict(X_train);
        y_test_pred_lin = svr_lin.predict(X_test);
        
        y_train_pred_poly = svr_poly.predict(X_train);
        y_test_pred_poly = svr_poly.predict(X_test);
       
        # standard error 
        std_train_error_rbf = Utils.std_error(y_train, y_train_pred_rbf);
        std_test_error_rbf =  Utils.std_error(y_test, y_test_pred_rbf);
        
        std_train_error_lin = Utils.std_error(y_train, y_train_pred_lin);
        std_test_error_lin =  Utils.std_error(y_test, y_test_pred_lin);
        
        std_train_error_poly = Utils.std_error(y_train, y_train_pred_poly);
        std_test_error_poly =  Utils.std_error(y_test, y_test_pred_poly);
        
        
        print('Train error(rbf): \n',std_train_error_rbf);
        print('Test error(rbf): \n',std_test_error_rbf);
        
        print('Train error(lin): \n',std_train_error_lin);
        print('Test error(lin): \n',std_test_error_lin);
        
        print('Train error(poly): \n',std_train_error_poly);
        print('Test error(poly): \n',std_test_error_poly);
        
        # Plot
        
        #RBF
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        plt.title('Support Vector Regression - RBF');
        plt.xlabel('Real Y');
        plt.ylabel('Predicted Y');
        
        ax1.scatter(np.array(y_train_pred_rbf.T), np.array(y_train.T), s=10, c='b', alpha=0.5,  label='Train '+ str(round(std_train_error_rbf,4)));
        ax1.scatter(np.array(y_test_pred_rbf.T), np.array(y_test.T), s=10, c='g', alpha=0.5, label='Test '+ str(round(std_test_error_rbf,4)));
        plt.legend(loc='upper left');
        
        plt.show()
        
        #lin
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        plt.title('Support Vector Regression (lin)');
        plt.xlabel('Real Y');
        plt.ylabel('Predicted Y');
        
        ax1.scatter(np.array(y_train_pred_lin.T), np.array(y_train.T), s=10, c='b', alpha=0.5,  label='Train '+ str(round(std_train_error_lin,4)));
        ax1.scatter(np.array(y_test_pred_lin.T), np.array(y_test.T), s=10, c='g', alpha=0.5, label='Test '+ str(round(std_test_error_lin,4)));
        plt.legend(loc='upper left');
        
        plt.show()

        #poly
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        plt.title('Support Vector Regression (poly)');
        plt.xlabel('Real Y');
        plt.ylabel('Predicted Y');
        
        ax1.scatter(np.array(y_train_pred_poly.T), np.array(y_train.T), s=10, c='b', alpha=0.5,  label='Train '+ str(round(std_train_error_poly,4)));
        ax1.scatter(np.array(y_test_pred_poly.T), np.array(y_test.T), s=10, c='g', alpha=0.5, label='Test '+ str(round(std_test_error_poly,4)));
        plt.legend(loc='upper left');
        
        plt.show()


    
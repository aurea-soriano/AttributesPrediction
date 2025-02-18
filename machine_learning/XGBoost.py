'''
Created on Sep 11, 2019

@author: aurea
'''
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn import linear_model, metrics 
from sklearn.model_selection import train_test_split 
from utils.Metrics import Metrics
import xgboost as xgb


class XGBoost(object):
    '''
    classdocs
    '''


    def __init__(self, _features_matrix, _target, _attributes_names, _title):
        # defining feature matrix(X) and response vector(y) 
        self.X = _features_matrix;
        self.y = _target;
        self.title = _title;
        self.attributes_names = _attributes_names;
        self.prediction();
        
    def prediction(self):
        
        print('****************XGBoost Regression****************');
        # splitting X and y into training and testing sets 
        mm_y = Metrics.min_max_scaling(self.y);
        X_train, X_test, y_train, y_test = train_test_split(self.X, mm_y, test_size=0.3, 
                                                    random_state=1) 
        train_num = X_train.shape[0];
        test_num = X_test.shape[0];
        
        dtrain = xgb.DMatrix(X_train, label=y_train);
        dtest = xgb.DMatrix(X_test, label=y_test);

        param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
        param['nthread'] = 4
        param['eval_metric'] = 'auc'
        
        evallist = [(dtest, 'eval'), (dtrain, 'train')]
             
        num_round = 10
        bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=10)
        y_train_pred = bst.predict(dtrain, ntree_limit=bst.best_ntree_limit)
        y_test_pred = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)
        
    
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
        
        '''
        xgb.plot_importance(bst)
        xgb.plot_tree(bst, num_trees=2)
        xgb.to_graphviz(bst, num_trees=2)
        '''
        
        # Plot
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        plt.title('XGBoost - '+self.title);
        plt.xlabel('Real Y');
        plt.ylabel('Predicted Y');
        #plt.xlim([-0.4,0.4]);
        #plt.ylim([-0.4,0.4]);
        ax1.scatter(np.array(y_train_pred), np.array(y_train), s=20, c='b', alpha=0.5,  label='StdError: '+ str(round(std_train_error,4)) + ' R2: ' + str(round(r2_score_train,4)));
        #ax1.scatter(np.array(y_test_pred.T), np.array(y_test.T), s=10, c='g', alpha=0.5, label='Test '+ str(round(std_test_error,4)));
        plt.legend(loc='upper left');
        
        plt.show()

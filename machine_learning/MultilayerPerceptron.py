'''
Created on Sep 7, 2019

@author: aurea
'''
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn import linear_model, metrics 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPRegressor
from utils.Metrics import Metrics

class MultilayerPerceptron(object):
    '''
    classdocs
    '''


    def __init__(self, _features_matrix, _target,  _attributes_names, _title):
        # defining feature matrix(X) and response vector(y) 
        self.X = _features_matrix;
        self.y = _target;
        self.title = _title;
        self.attributes_names = _attributes_names;
        self.prediction();
        
    def prediction(self):
        
        print('****************MLP Regression****************');
        # splitting X and y into training and testing sets 
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, 
                                                    random_state=1) 
        train_num = X_train.shape[0];
        test_num = X_test.shape[0];
        #Multi-layer Perceptron is sensitive to feature scaling, 
        #so it is highly recommended to scale your data.
        #scaler = StandardScaler()  , "Target: DSW"
        # Don't cheat - fit only on training data
        #scaler.fit(X_train)  
        #X_train = scaler.transform(X_train)  
        # apply same transformation to test data
        #X_test = scaler.transform(X_test)  

        # create multilayer perceptron object 
        mlp = MLPRegressor(hidden_layer_sizes=(5,2,),
                                     activation='logistic',# {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
                                     solver='adam',
                                     learning_rate='constant', #{‘constant’, ‘invscaling’, ‘adaptive’}
                                     max_iter=1000,
                                     learning_rate_init=0.001,
                                     alpha=0.0001);
                   
           

        # train the model using the training sets 
        mlp.fit(X_train, y_train) 
        
        # variance score: 1 means perfect prediction 
        print('Variance score: {}'.format(mlp.score(X_test, y_test))) 
  
                
        # regression coefficients 
        #print('Coefficients: \n', mlp.coefs_) 

        # layers 
        #print('Layers: \n',mlp.n_layers_)
        
        #outputs
        #print('Outputs: \n',mlp.n_outputs_)
        
        #out activation
        #print('Out activation: \n',mlp.out_activation_)
        
              
        y_train_pred = mlp.predict(X_train);
        y_test_pred = mlp.predict(X_test);
        
       
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
        
        
        print('Train error: \n',std_train_error);
        print('Test error: \n',std_test_error);
        
        # Plot
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        plt.title('Multilayer Perceptron - '+self.title);
        plt.xlabel('Real Y');
        plt.ylabel('Predicted Y');
        #plt.xlim([-0.4,0.4]);
        #plt.ylim([-0.4,0.4]);
        ax1.scatter(np.array(y_train_pred), np.array(y_train), s=20, c='b', alpha=0.5,  label='StdError: '+ str(round(std_train_error,4)) + ' R2: ' + str(round(r2_score_train,4)));
        #ax1.scatter(np.array(y_test_pred.T), np.array(y_test.T), s=10, c='g', alpha=0.5, label='Test '+ str(round(std_test_error,4)));
        plt.legend(loc='upper left');
        
        plt.show()

        
    
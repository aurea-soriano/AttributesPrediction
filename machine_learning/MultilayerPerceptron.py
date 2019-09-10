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
from utils.Utils import Utils

class MultilayerPerceptron(object):
    '''
    classdocs
    '''


    def __init__(self, _features_matrix, _target):
        # defining feature matrix(X) and response vector(y) 
        self.X = _features_matrix;
        self.y = _target;
        self.prediction();
        
    def prediction(self):
        
        print('****************MLP Regression****************');
        # splitting X and y into training and testing sets 
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.4, 
                                                    random_state=1) 
        train_num = X_train.shape[0];
        test_num = X_test.shape[0];
        #Multi-layer Perceptron is sensitive to feature scaling, 
        #so it is highly recommended to scale your data.
        scaler = StandardScaler()  
        # Don't cheat - fit only on training data
        scaler.fit(X_train)  
        X_train = scaler.transform(X_train)  
        # apply same transformation to test data
        X_test = scaler.transform(X_test)  

        # create multilayer perceptron object 
        mlp = MLPRegressor(hidden_layer_sizes=(5,),
                                     activation='relu',
                                     solver='adam',
                                     learning_rate='adaptive',
                                     max_iter=1000,
                                     learning_rate_init=0.01,
                                     alpha=0.01);
                   
           

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
        std_train_error = Utils.std_error(y_train, y_train_pred);
        std_test_error =  Utils.std_error(y_test, y_test_pred);
        
        
        print('Train error: \n',std_train_error);
        print('Test error: \n',std_test_error);
        
        # Plot
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        plt.title('Multilayer Perceptron');
        plt.xlabel('Real Y');
        plt.ylabel('Predicted Y');
        plt.xlim([-0.4,0.4]);
        plt.ylim([-0.4,0.4]);
        ax1.scatter(np.array(y_train_pred.T), np.array(y_train.T), s=10, c='b', alpha=0.5,  label='Train '+ str(round(std_train_error,4)));
        #ax1.scatter(np.array(y_test_pred.T), np.array(y_test.T), s=10, c='g', alpha=0.5, label='Test '+ str(round(std_test_error,4)));
        plt.legend(loc='upper left');
        
        plt.show()

        
    
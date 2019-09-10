'''
Created on Sep 9, 2019

@author: aurea
'''

import numpy as np 

class Utils(object):
    '''
    classdocs
    '''
    
    
    @staticmethod
    def std_error(_real_y, _predicted_y):
        num = _real_y.shape[0];
        temp = 0;
        for i in range(num):
            temp += np.power((_real_y[i]-_predicted_y[i]),2);
        temp/=num;
        return np.float(np.sqrt(temp));
    
  
        
'''
Created on Sep 11, 2019

@author: aurea
'''

import numpy as np
import scipy

class Metrics(object):
    '''
    classdocs
    '''

    @staticmethod
    def std_error(real_y, predicted_y):
        num = len(real_y);
        temp = 0;
        for i in range(num):
            temp += np.power((real_y[i]-predicted_y[i]),2);
        temp/=num;
        return np.float(np.sqrt(temp));

    @staticmethod
    def pearson_correlation(X, Y):
        return scipy.stats.pearsonr(X, Y)[0];

    @staticmethod
    def calculate_pearson_matrix(attributes_values):
        matrix_result = [];
        for i in range(0,len(attributes_values)):
            row_result = [];
            for j in range(0,len(attributes_values)):
                row_result.append(Metrics.pearson_correlation(attributes_values[i], attributes_values[j]));
            matrix_result.append(row_result);
        return matrix_result;

    @staticmethod
    def min_max_scaling(vector):
        max_value = np.max(vector);
        min_value = np.min(vector);
        new_max = 1;
        new_min = 0;
        if  max_value == min_value:
            den = 0.00000001;
        else:
            den = max_value - min_value;
        return  (((new_max - new_min) * (vector - min_value)) / den) + new_min;

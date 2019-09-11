'''
Created on Sep 11, 2019

@author: aurea
'''
import pandas
from pandas.plotting import scatter_matrix
from utils.Metrics import Metrics
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from utils.FileWriter import FileWriter

class CorrelationAnalysis(object):
    '''
    classdocs
    '''


    def __init__(self, _attributes_list, _attributes_matrix, _attributes_names):
        '''
        Constructor
        '''
        self.attributes_list = _attributes_list;
        self.attributes_matrix = _attributes_matrix;
        self.attributes_names = _attributes_names;
        self.analyze();
        
    def analyze(self):
        pearson_matrix = Metrics.calculate_pearson_matrix(self.attributes_list);
        FileWriter.write("results/correlation_matrix.txt", self.attributes_names, pearson_matrix);
        
        sns.set(font_scale=0.7) 
        ax = sns.heatmap(pearson_matrix, cmap="coolwarm",  vmin=-1, vmax=1,annot=True, xticklabels = self.attributes_names,
                         yticklabels = self.attributes_names, square=True)
        ax.xaxis.set_ticks_position('top')
        ax.set_ylim(9,0);
        plt.xticks(rotation=45)  
        plt.show()
       
        df = pandas.DataFrame(self.attributes_matrix, columns=self.attributes_names)
        scatter_matrix(df, alpha=0.2)
        plt.show() 
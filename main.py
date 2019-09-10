'''
Created on Sep 7, 2019

@author: aurea
'''
from utils.FileReader import FileReader
from machine_learning.LinearRegression import LinearRegression
from machine_learning.MultilayerPerceptron import MultilayerPerceptron
import numpy as np

class Main(object):
    '''
    classdocs
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
        self.create_training_data();
        
    def create_training_data(self):
        attributes_files = ["Porosity", "Porosity-Effective Ref", "NTG", "Sw_base", "Sg_Base", "dSg"];
        target_file = "dSw";
        
        training_matrix = []
        x = []
        y = []
        porosity = []
        porosity_effective = []
        ntg = []
        sw_base = []
        sg_base = []
        dsg = []
        dsw = []
       
        # reading Porosity
        fileReader = FileReader("data/"+attributes_files[0], 20);
        fileReader.read();
        x = fileReader.resulting_matrix[:,0];
        y = fileReader.resulting_matrix[:,1];
        porosity = fileReader.resulting_matrix[:,2];
        
        # reading Porosity-effective
        fileReader = FileReader("data/"+attributes_files[1], 20);
        fileReader.read();
        porosity_effective = fileReader.resulting_matrix[:,2];
        
        # reading ntg
        fileReader = FileReader("data/"+attributes_files[2], 20);
        fileReader.read();
        ntg = fileReader.resulting_matrix[:,2];
        
        # reading sw_base
        fileReader = FileReader("data/"+attributes_files[3], 20);
        fileReader.read();
        sw_base = fileReader.resulting_matrix[:,2];
        
        # reading sg_base
        fileReader = FileReader("data/"+attributes_files[4], 20);
        fileReader.read();
        sg_base = fileReader.resulting_matrix[:,2];
        
        # reading dsg
        fileReader = FileReader("data/"+attributes_files[5], 20);
        fileReader.read();
        dsg = fileReader.resulting_matrix[:,2];
        
        # reading dsw "target"
        fileReader = FileReader("data/"+target_file, 20);
        fileReader.read();
        dsw = fileReader.resulting_matrix[:,2];
        
        features_matrix = np.concatenate((x, y, porosity, porosity_effective, ntg, sw_base,
                                          sg_base,dsg ), axis=1)
        attributes_names = ["X", "Y", "Porosity", "Porosity-Effective Ref", "NTG", "Sw_base", "Sg_Base", "dSg"];

        target_name= "dSw";
        
        linearRegression = LinearRegression(features_matrix, dsw, attributes_names, target_name);
        multilayerPerceptron = MultilayerPerceptron(features_matrix, dsw, attributes_names, target_name);

if __name__ == '__main__':
    app = Main("");
    pass
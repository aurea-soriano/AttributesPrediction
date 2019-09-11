'''
Created on Sep 7, 2019

@author: aurea
'''
from utils.FileReader import FileReader
from machine_learning.LinearRegression import LinearRegression
from machine_learning.MultilayerPerceptron import MultilayerPerceptron
from machine_learning.SVRegression import SVRegression
from machine_learning.XGBoost import XGBoost
from machine_learning.CorrelationAnalysis import CorrelationAnalysis
from machine_learning.RandomForest import RandomForest
from utils.Metrics import Metrics
import numpy as np
from scipy.spatial.transform import rotation


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
        attributes_files = ["Porosity", "Porosity-Effective Ref", "NTG", "Sw_base", "Sg_Base", "dSg", "dSw"];
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
        porosityMatrix = FileReader.read("data/"+attributes_files[0], 20);
        x = np.squeeze(np.asarray(porosityMatrix[:,0]));
        y = np.squeeze(np.asarray(porosityMatrix[:,1]));
        porosity = np.squeeze(np.asarray(porosityMatrix[:,2]));
        
        # reading Porosity-effective
        porosityEffectiveMatrix = FileReader.read("data/"+attributes_files[1], 20);
        porosity_effective = np.squeeze(np.asarray(porosityEffectiveMatrix[:,2]));
        
        # reading ntglinear_model,
        ntgMatrix = FileReader.read("data/"+attributes_files[2], 20);
        ntg = np.squeeze(np.asarray(ntgMatrix[:,2]));
        
        # reading sw_base
        swBaseMatrix = FileReader.read("data/"+attributes_files[3], 20);
        sw_base = np.squeeze(np.asarray(swBaseMatrix[:,2]));
        
        # reading sg_base
        sgBaseMatrix = FileReader.read("data/"+attributes_files[4], 20);
        sg_base = np.squeeze(np.asarray(sgBaseMatrix[:,2]));
        
        # reading dsg
        dsgMatrix = FileReader.read("data/"+attributes_files[5], 20);
        dsg = np.squeeze(np.asarray(dsgMatrix[:,2]));
        
        # reading dsw 
        dswMatrix = FileReader.read("data/"+attributes_files[6], 20);
        dsw = np.squeeze(np.asarray(dswMatrix[:,2]));
        
        
        
        attributes_list = [x, y, porosity, porosity_effective, ntg, sw_base, sg_base, dsg, dsw];
        attributes_matrix = np.c_[x,y, porosity, porosity_effective, ntg, sw_base, sg_base, dsg, dsw];
        attributes_names = ["x", "y", "Porosity", "Porosity-Eff", "NTG", "Sw_base", "Sg_Base", "dSg", "dSw"];
        #CorrelationAnalysis(attributes_list, attributes_matrix, attributes_names);
       
        
        #np.c_[x,y, porosity, porosity_effective, ntg, sw_base, sg_base, dsg, dsw];
        features_matrix_dsw = np.c_[porosity, porosity_effective, ntg, sw_base];
        features_matrix_dsg = np.c_[porosity, porosity_effective, ntg, sg_base];
    
        
        linearRegression = LinearRegression(features_matrix_dsw, dsw);
        multilayerPerceptron = MultilayerPerceptron(features_matrix_dsw, dsw);
        svRegression = SVRegression(features_matrix_dsw, dsw);
        xgBoost = XGBoost(features_matrix_dsw, dsw);
        randomForest = RandomForest(features_matrix_dsw, dsw);
        
        #linearRegression2 = LinearRegression(features_matrix_dsg, dsg);
        #multilayerPerceptron2 = MultilayerPerceptron(features_matrix_dsg, dsg);
        #svRegression2 = SVRegression(features_matrix_dsg, dsg);
        #xgBoost2 = xgBoost(features_matrix_dsg, dsg);

if __name__ == '__main__':
    app = Main("");
    pass
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
        self.attributes_files = ["Porosity", "Porosity-Effective Ref", "NTG", "Sw_base", "Sg_Base", "dSg", "dSw", "avg_dRMS"];
        
        self.training_matrix = []
        self.x = []
        self.y = []
        self.porosity = []
        self.porosity_effective = []
        self.ntg = []
        self.sw_base = []
        self.sg_base = []
        self.dsg = []
        self.dsw = []
        self.avg_dRMS = []
        self.xdRMS = []
        self.ydRMS = []
        self. attributes_list = []
        self.attributes_matrix= []
        self.attributes_names = []
        self.features_matrix_dsw = []
        self.features_matrix_dsg = []
        
    def create_training_data(self):
        
        # reading Porosity
        porosityMatrix = FileReader.read("data/"+self.attributes_files[0], 20);
        self.x = np.squeeze(np.asarray(porosityMatrix[:,0]));
        self.y = np.squeeze(np.asarray(porosityMatrix[:,1]));
        self.porosity = np.squeeze(np.asarray(porosityMatrix[:,2]));
        
        # reading Porosity-effective
        porosityEffectiveMatrix = FileReader.read("data/"+self.attributes_files[1], 20);
        self.porosity_effective = np.squeeze(np.asarray(porosityEffectiveMatrix[:,2]));
        
        # reading ntglinear_model,
        ntgMatrix = FileReader.read("data/"+self.attributes_files[2], 20);
        self.ntg = np.squeeze(np.asarray(ntgMatrix[:,2]));
        
        # reading sw_base
        swBaseMatrix = FileReader.read("data/"+self.attributes_files[3], 20);
        self.sw_base = np.squeeze(np.asarray(swBaseMatrix[:,2]));
        
        # reading sg_base
        sgBaseMatrix = FileReader.read("data/"+self.attributes_files[4], 20);
        self.sg_base = np.squeeze(np.asarray(sgBaseMatrix[:,2]));
        
        # reading dsg
        dsgMatrix = FileReader.read("data/"+self.attributes_files[5], 20);
        self.dsg = np.squeeze(np.asarray(dsgMatrix[:,2]));
        
        # reading dsw 
        dswMatrix = FileReader.read("data/"+self.attributes_files[6], 20);
        self.dsw = np.squeeze(np.asarray(dswMatrix[:,2]));
        
        
        # reading dsw 
        avg_dRMSMatrix = FileReader.read("data/"+self.attributes_files[7], 20);
        self.xdRMS = np.squeeze(np.asarray(avg_dRMSMatrix[0:len(self.dsw),0]));
        self.ydRMS = np.squeeze(np.asarray(avg_dRMSMatrix[0:len(self.dsw),1]));
        self.avg_dRMS= np.squeeze(np.asarray(avg_dRMSMatrix[0:len(self.dsw),2]));
        
        
        self.attributes_list = [self.x, self.y, self.porosity, self.porosity_effective, self.ntg, self.sw_base, self.sg_base, self.dsg, self.dsw, self.avg_dRMS];
        self.attributes_matrix = np.c_[self.x, self.y, self.porosity, self.porosity_effective, self.ntg, self.sw_base, self.sg_base, self.dsg, self.dsw, self.avg_dRMS];
        self.attributes_names = ["x", "y", "Porosity", "Porosity-Eff", "NTG", "Sw_base", "Sg_Base", "dSg", "dSw", "avg_dRMS"];
       
       
        
        
    def machine_learning_analysis(self):  
        
        
        attribute_names_dsw = ["x", "y", "porosity", "porosity_effective", "ntg", "sw_base", "avg_dRMS"];
        attribute_names_dsg = ["x", "y", "porosity", "porosity_effective", "ntg", "sg_base", "avg_dRMS"];
        
        #np.c_[x,y, porosity, porosity_effective, ntg, sw_base, sg_base, dsg, dsw];
        self.features_matrix_dsw = np.c_[self.x, self.y, self.porosity, self.porosity_effective, self.ntg, self.sw_base, self.avg_dRMS];
        self.features_matrix_dsg = np.c_[self.x, self.y, self.porosity, self.porosity_effective, self.ntg, self.sg_base, self.avg_dRMS];
     
        #CorrelationAnalysis(self.attributes_list, self.attributes_matrix, self.attributes_names);
        
        #linearRegression = LinearRegression(self.features_matrix_dsw, self.dsw);
        #multilayerPerceptron = MultilayerPerceptron(self.features_matrix_dsw, self.dsw);
        #svRegression = SVRegression(self.features_matrix_dsw, self.dsw);
        #xgBoost = XGBoost(self.features_matrix_dsw, self.dsw);
         
        randomForest = RandomForest(self.features_matrix_dsw, self.dsw, attribute_names_dsw, "Target: DSW");
        
        #linearRegression2 = LinearRegression(self.features_matrix_dsg, self.dsg);
        #multilayerPerceptron2 = MultilayerPerceptron(self.features_matrix_dsg, self.dsg);
        #svRegression2 = SVRegression(self.features_matrix_dsg, self.dsg);
        #xgBoost2 = XGBoost(self.features_matrix_dsg, self.dsg);
        randomForest = RandomForest(self.features_matrix_dsg, self.dsg, attribute_names_dsg, "Target: DSG");

if __name__ == '__main__':
    main = Main("");
    main.create_training_data();
    main.machine_learning_analysis();
    
    pass
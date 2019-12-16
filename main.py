'''
Created on Sep 7, 2019

@author: aurea
'''

import sys
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
        self.attributes_files = ["Porosity", "Porosity-Effective Ref", "NTG", "Sw_base",
        "Sg_Base", "dSg", "dSw", "fair_int_dRMS_10m", "fair_int_dRMS_20m"];

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
        self.int_dRMS = []
        self.int_dRMS_10 = []
        self.int_dRMS_20 = []
        self.xdRMS = []
        self.ydRMS = []
        self. attributes_list = []
        self.attributes_matrix= []
        self.attributes_names = []
        self.features_matrix_dsw = []
        self.features_matrix_dsg = []

    def create_training_data(self):

        # reading Porosity
        porosityMatrices = FileReader.read("data/"+self.attributes_files[0], 20);
        self.x = porosityMatrices[0].flatten();
        self.y = porosityMatrices[1].flatten();
        self.porosity = porosityMatrices[2].flatten();

        # reading Porosity-effective
        porosityEffectiveMatrix = FileReader.read("data/"+self.attributes_files[1], 20);
        self.porosity_effective = porosityEffectiveMatrix[2].flatten();

        # reading ntglinear_model,
        ntgMatrix = FileReader.read("data/"+self.attributes_files[2], 20);
        self.ntg = ntgMatrix[2].flatten();

        # reading sw_base
        swBaseMatrix = FileReader.read("data/"+self.attributes_files[3], 20);
        self.sw_base = swBaseMatrix[2].flatten();

        # reading sg_base
        sgBaseMatrix = FileReader.read("data/"+self.attributes_files[4], 20);
        self.sg_base = sgBaseMatrix[2].flatten();

        # reading dsg
        dsgMatrix = FileReader.read("data/"+self.attributes_files[5], 20);
        self.dsg = dsgMatrix[2].flatten();

        # reading dsw
        dswMatrix = FileReader.read("data/"+self.attributes_files[6], 20);
        self.dsw = dswMatrix[2].flatten();


        # reading actual dRMS
        int_dRMSMatrix_10 = FileReader.read("data/"+self.attributes_files[7], 20);
        self.xdRMS = int_dRMSMatrix_10[0].flatten();
        self.ydRMS = int_dRMSMatrix_10[1].flatten();
        self.int_dRMS_10= int_dRMSMatrix_10[2].flatten();

        # reading actual dRMS
        int_dRMSMatrix_20 = FileReader.read("data/"+self.attributes_files[8], 20);
        self.xdRMS = int_dRMSMatrix_20[0].flatten();
        self.ydRMS = int_dRMSMatrix_20[1].flatten();
        self.int_dRMS_20= int_dRMSMatrix_20[2].flatten();


    def machine_learning_analysis(self):

        #10
        attribute_names_dsw = ["x","y", "int_dRMS"];
        attribute_names_dsg = ["x","y", "int_dRMS"];
        attributes_list = [self.x, self.y, self.dsg, self.dsw, self.int_dRMS_10];
        attributes_matrix = np.c_[self.x, self.y, self.dsg, self.dsw, self.int_dRMS_10];
        attributes_names = ["x","y","dSg", "dSw", "int_dRMS"];



        #np.c_[x,y, porosity, porosity_effective, ntg, sw_base, sg_base, dsg, dsw];
        self.features_matrix_dsw = np.c_[self.int_dRMS_10];#np.c_[self.x, self.y,self.int_dRMS_10];
        self.features_matrix_dsg = np.c_[self.int_dRMS_10];#np.c_[self.x, self.y,self.int_dRMS_10];


        #CorrelationAnalysis(attributes_list, attributes_matrix, attributes_names);

        linearRegression = LinearRegression(self.features_matrix_dsw, self.dsw, attribute_names_dsw, "Target: DSW");
        multilayerPerceptron = MultilayerPerceptron(self.features_matrix_dsw, self.dsw, attribute_names_dsw, "Target: DSW");
        svRegression = SVRegression(self.features_matrix_dsw, self.dsw, attribute_names_dsw, "Target: DSW");
        xgBoost = XGBoost(self.features_matrix_dsw, self.dsw, attribute_names_dsw, "Target: DSW");
        randomForest = RandomForest(self.features_matrix_dsw, self.dsw, attribute_names_dsw, "Target: DSW");

        linearRegression2 = LinearRegression(self.features_matrix_dsg, self.dsg, attribute_names_dsg, "Target: DSG");
        multilayerPerceptron2 = MultilayerPerceptron(self.features_matrix_dsg, self.dsg, attribute_names_dsg, "Target: DSG");
        svRegression2 = SVRegression(self.features_matrix_dsg, self.dsg, attribute_names_dsg, "Target: DSG");
        xgBoost2 = XGBoost(self.features_matrix_dsg, self.dsg, attribute_names_dsg, "Target: DSG");
        randomForest2 = RandomForest(self.features_matrix_dsg, self.dsg, attribute_names_dsg, "Target: DSG");


        #20
        attribute_names_dsw = ["x","y", "int_dRMS"];
        attribute_names_dsg = ["x","y", "int_dRMS"];
        attributes_list = [self.x, self.y, self.dsg, self.dsw, self.int_dRMS_20];
        attributes_matrix = np.c_[self.x, self.y, self.dsg, self.dsw, self.int_dRMS_20];
        attributes_names = ["x","y","dSg", "dSw", "int_dRMS"];



        #np.c_[x,y, porosity, porosity_effective, ntg, sw_base, sg_base, dsg, dsw];
        self.features_matrix_dsw = np.c_[self.int_dRMS_20];#np.c_[self.x, self.y,self.int_dRMS_20];
        self.features_matrix_dsg = np.c_[self.int_dRMS_20];#np.c_[self.x, self.y,self.int_dRMS_20];


        #CorrelationAnalysis(attributes_list, attributes_matrix, attributes_names);

        linearRegression = LinearRegression(self.features_matrix_dsw, self.dsw, attribute_names_dsw, "Target: DSW");
        multilayerPerceptron = MultilayerPerceptron(self.features_matrix_dsw, self.dsw, attribute_names_dsw, "Target: DSW");
        svRegression = SVRegression(self.features_matrix_dsw, self.dsw, attribute_names_dsw, "Target: DSW");
        xgBoost = XGBoost(self.features_matrix_dsw, self.dsw, attribute_names_dsw, "Target: DSW");
        randomForest = RandomForest(self.features_matrix_dsw, self.dsw, attribute_names_dsw, "Target: DSW");

        linearRegression2 = LinearRegression(self.features_matrix_dsg, self.dsg, attribute_names_dsg, "Target: DSG");
        multilayerPerceptron2 = MultilayerPerceptron(self.features_matrix_dsg, self.dsg, attribute_names_dsg, "Target: DSG");
        svRegression2 = SVRegression(self.features_matrix_dsg, self.dsg, attribute_names_dsg, "Target: DSG");
        xgBoost2 = XGBoost(self.features_matrix_dsg, self.dsg, attribute_names_dsg, "Target: DSG");
        randomForest2 = RandomForest(self.features_matrix_dsg, self.dsg, attribute_names_dsg, "Target: DSG");


if __name__ == '__main__':
    main = Main("");
    main.create_training_data();
    main.machine_learning_analysis();

    pass

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
import matplotlib.pyplot as plt
import math

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
        self.col = []
        self.row =[ ]
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
        self.x_size = 0
        self.y_size = 0

    def create_training_data(self):

        # reading Porosity
        porosityMatrices = FileReader.read("data/"+self.attributes_files[0], 20);
        self.x_size, self.y_size = porosityMatrices[0].shape;
        x = porosityMatrices[0].flatten();
        y = porosityMatrices[1].flatten();
        porosity = porosityMatrices[2].flatten();
        col = porosityMatrices[3].flatten();
        row = porosityMatrices[4].flatten();

        # reading Porosity-effective
        porosityEffectiveMatrix = FileReader.read("data/"+self.attributes_files[1], 20);
        porosity_effective = porosityEffectiveMatrix[2].flatten();

        # reading ntglinear_model,
        ntgMatrix = FileReader.read("data/"+self.attributes_files[2], 20);
        ntg = ntgMatrix[2].flatten();

        # reading sw_base
        swBaseMatrix = FileReader.read("data/"+self.attributes_files[3], 20);
        sw_base = swBaseMatrix[2].flatten();

        # reading sg_base
        sgBaseMatrix = FileReader.read("data/"+self.attributes_files[4], 20);
        sg_base = sgBaseMatrix[2].flatten();

        # reading dsg
        dsgMatrix = FileReader.read("data/"+self.attributes_files[5], 20);
        dsg = dsgMatrix[2].flatten();

        # reading dsw
        dswMatrix = FileReader.read("data/"+self.attributes_files[6], 20);
        dsw = dswMatrix[2].flatten();


        # reading actual dRMS
        int_dRMSMatrix_10 = FileReader.read("data/"+self.attributes_files[7], 20);
        xdRMS = int_dRMSMatrix_10[0].flatten();
        ydRMS = int_dRMSMatrix_10[1].flatten();
        int_dRMS_10= int_dRMSMatrix_10[2].flatten();

        # reading actual dRMS
        int_dRMSMatrix_20 = FileReader.read("data/"+self.attributes_files[8], 20);
        xdRMS = int_dRMSMatrix_20[0].flatten();
        ydRMS = int_dRMSMatrix_20[1].flatten();
        int_dRMS_20= int_dRMSMatrix_20[2].flatten();


        for i in range(len(x)):
            if math.isnan(x[i]) or math.isnan(y[i]) or math.isnan(dsg[i]) or math.isnan(dsw[i]) or math.isnan(xdRMS[i]) or  math.isnan(ydRMS[i]) or math.isnan(int_dRMS_10[i]) or math.isnan(int_dRMS_20[i]):
                #nothing
                print("")
            else:
                self.x.append(x[i])
                self.y.append(y[i])
                self.porosity.append(porosity[i])
                self.porosity_effective.append(porosity_effective[i])
                self.col.append(col[i])
                self.row.append(row[i])
                self.ntg.append(ntg[i])
                self.sw_base.append(sw_base[i])
                self.sg_base.append(sg_base[i])
                self.dsg.append(dsg[i])
                self.dsw.append(dsw[i])
                self.xdRMS.append(xdRMS[i])
                self.ydRMS.append(ydRMS[i])
                self.int_dRMS_10.append(int_dRMS_10[i])
                self.int_dRMS_20.append(int_dRMS_20[i])

    def machine_learning_analysis(self):

        current_cmap = plt.get_cmap("seismic")
        current_cmap.set_bad(color='black')

        # dRMS
        #10
        attribute_names_dsw = ["x","y", "int_dRMS"];
        attribute_names_dsg = ["x","y", "int_dRMS"];
        attributes_list = [self.x, self.y, self.dsg, self.dsw, self.int_dRMS_10];
        attributes_matrix = np.c_[self.x, self.y, self.dsg, self.dsw, self.int_dRMS_10];
        attributes_names = ["x","y","dSg", "dSw", "int_dRMS"];



        #np.c_[x,y, porosity, porosity_effective, ntg, sw_base, sg_base, dsg, dsw];
        self.features_matrix_dsw = np.c_[self.int_dRMS_10];#
        self.features_matrix_dsg = np.c_[self.int_dRMS_10];#


        #CorrelationAnalysis(attributes_list, attributes_matrix, attributes_names);

        #linearRegression = LinearRegression(self.features_matrix_dsw, self.dsw, attribute_names_dsw, "Target: DSW");
        #multilayerPerceptron = MultilayerPerceptron(self.features_matrix_dsw, self.dsw, attribute_names_dsw, "Target: DSW");
        #svRegression = SVRegression(self.features_matrix_dsw, self.dsw, attribute_names_dsw, "Target: DSW");
        #xgBoost = XGBoost(self.features_matrix_dsw, self.dsw, attribute_names_dsw, "Target: DSW");
        randomForest = RandomForest(self.features_matrix_dsw, self.dsw, attribute_names_dsw, "Target: DSW");
        predicted_matrix = np.empty((self.x_size, self.y_size));
        predicted_matrix[:] = np.nan
        print(len(randomForest.predicted_y))
        for x in range(len(randomForest.predicted_y)):
            predicted_matrix[int(self.col[x])-1][int(self.row[x])-1]= randomForest.predicted_y[x];
        plt.imshow(np.rot90(predicted_matrix), current_cmap,  vmin=-0.5, vmax=0.5);
        plt.colorbar()
        plt.show()


        #linearRegression2 = LinearRegression(self.features_matrix_dsg, self.dsg, attribute_names_dsg, "Target: DSG");
        #multilayerPerceptron2 = MultilayerPerceptron(self.features_matrix_dsg, self.dsg, attribute_names_dsg, "Target: DSG");
        #svRegression2 = SVRegression(self.features_matrix_dsg, self.dsg, attribute_names_dsg, "Target: DSG");
        #xgBoost2 = XGBoost(self.features_matrix_dsg, self.dsg, attribute_names_dsg, "Target: DSG");
        randomForest2 = RandomForest(self.features_matrix_dsg, self.dsg, attribute_names_dsg, "Target: DSG");
        predicted_matrix = np.empty((self.x_size, self.y_size));
        predicted_matrix[:] = np.nan
        for x in range(len(randomForest2.predicted_y)):
            predicted_matrix[int(self.col[x])-1][int(self.row[x])-1]= randomForest2.predicted_y[x];
        plt.imshow(np.rot90(predicted_matrix), cmap=current_cmap,  vmin=-0.05, vmax=0.05);
        plt.colorbar()
        plt.show()

        #20
        attribute_names_dsw = ["x","y", "int_dRMS"];
        attribute_names_dsg = ["x","y", "int_dRMS"];
        attributes_list = [self.x, self.y, self.dsg, self.dsw, self.int_dRMS_20];
        attributes_matrix = np.c_[self.x, self.y, self.dsg, self.dsw, self.int_dRMS_20];
        attributes_names = ["x","y","dSg", "dSw", "int_dRMS"];



        #np.c_[x,y, porosity, porosity_effective, ntg, sw_base, sg_base, dsg, dsw];
        self.features_matrix_dsw = np.c_[self.int_dRMS_20];#
        self.features_matrix_dsg = np.c_[self.int_dRMS_20];#


        #CorrelationAnalysis(attributes_list, attributes_matrix, attributes_names);

        #linearRegression = LinearRegression(self.features_matrix_dsw, self.dsw, attribute_names_dsw, "Target: DSW");
        #multilayerPerceptron = MultilayerPerceptron(self.features_matrix_dsw, self.dsw, attribute_names_dsw, "Target: DSW");
        #svRegression = SVRegression(self.features_matrix_dsw, self.dsw, attribute_names_dsw, "Target: DSW");
        #xgBoost = XGBoost(self.features_matrix_dsw, self.dsw, attribute_names_dsw, "Target: DSW");
        randomForest = RandomForest(self.features_matrix_dsw, self.dsw, attribute_names_dsw, "Target: DSW");
        predicted_matrix = np.empty((self.x_size, self.y_size));
        predicted_matrix[:] = np.nan
        for x in range(len(randomForest.predicted_y)):
            predicted_matrix[int(self.col[x])-1][int(self.row[x])-1]= randomForest.predicted_y[x];
        plt.imshow(np.rot90(predicted_matrix), current_cmap, vmin=-0.5, vmax=0.5);
        plt.colorbar()
        plt.show()

        #linearRegression2 = LinearRegression(self.features_matrix_dsg, self.dsg, attribute_names_dsg, "Target: DSG");
        #multilayerPerceptron2 = MultilayerPerceptron(self.features_matrix_dsg, self.dsg, attribute_names_dsg, "Target: DSG");
        #svRegression2 = SVRegression(self.features_matrix_dsg, self.dsg, attribute_names_dsg, "Target: DSG");
        #xgBoost2 = XGBoost(self.features_matrix_dsg, self.dsg, attribute_names_dsg, "Target: DSG");
        randomForest2 = RandomForest(self.features_matrix_dsg, self.dsg, attribute_names_dsg, "Target: DSG");
        predicted_matrix = np.empty((self.x_size, self.y_size));
        predicted_matrix[:] = np.nan
        for x in range(len(randomForest2.predicted_y)):
            predicted_matrix[int(self.col[x])-1][int(self.row[x])-1]= randomForest2.predicted_y[x];
        plt.imshow(np.rot90(predicted_matrix), current_cmap,  vmin=-0.05, vmax=0.05);
        plt.colorbar()
        plt.show()

        # X Y DRMS

        #10
        attribute_names_dsw = ["x","y", "int_dRMS"];
        attribute_names_dsg = ["x","y", "int_dRMS"];
        attributes_list = [self.x, self.y, self.dsg, self.dsw, self.int_dRMS_10];
        attributes_matrix = np.c_[self.x, self.y, self.dsg, self.dsw, self.int_dRMS_10];
        attributes_names = ["x","y","dSg", "dSw", "int_dRMS"];



        #np.c_[x,y, porosity, porosity_effective, ntg, sw_base, sg_base, dsg, dsw];
        self.features_matrix_dsw = np.c_[self.x, self.y,self.int_dRMS_10]; #np.c_[self.int_dRMS_10];#
        self.features_matrix_dsg = np.c_[self.x, self.y,self.int_dRMS_10]; #np.c_[self.int_dRMS_10];#


        #CorrelationAnalysis(attributes_list, attributes_matrix, attributes_names);

        #linearRegression = LinearRegression(self.features_matrix_dsw, self.dsw, attribute_names_dsw, "Target: DSW");
        #multilayerPerceptron = MultilayerPerceptron(self.features_matrix_dsw, self.dsw, attribute_names_dsw, "Target: DSW");
        #svRegression = SVRegression(self.features_matrix_dsw, self.dsw, attribute_names_dsw, "Target: DSW");
        #xgBoost = XGBoost(self.features_matrix_dsw, self.dsw, attribute_names_dsw, "Target: DSW");
        randomForest = RandomForest(self.features_matrix_dsw, self.dsw, attribute_names_dsw, "Target: DSW");
        predicted_matrix = np.empty((self.x_size, self.y_size));
        predicted_matrix[:] = np.nan
        print(len(randomForest.predicted_y))
        for x in range(len(randomForest.predicted_y)):
            predicted_matrix[int(self.col[x])-1][int(self.row[x])-1]= randomForest.predicted_y[x];
        plt.imshow(np.rot90(predicted_matrix), current_cmap,  vmin=-0.5, vmax=0.5);
        plt.colorbar()
        plt.show()


        #linearRegression2 = LinearRegression(self.features_matrix_dsg, self.dsg, attribute_names_dsg, "Target: DSG");
        #multilayerPerceptron2 = MultilayerPerceptron(self.features_matrix_dsg, self.dsg, attribute_names_dsg, "Target: DSG");
        #svRegression2 = SVRegression(self.features_matrix_dsg, self.dsg, attribute_names_dsg, "Target: DSG");
        #xgBoost2 = XGBoost(self.features_matrix_dsg, self.dsg, attribute_names_dsg, "Target: DSG");
        randomForest2 = RandomForest(self.features_matrix_dsg, self.dsg, attribute_names_dsg, "Target: DSG");
        predicted_matrix = np.empty((self.x_size, self.y_size));
        predicted_matrix[:] = np.nan
        for x in range(len(randomForest2.predicted_y)):
            predicted_matrix[int(self.col[x])-1][int(self.row[x])-1]= randomForest2.predicted_y[x];
        plt.imshow(np.rot90(predicted_matrix), current_cmap,  vmin=-0.05, vmax=0.05);
        plt.colorbar()
        plt.show()

        #20
        attribute_names_dsw = ["x","y", "int_dRMS"];
        attribute_names_dsg = ["x","y", "int_dRMS"];
        attributes_list = [self.x, self.y, self.dsg, self.dsw, self.int_dRMS_20];
        attributes_matrix = np.c_[self.x, self.y, self.dsg, self.dsw, self.int_dRMS_20];
        attributes_names = ["x","y","dSg", "dSw", "int_dRMS"];



        #np.c_[x,y, porosity, porosity_effective, ntg, sw_base, sg_base, dsg, dsw];
        self.features_matrix_dsw = np.c_[self.x, self.y,self.int_dRMS_20];#np.c_[self.int_dRMS_20];#
        self.features_matrix_dsg = np.c_[self.x, self.y,self.int_dRMS_20];#np.c_[self.int_dRMS_20];#


        #CorrelationAnalysis(attributes_list, attributes_matrix, attributes_names);

        #linearRegression = LinearRegression(self.features_matrix_dsw, self.dsw, attribute_names_dsw, "Target: DSW");
        #multilayerPerceptron = MultilayerPerceptron(self.features_matrix_dsw, self.dsw, attribute_names_dsw, "Target: DSW");
        #svRegression = SVRegression(self.features_matrix_dsw, self.dsw, attribute_names_dsw, "Target: DSW");
        #xgBoost = XGBoost(self.features_matrix_dsw, self.dsw, attribute_names_dsw, "Target: DSW");
        randomForest = RandomForest(self.features_matrix_dsw, self.dsw, attribute_names_dsw, "Target: DSW");
        predicted_matrix = np.empty((self.x_size, self.y_size));
        predicted_matrix[:] = np.nan
        for x in range(len(randomForest.predicted_y)):
            predicted_matrix[int(self.col[x])-1][int(self.row[x])-1]= randomForest.predicted_y[x];
        plt.imshow(np.rot90(predicted_matrix), current_cmap,  vmin=-0.5, vmax=0.5);
        plt.colorbar()
        plt.show()

        #linearRegression2 = LinearRegression(self.features_matrix_dsg, self.dsg, attribute_names_dsg, "Target: DSG");
        #multilayerPerceptron2 = MultilayerPerceptron(self.features_matrix_dsg, self.dsg, attribute_names_dsg, "Target: DSG");
        #svRegression2 = SVRegression(self.features_matrix_dsg, self.dsg, attribute_names_dsg, "Target: DSG");
        #xgBoost2 = XGBoost(self.features_matrix_dsg, self.dsg, attribute_names_dsg, "Target: DSG");
        randomForest2 = RandomForest(self.features_matrix_dsg, self.dsg, attribute_names_dsg, "Target: DSG");
        predicted_matrix = np.empty((self.x_size, self.y_size));
        predicted_matrix[:] = np.nan
        for x in range(len(randomForest2.predicted_y)):
            predicted_matrix[int(self.col[x])-1][int(self.row[x])-1]= randomForest2.predicted_y[x];
        plt.imshow(np.rot90(predicted_matrix), current_cmap,  vmin=-0.05, vmax=0.05);
        plt.colorbar()
        plt.show()

if __name__ == '__main__':
    main = Main("");
    main.create_training_data();
    main.machine_learning_analysis();

    pass

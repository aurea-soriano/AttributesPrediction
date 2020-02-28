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
from machine_learning.RandomForestOtherTest import RandomForestOtherTest
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
        "Sg_Base", "dSg", "dSw", "fair_int_dRMS_10m", "fair_int_dRMS_20m", "fair_int_dRMS"];

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
        self.fair_int_dRMS = []
        self.fair_int_dRMS_10m = []
        self.xdRMS = []
        self.fair_int_dRMS_20m = []
        self.ydRMS = []
        self.attributes_list = []
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
        fair_int_dRMSMatrix_10m = FileReader.read("data/"+self.attributes_files[7], 20);
        xdRMS = fair_int_dRMSMatrix_10m[0].flatten();
        ydRMS = fair_int_dRMSMatrix_10m[1].flatten();
        fair_int_dRMS_10m= fair_int_dRMSMatrix_10m[2].flatten();

        # reading actual dRMS
        fair_int_dRMSMatrix_20m = FileReader.read("data/"+self.attributes_files[8], 20);
        xdRMS = fair_int_dRMSMatrix_20m[0].flatten();
        ydRMS = fair_int_dRMSMatrix_20m[1].flatten();
        fair_int_dRMS_20m= fair_int_dRMSMatrix_20m[2].flatten();

        # reading synthetic dRMS
        fair_int_dRMSMatrix = FileReader.read("data/"+self.attributes_files[9], 20);
        xdRMS = fair_int_dRMSMatrix[0].flatten();
        ydRMS = fair_int_dRMSMatrix[1].flatten();
        fair_int_dRMS= fair_int_dRMSMatrix[2].flatten();


        for i in range(len(x)):
            if math.isnan(x[i]) or math.isnan(y[i]) or math.isnan(dsg[i]) or math.isnan(dsw[i]) or math.isnan(xdRMS[i]) or  math.isnan(ydRMS[i]) or math.isnan(fair_int_dRMS_10m[i]) or math.isnan(fair_int_dRMS_20m[i]) or math.isnan(fair_int_dRMS[i]):
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
                self.fair_int_dRMS_10m.append(fair_int_dRMS_10m[i])
                self.fair_int_dRMS_20m.append(fair_int_dRMS_20m[i])
                self.fair_int_dRMS.append(fair_int_dRMS[i])

    def machine_learning_analysis(self):

        current_cmap = plt.get_cmap("seismic")
        current_cmap.set_bad(color='black')

        attribute_names_dsw = ["fair_int_dRMS"];
        attribute_names_dsg = ["fair_int_dRMS"];


        #np.c_[x,y, porosity, porosity_effective, ntg, sw_base, sg_base, dsg, dsw];
        self.features_matrix_dsw = np.c_[self.fair_int_dRMS];#
        self.features_matrix_dsg = np.c_[self.fair_int_dRMS];#



        randomForest = RandomForest(self.features_matrix_dsw, self.dsw, attribute_names_dsw, "Target: DSW");
        predicted_matrix = np.empty((self.x_size, self.y_size));
        predicted_matrix[:] = np.nan
        print(len(randomForest.predicted_y))
        for x in range(len(randomForest.predicted_y)):
            predicted_matrix[int(self.col[x])-1][int(self.row[x])-1]= randomForest.predicted_y[x];
        plt.imshow(np.rot90(predicted_matrix), current_cmap,  vmin=-0.5, vmax=0.5);
        plt.colorbar()
        plt.show()

        dsw_matrix = np.empty((self.x_size, self.y_size));
        dsw_matrix[:] = np.nan
        for x in range(len(self.dsw)):
            dsw_matrix[int(self.col[x])-1][int(self.row[x])-1]= self.dsw[x];
        plt.imshow(np.rot90(dsw_matrix), current_cmap,  vmin=-0.5, vmax=0.5);
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

        dsg_matrix = np.empty((self.x_size, self.y_size));
        dsg_matrix[:] = np.nan
        for x in range(len(self.dsg)):
            dsg_matrix[int(self.col[x])-1][int(self.row[x])-1]= self.dsg[x];
        plt.imshow(np.rot90(dsg_matrix), current_cmap,  vmin=-0.5, vmax=0.5);
        plt.colorbar()
        plt.show()


        drms_matrix = np.empty((self.x_size, self.y_size));
        drms_matrix[:] = np.nan
        for x in range(len(self.fair_int_dRMS)):
            drms_matrix[int(self.col[x])-1][int(self.row[x])-1]= self.fair_int_dRMS[x];
        plt.imshow(np.rot90(drms_matrix), current_cmap,  vmin=-50000, vmax=50000);
        plt.colorbar()
        plt.show()

        drms_matrix10 = np.empty((self.x_size, self.y_size));
        drms_matrix10[:] = np.nan
        for x in range(len(self.fair_int_dRMS_10m)):
            drms_matrix[int(self.col[x])-1][int(self.row[x])-1]= self.fair_int_dRMS_10m[x];
        plt.imshow(np.rot90(drms_matrix10), current_cmap,  vmin=-50000, vmax=50000);
        plt.colorbar()
        plt.show()


        drms_matrix20 = np.empty((self.x_size, self.y_size));
        drms_matrix20[:] = np.nan
        for x in range(len(self.fair_int_dRMS_20m)):
            drms_matrix[int(self.col[x])-1][int(self.row[x])-1]= self.fair_int_dRMS_20m[x];
        plt.imshow(np.rot90(drms_matrix20), current_cmap,  vmin=-50000, vmax=50000);
        plt.colorbar()
        plt.show()


        #testing
        features_matrix_dsw_test = np.c_[self.fair_int_dRMS_10m];#
        features_matrix_dsg_test = np.c_[self.fair_int_dRMS_10m];#

        randomForestOtherTest = RandomForestOtherTest(self.features_matrix_dsw, self.dsw, attribute_names_dsw, "Target: DSW", features_matrix_dsw_test);
        predicted_matrix = np.empty((self.x_size, self.y_size));
        predicted_matrix[:] = np.nan
        print(len(randomForestOtherTest.predicted_y))
        for x in range(len(randomForestOtherTest.predicted_y)):
            predicted_matrix[int(self.col[x])-1][int(self.row[x])-1]= randomForestOtherTest.predicted_y[x];
        plt.imshow(np.rot90(predicted_matrix), current_cmap,  vmin=-0.5, vmax=0.5);
        plt.colorbar()
        plt.show()

        randomForestOtherTest = RandomForestOtherTest(self.features_matrix_dsg, self.dsg, attribute_names_dsg, "Target: DSG", features_matrix_dsg_test);
        predicted_matrix = np.empty((self.x_size, self.y_size));
        predicted_matrix[:] = np.nan
        print(len(randomForestOtherTest.predicted_y))
        for x in range(len(randomForestOtherTest.predicted_y)):
            predicted_matrix[int(self.col[x])-1][int(self.row[x])-1]= randomForestOtherTest.predicted_y[x];
        plt.imshow(np.rot90(predicted_matrix), current_cmap,  vmin=-0.5, vmax=0.5);
        plt.colorbar()
        plt.show()


        features_matrix_dsw_test = np.c_[self.fair_int_dRMS_20m];#
        features_matrix_dsg_test = np.c_[self.fair_int_dRMS_20m];#

        randomForestOtherTest = RandomForestOtherTest(self.features_matrix_dsw, self.dsw, attribute_names_dsw, "Target: DSW", features_matrix_dsw_test);
        predicted_matrix = np.empty((self.x_size, self.y_size));
        predicted_matrix[:] = np.nan
        print(len(randomForestOtherTest.predicted_y))
        for x in range(len(randomForestOtherTest.predicted_y)):
            predicted_matrix[int(self.col[x])-1][int(self.row[x])-1]= randomForestOtherTest.predicted_y[x];
        plt.imshow(np.rot90(predicted_matrix), current_cmap,  vmin=-0.5, vmax=0.5);
        plt.colorbar()
        plt.show()

        randomForestOtherTest = RandomForestOtherTest(self.features_matrix_dsg, self.dsg, attribute_names_dsg, "Target: DSG", features_matrix_dsg_test);
        predicted_matrix = np.empty((self.x_size, self.y_size));
        predicted_matrix[:] = np.nan
        print(len(randomForestOtherTest.predicted_y))
        for x in range(len(randomForestOtherTest.predicted_y)):
            predicted_matrix[int(self.col[x])-1][int(self.row[x])-1]= randomForestOtherTest.predicted_y[x];
        plt.imshow(np.rot90(predicted_matrix), current_cmap,  vmin=-0.5, vmax=0.5);
        plt.colorbar()
        plt.show()

if __name__ == '__main__':
    main = Main("");
    main.create_training_data();
    main.machine_learning_analysis();

    pass

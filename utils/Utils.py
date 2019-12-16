'''
Created on Sep 11, 2019

@author: aurea
'''

import numpy as np
import scipy
from scipy import misc

class Utils(object):
    '''
    classdocs
    '''

    @staticmethod
    def average_drms_resolution():
        '''
        Constructor
        '''
        input_file = "../data/dRMS";
        output_file = "../data/avg_dRMS";
        fileReader = open(input_file, "r");
        fileWriter= open(output_file,"w+");
        count = 0;
        nonFlag = True; #True==4 False==3
        rows_vector = [];
        for line in fileReader:
            if(count<20):
                fileWriter.write(line);
            else:
                if(nonFlag==True):
                    if(len(rows_vector)<4):
                        text = line.split(' ');
                        vector = np.array(text);
                        vector = vector.astype(np.float);
                        rows_vector.append(vector)
                    else:
                        nonFlag= False;
                        result_vector = np.average(rows_vector, axis=0);
                        rows_vector = [];
                        text = line.split(' ');
                        vector = np.array(text);
                        vector = vector.astype(np.float);
                        rows_vector.append(vector)
                        for i in range(len(result_vector)):
                            fileWriter.write(str(result_vector[i]));
                            if(i  != len(result_vector)-1):
                                fileWriter.write(" ");
                        fileWriter.write("\n");
                else:
                    if(len(rows_vector)<3):
                        text = line.split(' ');
                        vector = np.array(text);
                        vector = vector.astype(np.float);
                        rows_vector.append(vector)
                    else:
                        nonFlag= True;
                        result_vector = np.average(rows_vector, axis=0);
                        rows_vector = [];
                        text = line.split(' ');
                        vector = np.array(text);
                        vector = vector.astype(np.float);
                        rows_vector.append(vector)
                        for i in range(len(result_vector)):
                            fileWriter.write(str(result_vector[i]));
                            if(i  != len(result_vector)-1):
                                fileWriter.write(" ");
                        fileWriter.write("\n");

            count+=1;
        if(len(rows_vector) > 0):
            result_vector = np.average(rows_vector, axis=0);
            rows_vector = [];
            for i in range(len(result_vector)):
                fileWriter.write(str(result_vector[i]));
                if(i  != len(result_vector)-1):
                    fileWriter.write(" ");
            fileWriter.write("\n");

        fileReader.close();
        fileWriter.close();

    @staticmethod
    def interpolated_drms_resolution():
        '''
        Constructor
        '''
        input_file = "../data/dRMS_20m";
        output_file = "../data/int_dRMS_20m";
        fileReader = open(input_file, "r");
        fileWriter= open(output_file,"w+");
        rows_vector = [];
        count = 0;
        for line in fileReader:
            if(count<20):
                fileWriter.write(line);
            else:
                text = line.split(' ');
                vector = np.array(text);
                vector = vector.astype(np.float);
                rows_vector.append(vector)
            count+=1;

        A = np.array(rows_vector)
        print(A.shape)

        B = misc.imresize(A, [18455,5], interp='bilinear', mode='F')

        print(B.shape)

        for i in range(0, B.shape[0]):
            for j in range(0, B.shape[1]):
                fileWriter.write(str(B[i][j]));
                if(j  != B.shape[1]-1):
                    fileWriter.write(" ");
            fileWriter.write("\n");

        fileReader.close();
        fileWriter.close();

    @staticmethod
    def fair_interpolated_drms_resolution(new_x, new_y, input_file, output_file):
            '''
            Constructor
            '''

            fileReader = open(input_file, "r");
            fileWriter= open(output_file,"w+");
            rows_vector = [];
            count = 0;

            old_x = 0;
            old_y = 0;
            matrix_x = [];
            matrix_y = [];
            matrix_value = [];
            matrix_col = [];
            matrix_row = [];

            for line in fileReader:
                if(count<20 and count!=13):
                    fileWriter.write(line);
                elif(count==13):
                    text = line.split(':');
                    text = text[1].split('x');
                    old_x = int(text[0]);
                    old_y = int(text[1]);
                    matrix_x = np.zeros((old_x, old_y));
                    matrix_y = np.zeros((old_x, old_y));
                    matrix_value = np.zeros((old_x, old_y));
                    matrix_col = np.zeros((old_x, old_y));
                    matrix_row = np.zeros((old_x, old_y));

                    fileWriter.write("# Grid_size: "+str(new_x)+ " x "+ str(new_y)+"\n");
                else:
                    text = line.split(' ');
                    matrix_x[int(text[3])-1][int(text[4])-1] = float(text[0]);
                    matrix_y[int(text[3])-1][int(text[4])-1] = float(text[1]);
                    matrix_value[int(text[3])-1][int(text[4])-1] = float(text[2]);
                    matrix_col[int(text[3])-1][int(text[4])-1] = float(text[3]);
                    matrix_row[int(text[3])-1][int(text[4])-1] = float(text[4]);
                count+=1;

            matrix_x = misc.imresize(matrix_x, [new_x,new_y], interp='bilinear', mode='F')
            matrix_y = misc.imresize(matrix_y, [new_x,new_y], interp='bilinear', mode='F')
            matrix_value = misc.imresize(matrix_value, [new_x,new_y], interp='bilinear', mode='F')
            matrix_col = misc.imresize(matrix_col, [new_x,new_y], interp='bilinear', mode='F')
            matrix_row = misc.imresize(matrix_row, [new_x,new_y], interp='bilinear', mode='F')

            for i in range(0, matrix_x.shape[0]):
                for j in range(0, matrix_x.shape[1]):
                    fileWriter.write(str(matrix_x[i][j])+" "+str(matrix_y[i][j])+" "+
                    str(matrix_value[i][j])+" "+str(i+1)+" "+str(j+1)+"\n");

            fileReader.close();
            fileWriter.close();

if __name__ == '__main__':
    #Utils.average_drms_resolution();
    #Utils.interpolated_drms_resolution();
    Utils.fair_interpolated_drms_resolution(193, 151, "../data/dRMS_10m", "../data/fair_int_dRMS_10m");

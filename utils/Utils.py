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
        input_file = "../data/dRMS";
        output_file = "../data/int_dRMS";
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
        
if __name__ == '__main__':
    #Utils.average_drms_resolution();
    Utils.interpolated_drms_resolution();
    
    
              
        
        
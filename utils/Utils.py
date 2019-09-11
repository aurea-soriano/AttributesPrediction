'''
Created on Sep 11, 2019

@author: aurea
'''

import numpy as np 
import scipy

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
        countRows =0;
        nonFlag = True; #True==4 False==3
        rows_vector = [];
        for line in fileReader: 
            if(count<20):
                fileWriter.write(line);
            else:
                if(nonFlag==True):
                    if(countRows<3):
                        text = line.split(' ');
                        vector = np.array(text);
                        vector = vector.astype(np.float);
                        rows_vector.append(vector)
                        countRows+=1;
                    else:
                        countRows =0;
                        nonFlag= False;
                        result_vector = np.average(rows_vector, axis=0);
                        rows_vector = [];
                        for result in result_vector:
                            fileWriter.write(str(result)+" ");
                        fileWriter.write("\n");
                else:
                    if(countRows<3):
                        text = line.split(' ');
                        vector = np.array(text);
                        vector = vector.astype(np.float);
                        rows_vector.append(vector)
                        countRows+=1;
                    else:
                        countRows =0;
                        nonFlag= True;
                        result_vector = np.average(rows_vector, axis=0);
                        rows_vector = [];
                        for result in result_vector:
                            fileWriter.write(str(result)+" ");
                        fileWriter.write("\n");
                        
            count+=1;        
       
        fileReader.close();
        fileWriter.close();
        
if __name__ == '__main__':
    Utils.average_drms_resolution();
              
        
        
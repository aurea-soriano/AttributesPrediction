'''
Created on Sep 11, 2019

@author: aurea
'''

import numpy as np

class FileWriter(object):
    '''
    classdocs
    '''
    @staticmethod   
    def write(text_name, attributes_names, matrix):
        file= open(text_name,"w+");
        if len(attributes_names)>0:
            for i in range(len(attributes_names)):
                file.write(attributes_names[i]);
                if(i< len(attributes_names)-1):
                    file.write(",");
                else:
                    file.write("\n");
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                file.write(str(matrix[i][j]) + " ");
            file.write("\n");
        file.close();
        
        
def main():
    matrix = [[1,2,3], [1,2,3]];
    FileWriter.write("../results/correlation_matrix.txt", ["x", "y", "Porosity", "Porosity-Effective Ref", "NTG", "Sw_base", "Sg_Base", "dSg", "dSw"], matrix);
             
if __name__ == '__main__':
    main();
    pass
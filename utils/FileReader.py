'''
Created on Sep 7, 2019

@author: aurea
'''

import numpy as np

class FileReader(object):
    '''
    classdocs
    '''
    @staticmethod   
    def read(text_name, from_line):
        resulting_vector = [];
        resulting_matrix = [];
        file = open(text_name, "r");
        count = 0;
        for line in file: 
            if count>from_line:
                text = line.split(' ');
                vector = np.array(text);
                vector = vector.astype(np.float);
                #print(text);
                resulting_vector.append(vector);
            count+=1
        resulting_matrix = np.mat(resulting_vector)
        file.close()
        return resulting_matrix;
        
        
def main():
    dsg = FileReader.read("../data/avg_dRMS", 20);#
    print(dsg)
             
if __name__ == '__main__':
    main();
    pass
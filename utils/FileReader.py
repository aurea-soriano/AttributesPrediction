'''
Created on Sep 7, 2019

@author: aurea
'''

import numpy as np

class FileReader(object):
    '''
    classdocs
    '''
    
    def __init__(self, _text_name, _from_line):
        '''
        Constructor
        '''
        self.text_name = _text_name;
        self.from_line = _from_line;
        self.resulting_vector = [];
        self.resulting_matrix = [];
   
    def read(self):
        file = open(self.text_name, "r");
        count = 0;
        for line in file: 
            if count>self.from_line:
                text = line.split(' ');
                vector = np.array(text);
                vector = vector.astype(np.float);
                #print(text);
                self.resulting_vector.append(vector);
            count+=1
        self.resulting_matrix = np.mat(self.resulting_vector)
        file.close()
        
        
def main():
    fileReader = FileReader("../data/dSg", 20);
    fileReader.read();
             
if __name__ == '__main__':
    main();
    pass
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
        resulting_matrices = [];
        matrix_x = [];
        matrix_y = [];
        matrix_value = [];
        matrix_col = [];
        matrix_row = [];
        file = open(text_name, "r");
        count = 0;
        for line in file:
            if count == 13:
                text = line.split(':');
                text = text[1].split('x');
                old_x = int(text[0]);
                old_y = int(text[1]);
                matrix_x = np.zeros((old_x, old_y));
                matrix_y = np.zeros((old_x, old_y));
                matrix_value = np.zeros((old_x, old_y));
                matrix_col = np.zeros((old_x, old_y));
                matrix_row = np.zeros((old_x, old_y));

            elif count>from_line:
                text = line.split(' ');
                text = line.split(' ');
                matrix_x[int(text[3])-1][int(text[4])-1] = float(text[0]);
                matrix_y[int(text[3])-1][int(text[4])-1] = float(text[1]);
                matrix_value[int(text[3])-1][int(text[4])-1] = float(text[2]);
                matrix_col[int(text[3])-1][int(text[4])-1] = float(text[3]);
                matrix_row[int(text[3])-1][int(text[4])-1] = float(text[4]);
            count+=1
        resulting_matrices.append(matrix_x);
        resulting_matrices.append(matrix_y);
        resulting_matrices.append(matrix_value);
        resulting_matrices.append(matrix_col);
        resulting_matrices.append(matrix_row);
        file.close()
        return resulting_matrices;


def main():
    dsg = FileReader.read("../data/avg_dRMS", 20);#
    print(dsg)

if __name__ == '__main__':
    main();
    pass

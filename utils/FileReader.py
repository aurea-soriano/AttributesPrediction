'''
Created on Sep 7, 2019

@author: aurea
'''

import numpy as np
import matplotlib.pyplot as plt

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
                matrix_x = np.empty((old_x, old_y));
                matrix_x[:] = np.nan
                matrix_y = np.empty((old_x, old_y));
                matrix_y[:] = np.nan
                matrix_value = np.empty((old_x, old_y));
                matrix_value[:] = np.nan
                matrix_col = np.empty((old_x, old_y));
                matrix_col[:] = np.nan
                matrix_row = np.empty((old_x, old_y));
                matrix_row[:] = np.nan

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
    current_cmap = plt.get_cmap("seismic")
    current_cmap.set_bad(color='black')
    ds2 = FileReader.read("../data/dSw", 20);#
    plt.imshow(np.rot90(ds2[2]), current_cmap,  vmin=-0.5, vmax=0.5);
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    main();
    pass

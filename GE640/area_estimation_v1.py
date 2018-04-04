# This script create area estimation results (example like accuracy assessment excel sheet) for the confusion matrix produced by crosstab.py
import numpy as np

def area_estimation(error_matrix_txt, map_area_txt, output_csv_file):
    ndv = -9999
    conf_matrix = np.genfromtxt(error_matrix_txt, delimiter=',')
    conf_matrix = np.transpose(conf_matrix[1:,1:])
    class_num = np.shape(conf_matrix)[0]
    print(conf_matrix)
    print(class_num)
    sample_counts_matrix = np.ones((class_num+1, class_num+3)) * ndv
    sample_counts_matrix[0:class_num, 0:class_num] = conf_matrix
    for i in np.arange(0,class_num):
        sample_counts_matrix[i, class_num] = np.sum(conf_matrix[i, :])
    for j in np.arange(0,class_num):
        sample_counts_matrix[class_num, j] = np.sum(conf_matrix[:, j])
    sample_counts_matrix[class_num, class_num] = np.sum(conf_matrix[:,:])

    map_area = np.genfromtxt(map_area_txt, delimiter=',')
    print(np.transpose(map_area[1:,1:]))
    sample_counts_matrix[:, class_num+1:class_num+3] = np.transpose(map_area[1:,1:])
    np.set_printoptions(suppress=True)
    print('sample_counts_matrix')
    print(sample_counts_matrix)

    proportions_matrix = np.ones((class_num+2, class_num+4)) * ndv
    for i in np.arange(0, class_num):
        for j in np.arange(0, class_num):
            proportions_matrix[i,j] = sample_counts_matrix[i,j] / sample_counts_matrix[i,class_num] *  sample_counts_matrix[i,class_num+2]
    for i in np.arange(0, class_num):
        proportions_matrix[i,class_num] = sum(proportions_matrix[i,0:class_num])
    for i in np.arange(0, class_num):
        proportions_matrix[i,class_num] = sum(proportions_matrix[i,0:class_num])
    #total
    for j in np.arange(0, class_num+1):
        proportions_matrix[class_num, j] = sum(proportions_matrix[0:class_num, j])
    # area estimates
    for j in np.arange(0, class_num):
        proportions_matrix[class_num+1,j] = proportions_matrix[class_num,j] * sample_counts_matrix[class_num,class_num+1]
    #calculate users's accuracy
    for i in np.arange(0, class_num):
        proportions_matrix[i,class_num+1] = proportions_matrix[i, i] / proportions_matrix[i, class_num]
    # calculate procuder's accuracy
    for j in np.arange(0, class_num):
        proportions_matrix[j,class_num+2] = proportions_matrix[j, j] / proportions_matrix[class_num, j]
    # calculate overall accuracy
    proportions_matrix[0, class_num + 3] = 0
    for j in np.arange(0, class_num):
        proportions_matrix[0, class_num + 3] += proportions_matrix[j, j]
    print('users accuracy')
    print(proportions_matrix[:,class_num+1])
    print('proportions_matrix')
    print(proportions_matrix)

    # calculate uncertainty of the estimates
    # =$H4^2*(C4/$F4)*(1-C4/$F4)/($F4-1)
    uncertainty_matrix = np.ones((class_num+2, class_num)) * ndv
    for i in np.arange(0, class_num):
        for j in np.arange(0, class_num):
            H4 = sample_counts_matrix[i, class_num+2]
            C4 = sample_counts_matrix[i,j]
            F4 = sample_counts_matrix[i, class_num]
            uncertainty_matrix[i,j] = H4 ** 2 * (C4 /F4)*(1 - C4 /F4) / (F4 - 1)

    #Standard errors
    for j in np.arange(0, class_num):
        uncertainty_matrix[class_num,j] = np.sqrt(np.sum(uncertainty_matrix[0:class_num,j]))
        uncertainty_matrix[class_num+1, j] = uncertainty_matrix[class_num, j] * sample_counts_matrix[class_num, class_num+1]
    # save the output of sample counts
    sample_counts_file = output_folder + "/sample_counts.txt"
    np.savetxt(sample_counts_file, sample_counts_matrix, delimiter=',',fmt='%.3f')

    #save the output of proportion matrix
    proportions_file = output_folder + "/proportions_matrix.txt"
    np.savetxt(proportions_file, proportions_matrix, delimiter=',', fmt='%.3f')

    #save the output of uncertainty matrix
    uncertainty_file = output_folder + "/uncertainty_matrix.txt"
    np.savetxt(uncertainty_file, uncertainty_matrix, delimiter=',', fmt='%.10f')

error_matrix_txt = r'/Users/shijuanchen/Desktop/Spring 2018/GE640 Digital Image Processing/labs/lab 6-8/errormatrix.txt'
map_area_txt = r'/Users/shijuanchen/Desktop/Spring 2018/GE640 Digital Image Processing/labs/lab 6-8/map_area.txt'
output_folder = r'/Users/shijuanchen/Desktop/Spring 2018/GE640 Digital Image Processing/labs/lab 6-8/accuracy_assess'
area_estimation(error_matrix_txt, map_area_txt, output_folder)
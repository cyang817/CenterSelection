'''******************ANALYZE THE RESULTS OF COEFFICIENTS SELECTION********************'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import os



'''DATA EXPLANATION'''
'''
keys: different combination of distance and radial basis functions.
     The values is a list containing two matrices. The first matrix is the
     mcc values of our method. The second matrix is the values of the CRBF.
     
BRC_results['Hamming_Gaussian'][0]: are the mcc values of our method using Hamming distance
                                    and Gaussian radial basis function. The rows are different number of
                                    centers. The columns are different independent experiments
                                    
BRC_results['Hamming_Gaussian'][1]: are the mcc values of the CRBF using Hamming distance
                                    and Gaussian radial basis function. The rows are different number of
                                    centers. The columns are different independent experiments
'''
'''*********PLOTS**************'''

def Draw_pictures():
    os.chdir('/home/leo/Documents/Project_SelectCenters/Code/Results/BRE')        
    BRC_results = pd.read_pickle('BRE_results_frame.pkl')        
    keys = BRC_results.keys()
    
    nCenters_list = np.array(BRC_results['nCenters_reg_lists'][0])
    np.log(nCenters_list)
    large = math.log(nCenters_list[0])
    
    for i in range(16):
        np.average(BRC_results[keys[i]][0], axis = 1)
        plt.plot(np.log(nCenters_list), np.average(BRC_results[keys[i]][0], axis = 1), linewidth = 2)
        plt.scatter(np.log(nCenters_list), np.average(BRC_results[keys[i]][0], axis = 1))
        plt.plot(np.log(nCenters_list), np.average(BRC_results[keys[i]][1], axis = 1), linewidth = 2)
        plt.scatter(np.log(nCenters_list), np.average(BRC_results[keys[i]][1], axis = 1))
        plt.legend(['Our method', 'CRBF'], fontsize = 'large')
        plt.xlabel('Log number of centers', fontsize='large')
        plt.ylabel('MCC', fontsize = 'large')
        plt.xlim([large, 0])
        plt.savefig('BRE'+'_'+keys[i])
        plt.close()




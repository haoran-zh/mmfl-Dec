#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 12:31:45 2023

@author: msiew
"""


# mmm (diffmodels) noniid 

# currently----
# noniid mcm
# iid mcf

# future----
# iid mcf mcf
# noniid mc mc mc 
# non iid m c_c100 m c c_100
# non iid m c f, m c f


import numpy as np
import matplotlib.pyplot as plt


def AvgAcc1trial(data1, data2, data3):
    avg_OurAlgo=np.mean(data1,axis=0)
    avg_Rand=np.mean(data2,axis=0)
    avg_RR=np.mean(data3,axis=0)
    
    return avg_OurAlgo, avg_Rand, avg_RR
    

    
# def MaxAcc1trial(data1, data2, data3, name1,name2, name3, numRounds):
#     avgdata1=np.max(data1,axis=0)
#     avgdata2=np.max(data2,axis=0)
#     avgdata3=np.max(data3,axis=0)
    
#     plt.plot(np.arange(0,100), avgdata1,label=name1)
#     plt.plot(np.arange(0,100), avgdata2, label=name2)
#     plt.plot(np.arange(0,100), avgdata3, label=name3)
#     plt.legend()
#     plt.title('Maximum Accuracy over multiple tasks')
#     plt.show()
    
def MinAcc1trial(data1, data2, data3):
    min_OurAlgo=np.min(data1,axis=0)
    min_Rand=np.min(data2,axis=0)
    min_RR=np.min(data3,axis=0)
    
    return min_OurAlgo, min_Rand, min_RR
    


def diff1trial(data1, data2,data3):
    diff_ourAlgo=np.max(data1,axis=0)-np.min(data1,axis=0)
    diff_rand=np.max(data2,axis=0)-np.min(data2,axis=0)
    diff_RR=np.max(data3,axis=0)-np.min(data3,axis=0)
    
    return diff_ourAlgo,diff_rand,diff_RR
    
def var1trial(data1, data2,data3):
    var_ourAlgo=np.var(data1,axis=0)
    var_rand=np.var(data2, axis=0)
    var_RR=np.var(data3, axis=0)
    
    return var_ourAlgo, var_rand, var_RR

def AvgTimeTaken_1trial(data1, data2,data3,numTasks):
      
    epsCheckpoints=[0.1,0.2,0.3,0.4, 0.5,0.6,0.7,0.8,0.9]
    epsReachedData1=np.zeros((numTasks,len(epsCheckpoints)))
    epsReachedData2=np.zeros((numTasks, len(epsCheckpoints)))
    epsReachedData3=np.zeros((numTasks, len(epsCheckpoints)))
    for b in range(numTasks):
        for a in range(len(epsCheckpoints)):
            indexdata1=np.searchsorted(data1[b,:], epsCheckpoints[a])
            #print(indexdata1)
            epsReachedData1[b,a]=indexdata1 if indexdata1 < len(data1[b,:]) else 102
            #print(epsReachedData1[a])
            
            indexdata2=np.searchsorted(data2[b,:], epsCheckpoints[a])
            epsReachedData2[b,a]=indexdata2 if indexdata2 < len(data2[b,:]) else 102
            
            indexdata3=np.searchsorted(data3[b,:], epsCheckpoints[a])
            epsReachedData3[b,a]=indexdata3 if indexdata3 < len(data3[b,:]) else 102

    
    return np.mean(epsReachedData1,axis=0), np.mean(epsReachedData2,axis=0), np.mean(epsReachedData3, axis=0)


def MaxTimeTaken_1trial(data1, data2,data3, numTasks):
   
    
    epsCheckpoints=[0.1,0.2,0.3,0.4, 0.5,0.6,0.7,0.8,0.9]
    epsReachedData1=np.zeros((numTasks,len(epsCheckpoints)))
    epsReachedData2=np.zeros((numTasks, len(epsCheckpoints)))
    epsReachedData3=np.zeros((numTasks, len(epsCheckpoints)))
    for b in range(numTasks):
        for a in range(len(epsCheckpoints)):
            indexdata1=np.searchsorted(data1[b,:], epsCheckpoints[a])
            #print(indexdata1)
            epsReachedData1[b,a]=indexdata1 if indexdata1 < len(data1[b,:]) else 102
            #print(epsReachedData1[a])
            
            indexdata2=np.searchsorted(data2[b,:], epsCheckpoints[a])
            epsReachedData2[b,a]=indexdata2 if indexdata2 < len(data2[b,:]) else 102
            
            indexdata3=np.searchsorted(data3[b,:], epsCheckpoints[a])
            epsReachedData3[b,a]=indexdata3 if indexdata3 < len(data3[b,:]) else 102

    
    return np.max(epsReachedData1,axis=0), np.max(epsReachedData2,axis=0), np.max(epsReachedData3, axis=0)



numRounds=120#100
# load 0th set
exp0_A3=np.load('mcf_i_globalAcc_exp0_algo0_e5_6pm.npy')
exp0_Rand=np.load('mcf_i_globalAcc_exp0_algo1_e5_6pm.npy')
exp0_RR=np.load('mcf_i_globalAcc_exp0_algo2_e5_6pm.npy')
exp0_RR=np.where(exp0_RR <=0, 0, exp0_RR)
#load 1st set
exp1_A3=np.load('mcf_i_globalAcc_exp1_algo0_e5_6pm.npy')
exp1_Rand=np.load('mcf_i_globalAcc_exp1_algo1_e5_6pm.npy')
exp1_RR=np.load('mcf_i_globalAcc_exp1_algo2_e5_6pm.npy')
exp1_RR=np.where(exp1_RR <=0, 0, exp1_RR)
#load 2nd set
exp2_A3=np.load('mcf_i_globalAcc_exp2_algo0_e5_6pm.npy')
exp2_Rand=np.load('mcf_i_globalAcc_exp2_algo1_e5_6pm.npy')
exp2_RR=np.load('mcf_i_globalAcc_exp2_algo2_e5_6pm.npy')
exp2_RR=np.where(exp2_RR <=0, 0, exp2_RR)
#load 3rd set
exp3_A3=np.load('mcf_i_globalAcc_exp3_algo0_e5_6pm.npy')
exp3_Rand=np.load('mcf_i_globalAcc_exp3_algo1_e5_6pm.npy')
exp3_RR=np.load('mcf_i_globalAcc_exp3_algo2_e5_6pm.npy')
exp3_RR=np.where(exp3_RR <=0, 0, exp3_RR)
#load original exp
exp4_A3=np.load('global_accB3.npy')
exp4_Rand=np.load('global_accRand.npy')
exp4_RR=np.load('global_accRR.npy')
exp4_RR=np.where(exp4_RR <=0, 0, exp4_RR)


plt.rcParams['font.size'] = 12

# average accuracy data and plots
Avg_e0_ourAlgo, Avg_e0_Rand, Avg_e0_RR =AvgAcc1trial(exp0_A3,exp0_Rand,exp0_RR)
Avg_e1_ourAlgo, Avg_e1_Rand, Avg_e1_RR =AvgAcc1trial(exp1_A3,exp1_Rand,exp1_RR)
Avg_e2_ourAlgo, Avg_e2_Rand, Avg_e2_RR =AvgAcc1trial(exp2_A3,exp2_Rand,exp2_RR)
Avg_e3_ourAlgo, Avg_e3_Rand, Avg_e3_RR =AvgAcc1trial(exp3_A3,exp3_Rand,exp3_RR)
Avg_e4_ourAlgo, Avg_e4_Rand, Avg_e4_RR =AvgAcc1trial(exp4_A3,exp4_Rand,exp4_RR)

Avg_e1_ourAlgo_AVG=(Avg_e0_ourAlgo+Avg_e1_ourAlgo+Avg_e2_ourAlgo+Avg_e3_ourAlgo)/4
Avg_e1_Rand_AVG=(Avg_e0_Rand+Avg_e1_Rand+Avg_e2_Rand+Avg_e3_Rand)/4
Avg_e1_RR_AVG=(Avg_e0_RR+Avg_e1_RR+Avg_e2_RR+Avg_e3_RR)/4

plt.plot(np.arange(0,numRounds), Avg_e1_ourAlgo_AVG,label='Alpha=3, alpha fair allocation')
plt.plot(np.arange(0,numRounds), Avg_e1_Rand_AVG, label='Random allocation of tasks')
plt.plot(np.arange(0,numRounds), Avg_e1_RR_AVG, label='Round Robin alocation of tasks')
plt.legend()
plt.ylabel('Average Accuracy')
plt.xlabel('Num. Global Iterations')
plt.grid(linestyle='--', linewidth=0.5)
plt.title('Average Accuracy over 3 Tasks, Average of 4 Runs')
plt.savefig('mcf_i_avgAcc.png')
plt.show()


# min acc data and plots
Min_e0_ourAlgo, Min_e0_Rand, Min_e0_RR =MinAcc1trial(exp0_A3,exp0_Rand,exp0_RR)
Min_e1_ourAlgo, Min_e1_Rand, Min_e1_RR =MinAcc1trial(exp1_A3,exp1_Rand,exp1_RR)
Min_e2_ourAlgo, Min_e2_Rand, Min_e2_RR =MinAcc1trial(exp2_A3,exp2_Rand,exp2_RR)
Min_e3_ourAlgo, Min_e3_Rand, Min_e3_RR =MinAcc1trial(exp3_A3,exp3_Rand,exp3_RR)
Min_e4_ourAlgo, Min_e4_Rand, Min_e4_RR =MinAcc1trial(exp4_A3,exp4_Rand,exp4_RR)

Min_e1_ourAlgo_AVG=(Min_e0_ourAlgo+Min_e1_ourAlgo+Min_e2_ourAlgo+Min_e3_ourAlgo)/4
Min_e1_Rand_AVG=(Min_e0_Rand+Min_e1_Rand+Min_e2_Rand+Min_e3_Rand)/4
Min_e1_RR_AVG=(Min_e0_RR+ Min_e1_RR+ Min_e2_RR+ Min_e3_RR)/4

plt.plot(np.arange(0,numRounds), Min_e1_ourAlgo_AVG,label='Alpha=3, alpha fair allocation')
plt.plot(np.arange(0,numRounds), Min_e1_Rand_AVG, label='Random allocation of tasks')
plt.plot(np.arange(0,numRounds), Min_e1_RR_AVG, label='Round Robin alocation of tasks')
plt.legend()
plt.ylabel('Minimum Accuracy')
plt.xlabel('Num. Global Iterations')
plt.grid(linestyle='--', linewidth=0.5)
plt.title('Minimum Accuracy over 3 Tasks, Average of 4 Runs')
plt.savefig('mcf_i_minAcc.png')
plt.show()


# variance acc data and plots
Var_e0_ourAlgo, Var_e0_Rand, Var_e0_RR =var1trial(exp0_A3,exp0_Rand,exp0_RR)
Var_e1_ourAlgo, Var_e1_Rand, Var_e1_RR =var1trial(exp1_A3,exp1_Rand,exp1_RR)
Var_e2_ourAlgo, Var_e2_Rand, Var_e2_RR =var1trial(exp2_A3,exp2_Rand,exp2_RR)
Var_e3_ourAlgo, Var_e3_Rand, Var_e3_RR =var1trial(exp3_A3,exp3_Rand,exp3_RR)
Var_e4_ourAlgo, Var_e4_Rand, Var_e4_RR =var1trial(exp4_A3,exp4_Rand,exp4_RR)

Var_e1_ourAlgo_AVG=(Var_e0_ourAlgo+ Var_e1_ourAlgo+Var_e2_ourAlgo+ Var_e3_ourAlgo)/4
Var_e1_Rand_AVG=(Var_e0_Rand+ Var_e1_Rand+Var_e2_Rand+ Var_e3_Rand)/4
Var_e1_RR_AVG=(Var_e0_RR+ Var_e1_RR+ Var_e2_RR+ Var_e3_RR)/4

plt.plot(np.arange(0,numRounds), Var_e1_ourAlgo_AVG,label='Alpha=3, alpha fair allocation')
plt.plot(np.arange(0,numRounds), Var_e1_Rand_AVG, label='Random allocation of tasks')
plt.plot(np.arange(0,numRounds), Var_e1_RR_AVG, label='Round Robin alocation of tasks')
plt.legend()
plt.ylabel('Variance')
plt.xlabel('Num. Global Iterations')
plt.grid(linestyle='--', linewidth=0.5)
plt.title('Variance over 3 Tasks, Average of 4 Runs')
plt.savefig('mcf_i_var.png')
plt.show()


# Average time taken
Atime_e0_ourAlgo, Atime_e0_Rand, Atime_e0_RR =AvgTimeTaken_1trial(exp0_A3,exp0_Rand,exp0_RR, numTasks=3)
Atime_e1_ourAlgo, Atime_e1_Rand, Atime_e1_RR =AvgTimeTaken_1trial(exp1_A3,exp1_Rand,exp1_RR, numTasks=3)
Atime_e2_ourAlgo, Atime_e2_Rand, Atime_e2_RR =AvgTimeTaken_1trial(exp2_A3,exp2_Rand,exp2_RR, numTasks=3)
Atime_e3_ourAlgo, Atime_e3_Rand, Atime_e3_RR =AvgTimeTaken_1trial(exp3_A3,exp3_Rand,exp3_RR, numTasks=3)
Atime_e4_ourAlgo, Atime_e4_Rand, Atime_e4_RR =AvgTimeTaken_1trial(exp4_A3,exp4_Rand,exp4_RR, numTasks=3)

Atime_e1_ourAlgo_AVG=(Atime_e0_ourAlgo+ Atime_e1_ourAlgo+Atime_e2_ourAlgo+ Atime_e3_ourAlgo)/4
Atime_e1_Rand_AVG=(Atime_e0_Rand+ Atime_e1_Rand+ Atime_e2_Rand+ Atime_e3_Rand)/4
Atime_e1_RR_AVG=(Atime_e0_RR+ Atime_e1_RR+ Atime_e2_RR+ Atime_e3_RR)/4

epsCheckpoints=[0.1,0.2,0.3,0.4, 0.5,0.6,0.7,0.8,0.9]
plt.plot(epsCheckpoints, Atime_e1_ourAlgo_AVG,'o-', label='Alpha=3, alpha fair allocation')
plt.plot(epsCheckpoints, Atime_e1_Rand_AVG, 'v-', label='Random allocation of tasks')
plt.plot(epsCheckpoints, Atime_e1_RR_AVG, '.-', label='Round Robin alocation of tasks')
plt.xlabel('Accuracy level eps')
plt.ylabel('Time taken in Num. global iterations')
plt.legend()
plt.grid(linestyle='--', linewidth=0.5)
plt.title('Average time taken for All Tasks to reach eps, Average of 4 Runs')
plt.savefig('mcf_i_avgTimeTaken.png')
plt.show()



# Maximum Time taken 
Mtime_e0_ourAlgo, Mtime_e0_Rand, Mtime_e0_RR =MaxTimeTaken_1trial(exp0_A3,exp0_Rand,exp0_RR, numTasks=3)
Mtime_e1_ourAlgo, Mtime_e1_Rand, Mtime_e1_RR =MaxTimeTaken_1trial(exp1_A3,exp1_Rand,exp1_RR, numTasks=3)
Mtime_e2_ourAlgo, Mtime_e2_Rand, Mtime_e2_RR =MaxTimeTaken_1trial(exp2_A3,exp2_Rand,exp2_RR, numTasks=3)
Mtime_e3_ourAlgo, Mtime_e3_Rand, Mtime_e3_RR =MaxTimeTaken_1trial(exp3_A3,exp3_Rand,exp3_RR, numTasks=3)
Mtime_e4_ourAlgo, Mtime_e4_Rand, Mtime_e4_RR =MaxTimeTaken_1trial(exp4_A3,exp4_Rand,exp4_RR, numTasks=3)

Mtime_e1_ourAlgo_AVG=(Mtime_e0_ourAlgo+ Mtime_e1_ourAlgo+Mtime_e2_ourAlgo+ Mtime_e3_ourAlgo)/4
Mtime_e1_Rand_AVG=(Mtime_e0_Rand+ Mtime_e1_Rand+ Mtime_e2_Rand+ Mtime_e3_Rand)/4
Mtime_e1_RR_AVG=(Mtime_e0_RR+ Mtime_e1_RR+ Mtime_e2_RR+ Mtime_e3_RR)/4

epsCheckpoints=[0.1,0.2,0.3,0.4, 0.5,0.6,0.7,0.8,0.9]
plt.plot(epsCheckpoints, Mtime_e1_ourAlgo_AVG,'o-', label='Alpha=3, alpha fair allocation')
plt.plot(epsCheckpoints, Mtime_e1_Rand_AVG, 'v-',  label='Random allocation of tasks')
plt.plot(epsCheckpoints, Mtime_e1_RR_AVG, '.-',  label='Round Robin alocation of tasks')
plt.xlabel('Accuracy level eps')
plt.ylabel('Time taken (Num. Global Epochs)')
plt.legend()
plt.grid(linestyle='--', linewidth=0.5)
plt.title('Max time for All Tasks to reach eps, Average of 4 Runs')
#plt.rcParams['font.size'] = 18
plt.savefig('mcf_i_maxTimeTaken.png')
plt.show()



# plt.plot(globAccB3[0,:], label='task 0, beta=3') 
# plt.legend()
# plt.show()

# # plt.plot(globAccB3[1,:], label='task 1, beta=3') 
# # plt.legend()
# # plt.show()

# # plt.plot(globAccB3[2,:], label='task 2, beta=3') 
# # plt.legend()
# # plt.show()

# # plt.plot(globAccRand[0,:], label='task 0, rand BL') 
# # plt.legend()
# # plt.show()

# # plt.plot(globAccRand[1,:], label='task 1, rand BL') 
# # plt.legend()
# # plt.show()


# # plt.plot(globAccRand[2,:], label='task 2, rand BL') 
# # plt.legend()
# # plt.show()
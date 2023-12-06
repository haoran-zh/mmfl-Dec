#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 16:08:00 2023

@author: msiew
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('MacOSX')

def AvgAcc(data1, data2, data3, name1,name2, name3, numRounds):
    avgdata1=np.mean(data1,axis=0)
    avgdata2=np.mean(data2,axis=0)
    avgdata3=np.mean(data3,axis=0)
    
    plt.plot(np.arange(0,numRounds), avgdata1,label=name1)
    plt.plot(np.arange(0,numRounds), avgdata2, label=name2)
    plt.plot(np.arange(0,numRounds), avgdata3, label=name3)
    plt.legend()
    plt.title('Average Accuracy over multiple tasks')
    plt.show()
    
def MaxAcc(data1, data2, data3, name1,name2, name3, numRounds):
    avgdata1=np.max(data1,axis=0)
    avgdata2=np.max(data2,axis=0)
    avgdata3=np.max(data3,axis=0)
    
    plt.plot(np.arange(0,numRounds), avgdata1,label=name1)
    plt.plot(np.arange(0,numRounds), avgdata2, label=name2)
    plt.plot(np.arange(0,numRounds), avgdata3, label=name3)
    plt.legend()
    plt.title('Maximum Accuracy over multiple tasks')
    plt.show()
    
def MinAcc(data1, data2, data3, name1,name2, name3, numRounds):
    avgdata1=np.min(data1,axis=0)
    avgdata2=np.min(data2,axis=0)
    avgdata3=np.min(data3,axis=0)
    
    plt.plot(np.arange(0,numRounds), avgdata1,label=name1)
    plt.plot(np.arange(0,numRounds), avgdata2, label=name2)
    plt.plot(np.arange(0,numRounds), avgdata3, label=name3)
    plt.legend()
    plt.title('Minimum Accuracy over multiple tasks')
    plt.show()
    
def AvgAcc3data(data1, data2,data3, name1,name2, name3, numRounds):
    avgdata1=np.mean(data1,axis=0)
    avgdata2=np.mean(data2,axis=0)
    avgdata3=np.mean(data3,axis=0)
    
    plt.plot(np.arange(0,numRounds), avgdata1,label=name1)
    plt.plot(np.arange(0,numRounds), avgdata2, label=name2)
    plt.plot(np.arange(0,numRounds), avgdata3, label=name3)
    plt.legend()
    plt.title('Average Accuracy over multiple tasks')
    plt.show()
    
def MinAcc3data(data1, data2,data3,name1,name2, name3, numRounds):
    avgdata1=np.min(data1,axis=0)
    avgdata2=np.min(data2,axis=0)
    avgdata3=np.min(data3,axis=0)
    
    plt.plot(np.arange(0,numRounds), avgdata1,label=name1)
    plt.plot(np.arange(0,numRounds), avgdata2, label=name2)
    plt.plot(np.arange(0,numRounds), avgdata3, label=name3)
    plt.legend()
    plt.title('Minimum Accuracy over multiple tasks')
    plt.show()

def difference(data1, data2,data3,name1,name2, name3, numRounds):
    diff_data1=np.max(data1,axis=0)-np.min(data1,axis=0)
    diff_data2=np.max(data2,axis=0)-np.min(data2,axis=0)
    diff_data3=np.max(data3,axis=0)-np.min(data3,axis=0)
    
    plt.plot(np.arange(0,numRounds), diff_data1,label=name1)
    plt.plot(np.arange(0,numRounds), diff_data2, label=name2)
    plt.plot(np.arange(0,numRounds), diff_data3, label=name3)
    
    plt.legend()
    plt.title('Difference in accuracy over tasks')
    plt.show()
    
def var(data1, data2,data3,name1,name2, name3, numRounds):
    var_data1=np.var(data1,axis=0)
    var_data2=np.var(data2, axis=0)
    var_data3=np.var(data3, axis=0)
    
    plt.plot(np.arange(0,numRounds), var_data1,label=name1)
    plt.plot(np.arange(0,numRounds), var_data2, label=name2)
    plt.plot(np.arange(0,numRounds), var_data3, label=name3)
    
    plt.legend()
    plt.title('Variance in accuracy over tasks')
    
    plt.show()

def AvgAcc_TimeTaken(data1, data2,data3,name1,name2, name3,numRounds,numTasks):
   
    
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

    
    
    plt.plot(epsCheckpoints, np.mean(epsReachedData1, axis=0),'o',label=name1, )
    plt.plot(epsCheckpoints, np.mean(epsReachedData2, axis=0), '.', label=name2)
    plt.plot(epsCheckpoints, np.mean(epsReachedData3, axis=0), '*', label=name3)

    plt.legend()
    plt.title('Average time taken to reach eps')
    plt.xlabel('eps')
    plt.show()
    
    return np.mean(epsReachedData1, axis=0), np.mean(epsReachedData2, axis=0)


def MaxAcc_TimeTaken(data1, data2,data3,name1,name2, name3, numRounds,numTasks):
   
    
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

    
    
    plt.plot(epsCheckpoints, np.max(epsReachedData1, axis=0),'o',label=name1, )
    plt.plot(epsCheckpoints, np.max(epsReachedData2, axis=0), '.', label=name2)
    plt.plot(epsCheckpoints, np.max(epsReachedData3, axis=0), '*', label=name2)
    plt.legend()
    plt.title('Maximum time taken to reach eps')
    plt.xlabel('eps')
    plt.show()
    
    #return np.mean(epsReachedData1, axis=0), np.mean(epsReachedData2, axis=0)
    

# def AvgAcc(data1, data2,name1,name2, numRounds):
#     avgdata1=np.mean(data1,axis=0)
#     avgdata2=np.mean(data2,axis=0)
    
#     plt.plot(np.arange(0,numRounds), avgdata1,label=name1)
#     plt.plot(np.arange(0,numRounds), avgdata2, label=name2)
#     plt.legend()
#     plt.title('Average Accuracy over multiple tasks')
#     plt.show()
    
# def MinAcc(data1, data2,name1,name2, numRounds):
#     avgdata1=np.min(data1,axis=0)
#     avgdata2=np.min(data2,axis=0)
    
#     plt.plot(np.arange(0,numRounds), avgdata1,label=name1)
#     plt.plot(np.arange(0,numRounds), avgdata2, label=name2)
#     plt.legend()
#     plt.title('Minimum Accuracy over multiple tasks')
#     plt.show()
    
# def AvgAcc3tasks(data1, data2,data3, name1,name2, name3, numRounds):
#     avgdata1=np.mean(data1,axis=0)
#     avgdata2=np.mean(data2,axis=0)
#     avgdata3=np.mean(data3,axis=0)
    
#     plt.plot(np.arange(0,numRounds), avgdata1,label=name1)
#     plt.plot(np.arange(0,numRounds), avgdata2, label=name2)
#     plt.plot(np.arange(0,numRounds), avgdata3, label=name3)
#     plt.legend()
#     plt.title('Average Accuracy over multiple tasks')
#     plt.show()
    
# def MinAcc3tasks(data1, data2,data3,name1,name2, name3, numRounds):
#     avgdata1=np.min(data1,axis=0)
#     avgdata2=np.min(data2,axis=0)
#     avgdata3=np.min(data3,axis=0)
    
#     plt.plot(np.arange(0,numRounds), avgdata1,label=name1)
#     plt.plot(np.arange(0,numRounds), avgdata2, label=name2)
#     plt.plot(np.arange(0,numRounds), avgdata3, label=name3)
#     plt.legend()
#     plt.title('Minimum Accuracy over multiple tasks')
#     plt.show()

# def difference(data1,data2, name1,name2,numRounds):
#     #diff_data1=np.abs(data1[0]-data1[1])
#     #diff_data2=np.abs(data2[0]-data2[1])
#     diff_data1=np.max(data1,axis=0)-np.min(data1,axis=0)
#     diff_data2=np.max(data2,axis=0)-np.min(data2,axis=0)
#     #diff_data3=np.max(data3,axis=0)-np.min(data3,axis=0)
    
    
#     plt.plot(np.arange(0,numRounds), diff_data1,label=name1)
#     plt.plot(np.arange(0,numRounds), diff_data2, label=name2)
    
#     plt.legend()
#     plt.title('Difference in accuracy over tasks')
#     plt.show()
    
# def var(data1,data2, name1,name2,numRounds):
#     var_data1=np.var(data1,axis=0)
#     var_data2=np.var(data2, axis=0)
    
#     plt.plot(np.arange(0,numRounds), var_data1,label=name1)
#     plt.plot(np.arange(0,numRounds), var_data2, label=name2)
    
#     plt.legend()
#     plt.title('Variance in accuracy over tasks')
    
#     plt.show()

# def AvgAcc_TimeTaken(data1, data2,name1,name2, numRounds,numTasks):
   
    
#     epsCheckpoints=[0.1,0.2,0.3,0.4, 0.5,0.6,0.7,0.8,0.9]
#     epsReachedData1=np.zeros((numTasks,len(epsCheckpoints)))
#     epsReachedData2=np.zeros((numTasks, len(epsCheckpoints)))
#     for b in range(numTasks):
#         for a in range(len(epsCheckpoints)):
#             indexdata1=np.searchsorted(data1[b,:], epsCheckpoints[a])
#             #print(indexdata1)
#             epsReachedData1[b,a]=indexdata1 if indexdata1 < len(data1[b,:]) else 102
#             #print(epsReachedData1[a])
            
#             indexdata2=np.searchsorted(data2[b,:], epsCheckpoints[a])
#             epsReachedData2[b,a]=indexdata2 if indexdata2 < len(data2[b,:]) else 102

    
    
#     plt.plot(epsCheckpoints, np.mean(epsReachedData1, axis=0),'o',label=name1, )
#     plt.plot(epsCheckpoints, np.mean(epsReachedData2, axis=0), '.', label=name2)
#     plt.legend()
#     plt.title('Average time taken to reach eps')
#     plt.xlabel('eps')
#     plt.show()
    
#     return np.mean(epsReachedData1, axis=0), np.mean(epsReachedData2, axis=0)


# def MaxAcc_TimeTaken(data1, data2,name1,name2, numRounds,numTasks):
   
    
#     epsCheckpoints=[0.1,0.2,0.3,0.4, 0.5,0.6,0.7,0.8,0.9]
#     epsReachedData1=np.zeros((numTasks,len(epsCheckpoints)))
#     epsReachedData2=np.zeros((numTasks, len(epsCheckpoints)))
#     for b in range(numTasks):
#         for a in range(len(epsCheckpoints)):
#             indexdata1=np.searchsorted(data1[b,:], epsCheckpoints[a])
#             #print(indexdata1)
#             epsReachedData1[b,a]=indexdata1 if indexdata1 < len(data1[b,:]) else 102
#             #print(epsReachedData1[a])
            
#             indexdata2=np.searchsorted(data2[b,:], epsCheckpoints[a])
#             epsReachedData2[b,a]=indexdata2 if indexdata2 < len(data2[b,:]) else 102

    
    
#     plt.plot(epsCheckpoints, np.max(epsReachedData1, axis=0),'o',label=name1, )
#     plt.plot(epsCheckpoints, np.max(epsReachedData2, axis=0), '.', label=name2)
#     plt.legend()
#     plt.title('Maximum time taken to reach eps')
#     plt.xlabel('eps')
#     plt.show()
    
#     #return np.mean(epsReachedData1, axis=0), np.mean(epsReachedData2, axis=0)
    


numRounds=100#92#100

#original
#globAccB2=np.load('global_accB2.npy')
#globAccB3=np.load('global_accB3_2.npy')
#globAccB3=np.load('global_accB3.npy')
globAccB3=np.load('global_accB3.npy')
globAccB3=globAccB3[:, 0:100]
globAccRand=np.load('global_accRand.npy')
#globAccRand=globAccRand[:,0:92]
#globAccRand=np.load('global_accRand_new.npy')
globAccRR=np.load('global_accRR.npy')
#globAccRR=globAccRR[:,0:92]
globAccRR=np.where(globAccRR <=0, 0, globAccRR)
#globAccB4=np.load('global_accB4.npy')

#globAccRand=np.load('global_accRAND_invUsersAggr.npy')

#Exp run 2. [3,5,3]
# numRounds=100#120
# globAccB3=np.load('mcf_i_globalAcc_exp0_algo0_1.npy')
# globAccB3=globAccB3[:, 0:100]
# globAccRand=np.load('mcf_i_globalAcc_exp0_algo1_1.npy')
# globAccRand=globAccRand[:,0:100]
# globAccRR=np.load('mcf_i_globalAcc_exp3_algo2.npy')
# globAccRR=globAccRR[:,0:100]
# globAccRR=np.where(globAccRR <=0, 0, globAccRR)

# #Exp 3. [1,5,1]
# numRounds=100#120
# globAccB3=np.load('mcf_i_globalAcc_exp0_algo0.npy')
# globAccB3=globAccB3[:, 0:100]
# globAccRand=np.load('mcf_i_globalAcc_exp0_algo1.npy')
# globAccRand=globAccRand[:,0:100]
# globAccRR=np.load('mcf_i_globalAcc_exp3_algo2.npy')
# globAccRR=globAccRR[:,0:100]
# globAccRR=np.where(globAccRR <=0, 0, globAccRR)

# exp 4. [5,5,5] again
numRounds=100#120
globAccB3=np.load('mcf_i_globalAcc_exp0_algo0_e5.npy')
globAccB3=globAccB3[:, 0:100]
globAccRand=np.load('mcf_i_globalAcc_exp0_algo1_e5.npy')
globAccRand=globAccRand[:,0:100]
globAccRR=np.load('global_accRR.npy')
globAccRR=globAccRR[:,0:100]
globAccRR=np.where(globAccRR <=0, 0, globAccRR)
#print('last few, t0' , globAccB3[0,95:100])
#print('last few, t1' , globAccB3[0,95:100])

#exp [5,5,5]
#numRounds=100#120
numRounds=120

globAccB3=np.load('mcf_i_globalAcc_exp3_algo0_e5_6pm.npy')
#globAccB3=globAccB3[:, 0:100]
globAccRand=np.load('mcf_i_globalAcc_exp3_algo1_e5_6pm.npy')
#globAccRand=globAccRand[:,0:100]
globAccRR=np.load('mcf_i_globalAcc_exp3_algo2_e5_6pm.npy')
#globAccRR=globAccRR[:,0:100]
globAccRR=np.where(globAccRR <=0, 0, globAccRR)
#print('last few, t0' , globAccB3[0,95:100])
#print('last few, t1' , globAccB3[0,95:100])
#mcf_i_globalAcc_exp0_algo2_e5_6pm


AvgAcc(globAccB3,globAccRand,globAccRR, 'beta = 3', 'rand BL','round robin BL', numRounds)

MinAcc(globAccB3,globAccRand,globAccRR, 'beta = 3', 'rand BL','round robin BL',numRounds)

#AvgAcc3tasks(globAccB3,globAccB2,globAccRand,'beta=3','beta=2', 'rand BL',numRounds)

#MinAcc3tasks(globAccB3,globAccB2, globAccRand,'beta=3','beta=2', 'rand BL',numRounds)

difference(globAccB3,globAccRand,globAccRR, 'beta = 3', 'rand BL','round robin BL',numRounds)

var(globAccB3,globAccRand,globAccRR, 'beta = 3', 'rand BL','round robin BL',numRounds)

data1time, data2time=AvgAcc_TimeTaken(globAccB3,globAccRand,globAccRR, 'beta = 3', 'rand BL','round robin BL',numRounds,numTasks=2)

MaxAcc_TimeTaken(globAccB3,globAccRand,globAccRR, 'beta = 3', 'rand BL','round robin BL',numRounds,numTasks=3)


# AvgAcc(globAccB3,globAccRR,'beta = 3', 'RR BL',numRounds)

# MinAcc(globAccB3,globAccRR,'beta = 3', 'RR BL',numRounds)

# #AvgAcc3tasks(globAccB3,globAccB2,globAccRand,'beta=3','beta=2', 'rand BL',numRounds)

# #MinAcc3tasks(globAccB3,globAccB2, globAccRand,'beta=3','beta=2', 'rand BL',numRounds)

# difference(globAccB3,globAccRR,'beta = 3', 'RR BL',numRounds)

# var(globAccB3,globAccRR,'beta = 3', 'RR BL',numRounds)

# data1time, data2time=AvgAcc_TimeTaken(globAccB3,globAccRR,'beta = 3', 'RR BL',numRounds,numTasks=2)

# MaxAcc_TimeTaken(globAccB3,globAccRR,'beta = 3', 'RR BL',numRounds,numTasks=2)

####B4

# AvgAcc(globAccB4,globAccRand, 'beta = 4', 'rand BL',numRounds)

# MinAcc(globAccB4,globAccRand,'beta = 4', 'rand BL',numRounds)

# #AvgAcc3tasks(globAccB3,globAccB2,globAccRand,'beta=3','beta=2', 'rand BL',numRounds)

# #MinAcc3tasks(globAccB3,globAccB2, globAccRand,'beta=3','beta=2', 'rand BL',numRounds)

# difference(globAccB4,globAccRand,'beta = 4', 'rand BL',numRounds)

# var(globAccB4,globAccRand,'beta = 4', 'rand BL',numRounds)

# data1time, data2time=AvgAcc_TimeTaken(globAccB4,globAccRand,'beta = 4', 'rand BL',numRounds,numTasks=2)

# MaxAcc_TimeTaken(globAccB4,globAccRand,'beta = 4', 'rand BL',numRounds,numTasks=2)


#Exp 3. [1,5,1]
numRounds=100#120
globAccB3_e1=np.load('mcf_i_globalAcc_exp2_algo0.npy')
globAccB3_e1=globAccB3_e1[:, 0:100]
globAccRand_e1=np.load('mcf_i_globalAcc_exp2_algo1.npy')
globAccRand_e1=globAccRand_e1[:,0:100]
globAccRR_e1=np.load('mcf_i_globalAcc_exp2_algo2.npy')
globAccRR_e1=globAccRR_e1[:,0:100]
globAccRR_e1=np.where(globAccRR_e1 <=0, 0, globAccRR_e1)

plt.plot(np.arange(0,numRounds), globAccB3[0,:],label='beta=3')
plt.plot(np.arange(0,numRounds), globAccRand[0,:],label='rand')
plt.plot(np.arange(0,numRounds), globAccRR[0,:],label='RR')
plt.plot(np.arange(0,numRounds), globAccB3_e1[0,:],label='beta=3, e1')
plt.plot(np.arange(0,numRounds), globAccRand_e1[0,:],label='rand, e1')
plt.plot(np.arange(0,numRounds), globAccRR_e1[0,:],label='RR, e1')
plt.legend()
plt.title('task 0')
plt.show()

plt.plot(np.arange(0,numRounds), globAccB3[1,:],label='beta=3')
plt.plot(np.arange(0,numRounds), globAccRand[1,:],label='rand')
plt.plot(np.arange(0,numRounds), globAccRR[1,:],label='RR')

plt.plot(np.arange(0,numRounds), globAccB3_e1[1,:],label='beta=3, e1')
plt.plot(np.arange(0,numRounds), globAccRand_e1[1,:],label='rand, e1')
plt.plot(np.arange(0,numRounds), globAccRR_e1[1,:],label='RR, e1')
plt.title('task 1')
plt.legend()
plt.show()

plt.plot(np.arange(0,numRounds), globAccB3[2,:],label='beta=3')
plt.plot(np.arange(0,numRounds), globAccRand[2,:],label='rand')
plt.plot(np.arange(0,numRounds), globAccRR[2,:],label='RR')
plt.plot(np.arange(0,numRounds), globAccB3_e1[2,:],label='beta=3, e1')
plt.plot(np.arange(0,numRounds), globAccRand_e1[2,:],label='rand, e1')
plt.plot(np.arange(0,numRounds), globAccRR_e1[2,:],label='RR, e1')
plt.title('task 2')
plt.legend()
plt.show()

# plt.plot(globAccB3[1,:], label='task 1, beta=3') 
# plt.legend()
# plt.show()

# plt.plot(globAccB3[2,:], label='task 2, beta=3') 
# plt.legend()
# plt.show()

# plt.plot(globAccRand[0,:], label='task 0, rand BL') 
# plt.legend()
# plt.show()

# plt.plot(globAccRand[1,:], label='task 1, rand BL') 
# plt.legend()
# plt.show()


# plt.plot(globAccRand[2,:], label='task 2, rand BL') 
# plt.legend()
# plt.show()

# plt.plot(globAccRR[0,:], label='task 0, RR BL') 
# plt.legend()
# plt.show()

# plt.plot(globAccRR[1,:], label='task 1, RR BL') 
# plt.legend()
# plt.show()


# plt.plot(globAccRR[2,:], label='task 2, RR BL') 
# plt.legend()
# plt.show()



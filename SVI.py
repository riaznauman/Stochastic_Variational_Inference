import time
import sys
# sys.path.append('././libraryModules')
# from libModule import classificationHyp as classificationHypTemp1, writeToExcelFile, detailResults
from datetime import datetime
import spams
import math
import numpy as np
import random as rd
import scipy.io
from scipy import stats
from numpy import matlib, newaxis
import sklearn.preprocessing as sk
import matplotlib.pyplot as plt
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.datasets import make_sparse_coded_signal
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from scipy import *
from scipy import sparse
from scipy.sparse.linalg import inv
import os
def display_iteration_detail_to_screen(trainToTestLabelsCount,classWiseAccuracy, kCoff,datasetsName,algosettings,accuracy, trainTime, trainRateTime, testRateTime, initDictSize,finalDictSize,smean):
    #print("============================================================================================\n")
    print(datasetsName)
    print(algosettings)
    print("[Training/Test Samples: " + str([i for i in trainToTestLabelsCount]) +"]")
    print("[ClassWise Accuracy: " + str(["%.2f" % elem for elem in classWiseAccuracy.tolist()])+ "%]")
    print(f"[ClassWise Average Accuracy: {classWiseAccuracy.sum()/size(classWiseAccuracy): 6.2f} %]")
    print(f"[K Cofficient: {kCoff:6.2f} %]")
    print(f"[Recognition rate: {(accuracy * 100):6.2f} %]")
    print(f"[Mean recognition rate: {(smean):6.2f} %]")
    print(f"[Training time: {(trainTime/60):6.2f} mins.]")
    print(f"[Training time per example: {(1000*trainRateTime):6.2f} milli-sec.]")
    print(f"[Classification time per sample: {(1000 * testRateTime):6.2f} milli-sec.]")
    print(f"[Initial dictionary size: {initDictSize:5d} atoms]")
    print(f"[Final dictionary size: {finalDictSize:5d} atoms]")
def write_iteration_detail_to_file(trainToTestLabelsCount,classWiseAccuracy, kCoff,ff,strn,algosettings,accuracy, trainTime, trainRateTime, testRateTime,  initDictSize, finalDictSize,smean):
    ff.write(strn)
    ff.write(algosettings + "\n")
    ff.write("[Training/Test Samples: " + str([i for i in trainToTestLabelsCount]) +"]\n")
    ff.write("[ClassWise Accuracy: " + str(["%.2f" % elem for elem in classWiseAccuracy.tolist()])+ "%]\n")
    ff.write(f"[ClassWise Average Accuracy: {classWiseAccuracy.sum()/size(classWiseAccuracy): 6.2f} %]\n")
    ff.write(f"[K Cofficient: {kCoff:6.2f} %]\n")
    ff.write(f"[Recognition rate: {(accuracy * 100):6.2f} %]\n")
    ff.write(f"[Mean recognition rate: {(smean):6.2f} %]\n")
    ff.write(f"[Training time: {(trainTime/60):6.2f} mins.]\n")
    ff.write(f"[Training time per example: {(1000*trainRateTime):6.2f} milli-sec.]\n")
    ff.write(f"[Classification time per sample: {(1000 * testRateTime):6.2f} milli-sec.]\n")
    ff.write(f"[Initial dictionary size: {initDictSize:5d} atoms]\n")
    ff.write(f"[Final dictionary size: {finalDictSize:5d} atoms]\n")
def write_summary_to_file(trainToTestLabelsCount,classWiseAcc, kCof,f, strn, algosettings,iter1,mean,sdev,meand, averageTrainTime,averageTrainRateTime, averageTestRateTime,dataBaseName):
    f.write(strn)
    f.write(algosettings + "\n")
    f.write("[Training/Test Samples: " + str([i for i in trainToTestLabelsCount]) + "]\n")
    f.write("[ClassWise Accuracy: " + str(["%.2f" % elem for elem in classWiseAcc.tolist()])+ "%]\n")
    f.write(f"[ClassWise Average Accuracy: {classWiseAcc.sum()/size(classWiseAcc): 6.2f} %]\n")
    f.write(f"[K Cofficient: {kCof:6.2f} %]\n")
    f.write(f"Average iterations = {iter1:4.0f} Nos\n")
    f.write(f"Average accuracy = {mean:6.2f} %\n")
    f.write(f"Standard Deviation = {sdev:5.2f}\n")
    f.write(f"Average dictionary size = {meand:5.0f}\n")
    f.write(f"Average training time = {averageTrainTime/60:8.2f} mins.\n")
    f.write(f"Average training time per example = {1000*averageTrainRateTime:8.2f} milli-sec.\n")
    f.write(f"Averge classification time per sample = {1000*averageTestRateTime:8.2f} milli-sec.\n")
    # modelName="JSDC_p"
    # strn=strn+ f" Average Final Dict Size = {meand:5.0f}, Av Tr.Time= {averageTrainTime/60:8.2f} mins., Av Tr.time per example = {1000*averageTrainRateTime:8.2f} milli-sec., Av classification time per sample = {1000*averageTestRateTime:8.2f} milli-sec."
    # writeToExcelFile(modelName, mean, kCof, classWiseAcc.sum()/size(classWiseAcc),classWiseAcc, trainToTestLabelsCount, dataBaseName,strn,"./HyperSpectral_Classification/results.xlsx")
def display_summary_to_screen(trainToTestLabelsCount,classWiseAcc, kCof,strn,algosettings, iter1, mean, sdev, meand, averageTrainTime,averageTrainRateTime, averageTestRateTime):
    print(strn)
    print(algosettings + "\n")
    print("[Training/Test Samples: " + str([i for i in trainToTestLabelsCount]) +"]\n")
    print("[ClassWise Accuracy: " + str(["%.2f" % elem for elem in classWiseAcc.tolist()])+ "%]\n")
    print(f"[ClassWise Average Accuracy: {classWiseAcc.sum()/size(classWiseAcc): 6.2f} %]\n")
    print(f"[K Cofficient: {kCof:6.2f} %]\n")
    print(f"Average iterations = {iter1:4.0f} Nos\n")
    print(f"Average accuracy = {mean:6.2f} %\n")
    print(f"Standard Deviation = {sdev:5.2f}\n")
    print(f"Average dictionary size = {meand:5.0f}\n")
    print(f"Average training time = {averageTrainTime/60:8.2f} mins.\n")
    print(f"Average training time per example = {1000*averageTrainRateTime:8.2f} milli-sec.\n")
    print(f"Averge classification time per sample = {1000*averageTestRateTime:8.2f} ms\n")
def train_test_data(features,labels,trainSamplePerCent):
    classes=labels.shape[0]
    trainColIndex=list()
    b = np.arange(0, labels.shape[1])
    for c in range(classes):
        classIndex=[x0 for x0, val in enumerate(b) if labels[c,val]!=0]  #[x0 for x0, val in enumerate(labels[c, :]) if val != 0]
        if len(classIndex)<50:
            trainColIndex = trainColIndex + rd.sample(classIndex, 10)
        else:
            trainColIndex = trainColIndex + rd.sample(classIndex, math.ceil(trainSamplePerCent*size(classIndex)))
    testColIndex=[x0 for x0, val in enumerate(b) if x0 not in trainColIndex]
    training_samples=features[:,trainColIndex]
    train_label=labels[:,trainColIndex]
    test=features[:,testColIndex]
    test_label=labels[:,testColIndex]
    dataIndex =testColIndex.copy()
    return training_samples, train_label, test, test_label,dataIndex
def SVI_main(iiii,dicSize,DB,Code_DPath,resultFile,DBdisc,dataBaseName,p0,kk,k0,mi,gibi,sparsity,a0,b0,c0,d0,e0,f0,g_eps_d,g_eps_w,g_sd,batchsize,percentTrainSamples,patchLength,foperation="w+"):
    f = open(Code_DPath+resultFile, foperation)
    abc=np.load(DB)
    features = abc['features']
    labels=abc['labels']
    rcounter=0
    counter=0
    meand=0.0
    averageTrainTime=0.0
    averageTrainRateTime=0.0
    averageTestRateTime=0.0
    mean=0.0
    iter1=0
    oiter1=0
    classWiseAcc=np.zeros((labels.shape[0]), dtype=np.float64)
    kCof=0.0
    sl=[]
    fh=0
    while counter<10:
        f = open(Code_DPath+resultFile, foperation)
        strn = ""
        training_samples, train_label, test, test_label, dataIndex  =train_test_data(features, labels, percentTrainSamples)
        kCoff, classWiseAccuracy,algosettings, iterations,accuracy, trainTime, trainRateTime, testRateTime, initDictSize, finalDictSize = SVI(abc,batchsize,patchLength,DBdisc,dicSize,iiii,training_samples, train_label, test,test_label,p0,kk,k0,mi,gibi,sparsity,a0,b0,c0,d0,e0,f0,g_eps_d,g_eps_w,g_sd,dataIndex)
        classWiseAcc=classWiseAcc+classWiseAccuracy
        kCof=kCof+kCoff
        mean=mean+(accuracy*100)
        sl = sl + [accuracy * 100]
        iter1 = iter1 + iterations
        counter = counter + 1
        rcounter = rcounter + 1
        smean=mean/rcounter
        meand = meand + finalDictSize
        averageTrainTime = averageTrainTime + trainTime
        averageTrainRateTime = averageTrainRateTime + trainRateTime
        averageTestRateTime=averageTestRateTime+testRateTime
        if fh==0:
            f.write("\n\n\n###Experiment for " + dataBaseName + "dataset ("+str(datetime.now())+")###\n")
            trainIdices=np.argmax(train_label,0)
            trainIdicesCount=[trainIdices.tolist().count(i) for  i in range(train_label.shape[0])]
            testIndices=np.argmax(test_label,0)
            testIndicesCount=[testIndices.tolist().count(i) for  i in range(train_label.shape[0])]
            trainToTestLabelsCount=[str(i) +'/' + str(j) for i,j in zip(trainIdicesCount, testIndicesCount)]
            fh=1  
        strn=f"\n======================Experiment No. {rcounter:3d} of " + dataBaseName + "===============\n"
        write_iteration_detail_to_file(trainToTestLabelsCount,classWiseAccuracy, kCoff,f, strn, algosettings,accuracy, trainTime, trainRateTime, testRateTime, initDictSize, finalDictSize, smean)
        display_iteration_detail_to_screen(trainToTestLabelsCount,classWiseAccuracy, kCoff,strn,algosettings,accuracy, trainTime, trainRateTime, testRateTime, initDictSize, finalDictSize,smean)
        f.close()
    classWiseAcc=classWiseAcc/rcounter
    kCof=kCof/rcounter
    classWiseAverage=classWiseAcc.sum()/size(classWiseAcc)
    mean=mean/rcounter
    sl=np.asarray(sl)
    sdev=sum((sl-mean)**2)
    sdev=np.sqrt(sdev/rcounter)
    meand=meand/rcounter
    averageTrainTime = averageTrainTime/rcounter
    averageTrainRateTime = averageTrainRateTime/rcounter
    averageTestRateTime=averageTestRateTime/rcounter
    iter1=iter1/rcounter
    f = open(Code_DPath+resultFile, foperation)
    strn=f"Summary of {int(rcounter):4d} experiments of {dataBaseName} dataset:- k0 = {k0:.0e}, p0 = {p0:4.3f}, kk = {kk:4.3f},Iterations = {int(iter1):4d}, sparsity = {sparsity:4d}, trainPersentage = {percentTrainSamples*100:4.3f}, Initial Directory Size = {round(train_label.shape[1]*1.25):4d}"
    write_summary_to_file(trainToTestLabelsCount,classWiseAcc, kCof,f, strn, algosettings,iter1,mean,sdev,meand, averageTrainTime,averageTrainRateTime, averageTestRateTime,dataBaseName)
    display_summary_to_screen(trainToTestLabelsCount,classWiseAcc, kCof,strn, algosettings, iter1, mean, sdev, meand, averageTrainTime,averageTrainRateTime, averageTestRateTime)
    f.close()
    return 0
def initDSeqSelection(train, label, k):
    m, n = train.shape
    if k < n + 1:
        ichh = list(range(k))
    elif k > n:
        a = list(range(n))
        b = list(range(k - n))
        ichh = a + b
    Dinit = train[:, ichh]
    label = label[:, ichh]
    sb=Dinit-np.mean(Dinit,0)[np.newaxis,:]
    Dinit = sk.normalize(sb, norm='l2', axis=0)
    return Dinit, label
def initClassifier(A, label):
    lmbda = 1
    C = A.dot(A.T) + lmbda * np.eye(A.shape[0])
    W = np.linalg.inv(C).dot(A.dot(label.T))
    W = W.T
    W = W-np.mean(W,0)[np.newaxis,:]
    return W
def ompAlgo(Di,train_s,sparsity):
    np.random.seed(0)
    X = np.asfortranarray(train_s, dtype=np.float64)
    D=np.asfortranarray(sk.normalize(Di, norm='l2', axis=0), dtype=np.float64) #try
    eps = 1.0
    numThreads = -1
    Alpha_init=spams.omp(X, D,L=sparsity,return_reg_path=False).toarray()
    return Alpha_init
def detailResults(estMat, Hlabel,data):
    estIndices=np.argmax(estMat,0)
    estIndicesCounts=[estIndices.tolist().count(i) for  i in range(Hlabel.shape[0])]
    refIndices=np.argmax(Hlabel,0)
    refIndicesCounts=[refIndices.tolist().count(i) for  i in range(Hlabel.shape[0])]
    matchnum=np.sum(estIndices==refIndices)
    ab=[i for i,j in zip(estIndices.tolist(),refIndices.tolist()) if i==j]
    actualLabelsCount=[ab.count(i) for  i in range(Hlabel.shape[0])]
    accuracy = (matchnum*1.0) / (Hlabel.shape[1])
    R=(np.asarray(actualLabelsCount)/Hlabel.shape[1])*(np.asarray(refIndicesCounts)/Hlabel.shape[1])
    R=R.sum()
    print(R)
    K=(accuracy-R)/(1-R)
    classWiseAccuracy=(np.asarray(actualLabelsCount)/np.asarray(refIndicesCounts))*100
    return K, classWiseAccuracy,accuracy
def classification(D, W, data, Hlabel, sparsity,dataIndex, OrgIndex,OrgData,ColumnLength,gt_data,sz):
    Dn = sk.normalize(D, norm='l2', axis=0)  # normc(D); %To take advantage of Cholesky based OMP
    A=ompAlgo(Dn, OrgData[:,gt_data>0], sparsity)
    forAnorms = np.matlib.repmat(np.sqrt(np.sum(D ** 2, axis=0)), A.shape[1], 1)
    A = A / forAnorms.T  # To adjust the sparse codes according to D's normalization
    A=W.dot(A)
    B=np.zeros((A.shape[0],gt_data.shape[0],gt_data.shape[1]), dtype=np.float64)
    B[:,gt_data>0]=A
    window=np.ones((sz,sz),dtype=np.float64)
    for i in range(B.shape[0]):
        B[i,:,:]=scipy.ndimage.convolve(B[i,:,:],window, mode='nearest')
    estMat=B[:,gt_data>0]
    estMat=estMat[:,dataIndex]
    kCoff, classWiseAccuracy,accuracy = detailResults(estMat, Hlabel,data)
    return kCoff, classWiseAccuracy,accuracy 
########################################################################################################
####(Functions for Gibb Sampling)#####
def sample_D_to_VI_Input(Xd, D, Sd, g_eps_d, Z, Hw, W,g_eps_w,batchsizee):
    Dv=np.zeros(D.shape,dtype=np.float64)
    Wv=np.zeros(W.shape,dtype=np.float64)
    D_avg = np.zeros(D.shape, dtype=np.float64)
    W_avg = np.zeros(W.shape, dtype=np.float64)
    m = Xd.shape[0]
    c = Hw.shape[0]
    K = D.shape[1]
    batchsize=0
    indexAtoms=list()
    if batchsizee<=0 or batchsizee>K:
        batchsize=K
    else:
        batchsize=batchsizee
    for i in range(0,K, batchsize):
        
        if (i+batchsize)<=K:
            indexAtoms=list(range(i,i+batchsize))
        else:
            indexAtoms=list(range(i,K))
        Xdd = Xd + D[:, indexAtoms].dot(Z[:, indexAtoms].T*Sd[:, indexAtoms].T)
        Hwd = Hw + W[:, indexAtoms].dot(Z[:, indexAtoms].T*Sd[:, indexAtoms].T)
        Sds=np.sum((Z[:,indexAtoms]*Sd[:,indexAtoms])**2,0)[np.newaxis,:]
        sig_Dk = 1 / (g_eps_d * Sds + m)
        if batchsize==1:
            mu_Dk = g_eps_d * sig_Dk * (Xdd.dot(Z[:,indexAtoms]*Sd[:,indexAtoms]))
        else:
            mu_Dk = g_eps_d * sig_Dk * ((Xd.dot(Z[:,indexAtoms]*Sd[:,indexAtoms]))+(D[:,indexAtoms]*Sds))
        D[:,indexAtoms] = (mu_Dk + np.random.randn(D.shape[0],len(indexAtoms)) * np.sqrt(sig_Dk))
        D_avg[:, indexAtoms] = mu_Dk
        Dv[:, indexAtoms] = sig_Dk
        sig_Dk_w = 1 / (g_eps_w * Sds + c)
        if batchsize==1:
            mu_Dk_w = g_eps_w * sig_Dk_w * (Hwd.dot(Z[:,indexAtoms]*Sd[:,indexAtoms]))
        else:
            mu_Dk_w = g_eps_w * sig_Dk_w * ((Hw.dot(Z[:,indexAtoms]*Sd[:,indexAtoms]))+(W[:,indexAtoms]*Sds))
        W[:,indexAtoms] = (mu_Dk_w + np.random.randn(W.shape[0],len(indexAtoms)) * np.sqrt(sig_Dk_w))
        W_avg[:, indexAtoms] = mu_Dk_w
        Wv[:, indexAtoms] = sig_Dk_w
        Xd = Xdd - D[:, indexAtoms].dot(Z[:, indexAtoms].T*Sd[:, indexAtoms].T)
        Hw = Hwd - W[:, indexAtoms].dot(Z[:, indexAtoms].T*Sd[:, indexAtoms].T)
    return D_avg,Dv,W_avg,Wv
def sample_D(Xd, D, Sd, g_eps_d, Z, Hw, W, Sw, g_eps_w,batchsizee):
    m = Xd.shape[0]
    c = Hw.shape[0]
    K = D.shape[1]
    batchsize=0
    indexAtoms=list()
    if batchsizee<=0 or batchsizee>K:
        batchsize=K
    else:
        batchsize=batchsizee
    for i in range(0,K, batchsize):
        if (i+batchsize)<=K:
            indexAtoms=list(range(i,i+batchsize))
        else:
            indexAtoms=list(range(i,K))
        Xdd = Xd + D[:, indexAtoms].dot(Z[:, indexAtoms].T*Sd[:, indexAtoms].T)
        Hwd = Hw + W[:, indexAtoms].dot(Z[:, indexAtoms].T*Sd[:, indexAtoms].T)
        Sds=np.sum((Z[:,indexAtoms]*Sd[:,indexAtoms])**2,0)[np.newaxis,:]
        sig_Dk = 1 / (g_eps_d * Sds + m)
        if batchsize==1:
            mu_Dk = g_eps_d * sig_Dk * (Xdd.dot(Z[:,indexAtoms]*Sd[:,indexAtoms]))
        else:
            mu_Dk = g_eps_d * sig_Dk * ((Xd.dot(Z[:,indexAtoms]*Sd[:,indexAtoms]))+(D[:,indexAtoms]*Sds))
        D[:,indexAtoms] = (mu_Dk + np.random.randn(D.shape[0],len(indexAtoms)) * np.sqrt(sig_Dk))
        sig_Dk_w = 1 / (g_eps_w * Sds + c)
        if batchsize==1:
            mu_Dk_w = g_eps_w * sig_Dk_w * (Hwd.dot(Z[:,indexAtoms]*Sd[:,indexAtoms]))
        else:
            mu_Dk_w = g_eps_w * sig_Dk_w * ((Hw.dot(Z[:,indexAtoms]*Sd[:,indexAtoms]))+(W[:,indexAtoms]*Sds))
        W[:,indexAtoms] = (mu_Dk_w + np.random.randn(W.shape[0],len(indexAtoms)) * np.sqrt(sig_Dk_w))
        Xd = Xdd - D[:, indexAtoms].dot(Z[:, indexAtoms].T*Sd[:, indexAtoms].T)
        Hw = Hwd - W[:, indexAtoms].dot(Z[:, indexAtoms].T*Sd[:, indexAtoms].T)
    return Xd, D, Hw, W
def sample_ZS(Xd, D, Sd, Z, Pid, g_sd, g_eps_d, Hw, W, Sw, g_sw, g_eps_w,batchsizee):
    K = D.shape[1] 
    batchsize=0
    indexAtoms=list()
    if batchsizee<=0 or batchsizee>K:
        batchsize=K
    else:
        batchsize=batchsizee
    for i in range(0,K, batchsize):
        if (i+batchsize)<=K:
            indexAtoms=list(range(i,i+batchsize))
        else:
            indexAtoms=list(range(i,K))
        Xdd = Xd + D[:, indexAtoms].dot(Z[:, indexAtoms].T*Sd[:, indexAtoms].T)
        Hwd = Hw + W[:, indexAtoms].dot(Z[:, indexAtoms].T*Sd[:, indexAtoms].T)
        DTD = np.sum(D[:, indexAtoms] ** 2,0)[np.newaxis,:]
        WTW = np.sum(W[:, indexAtoms] ** 2,0)[np.newaxis,:]
        sigS1d = 1. / (g_sd + (g_eps_d*(Z[:, indexAtoms]**2) * DTD+g_eps_w*(Z[:, indexAtoms]**2) * WTW))
        if batchsize==1:
            SdM=sigS1d * (g_eps_d * Z[:, indexAtoms]*((Xdd.T).dot(D[:, indexAtoms]))+g_eps_w * Z[:, indexAtoms]*((Hwd.T).dot(W[:, indexAtoms])))
        else:
            SdM=sigS1d * (g_eps_d * Z[:, indexAtoms]*((Xd.T).dot(D[:, indexAtoms])+((Z[:, indexAtoms]*Sd[:, indexAtoms])*DTD))+g_eps_w * Z[:, indexAtoms]*((Hw.T).dot(W[:, indexAtoms])+((Z[:, indexAtoms]*Sd[:, indexAtoms])*WTW)))
        Sd[:, indexAtoms] = np.random.randn(Sd.shape[0], len(indexAtoms)) * np.sqrt(sigS1d) + SdM
        if batchsize==1:
            temp1 = - 0.5 * g_eps_d * ((Sd[:, indexAtoms] ** 2) * DTD - 2 * Sd[:, indexAtoms] * ((Xdd.T).dot(D[:, indexAtoms])))
            temp2 = - 0.5 * g_eps_w * ((Sd[:, indexAtoms] ** 2) * WTW - 2 * Sd[:, indexAtoms] * ((Hwd.T).dot(W[:, indexAtoms])))
        else:
            temp1 = - 0.5 * g_eps_d * ((Sd[:, indexAtoms] ** 2) * DTD - 2 * Sd[:, indexAtoms] * ((Xd.T).dot(D[:, indexAtoms])+((Z[:, indexAtoms]*Sd[:, indexAtoms])*DTD)))
            temp2 = - 0.5 * g_eps_w * ((Sd[:, indexAtoms] ** 2) * WTW - 2 * Sd[:, indexAtoms] * ((Hw.T).dot(W[:, indexAtoms])+((Z[:, indexAtoms]*Sd[:, indexAtoms])*WTW)))
        temp =Pid[:, indexAtoms]*np.exp(temp1+temp2)
        A=np.random.rand(Z.shape[0],len(indexAtoms))
        B=Z[:, indexAtoms]  
        B[A >  ((1 - Pid[:, indexAtoms]) / (temp + 1 - Pid[:, indexAtoms]))] = 1
        B[A <= ((1 - Pid[:, indexAtoms]) / (temp + 1 - Pid[:, indexAtoms]))] = 0 
        Z[:, indexAtoms]=B
        
        Xd = Xdd - D[:, indexAtoms].dot(Z[:, indexAtoms].T*Sd[:, indexAtoms].T)
        Hw = Hwd - W[:, indexAtoms].dot(Z[:, indexAtoms].T*Sd[:, indexAtoms].T)
    return Xd, Hw, Sd, Sw, Z
def sample_ZS_to_VI_Input(Xd, D, Sd, Z,Pid,g_sd, g_eps_d, Hw, W,g_eps_w,batchsizee):
    Z_avg=np.zeros(Z.shape,dtype=np.float64)
    Sd_avg=np.zeros(Sd.shape, dtype=np.float64)
    Sdv = np.zeros(Sd.shape, dtype=np.float64)
    K = D.shape[1] 
    batchsize=0
    indexAtoms=list()
    if batchsizee<=0 or batchsizee>K:
        batchsize=K
    else:
        batchsize=batchsizee
    for i in range(0,K, batchsize):
        if (i+batchsize)<=K:
            indexAtoms=list(range(i,i+batchsize))
        else:
            indexAtoms=list(range(i,K))
        Xdd = Xd + D[:, indexAtoms].dot(Z[:, indexAtoms].T*Sd[:, indexAtoms].T)
        Hwd = Hw + W[:, indexAtoms].dot(Z[:, indexAtoms].T*Sd[:, indexAtoms].T)
    
        DTD = np.sum(D[:, indexAtoms] ** 2,0)[np.newaxis,:]
        WTW = np.sum(W[:, indexAtoms] ** 2,0)[np.newaxis,:]
        sigS1d = 1. / (g_sd + (g_eps_d*(Z[:, indexAtoms]**2) * DTD+g_eps_w*(Z[:, indexAtoms]**2) * WTW))
        if batchsize==1:
            SdM=sigS1d * (g_eps_d * Z[:, indexAtoms]*((Xdd.T).dot(D[:, indexAtoms]))+g_eps_w * Z[:, indexAtoms]*((Hwd.T).dot(W[:, indexAtoms])))
        else:
            SdM=sigS1d * (g_eps_d * Z[:, indexAtoms]*((Xd.T).dot(D[:, indexAtoms])+((Z[:, indexAtoms]*Sd[:, indexAtoms])*DTD))+g_eps_w * Z[:, indexAtoms]*((Hw.T).dot(W[:, indexAtoms])+((Z[:, indexAtoms]*Sd[:, indexAtoms])*WTW)))
        
        Sd[:, indexAtoms] = np.random.randn(Sd.shape[0], len(indexAtoms)) * np.sqrt(sigS1d) + SdM
        Sd_avg[:, indexAtoms]=SdM
        Sdv[:, indexAtoms] = sigS1d
        if batchsize==1:
            temp1 = - 0.5 * g_eps_d * ((Sd[:, indexAtoms] ** 2) * DTD - 2 * Sd[:, indexAtoms] * ((Xdd.T).dot(D[:, indexAtoms])))
            temp2 = - 0.5 * g_eps_w * ((Sd[:, indexAtoms] ** 2) * WTW - 2 * Sd[:, indexAtoms] * ((Hwd.T).dot(W[:, indexAtoms])))
        else:
            temp1 = - 0.5 * g_eps_d * ((Sd[:, indexAtoms] ** 2) * DTD - 2 * Sd[:, indexAtoms] * ((Xd.T).dot(D[:, indexAtoms])+((Z[:, indexAtoms]*Sd[:, indexAtoms])*DTD)))
            temp2 = - 0.5 * g_eps_w * ((Sd[:, indexAtoms] ** 2) * WTW - 2 * Sd[:, indexAtoms] * ((Hw.T).dot(W[:, indexAtoms])+((Z[:, indexAtoms]*Sd[:, indexAtoms])*WTW)))
        temp =Pid[:, indexAtoms]*np.exp(temp1+temp2)
        A=np.random.rand(Z.shape[0],len(indexAtoms))
        B=Z[:, indexAtoms]
        B[A >  ((1 - Pid[:, indexAtoms]) / (temp + 1 - Pid[:, indexAtoms]))] = 1
        B[A <= ((1 - Pid[:, indexAtoms]) / (temp + 1 - Pid[:, indexAtoms]))] = 0 
        Z[:, indexAtoms]=B
        Z_avg[:,indexAtoms]=1.0-((1 - Pid[:, indexAtoms]) / (temp + 1 - Pid[:, indexAtoms]))
        Xd = Xdd - D[:, indexAtoms].dot(Z[:, indexAtoms].T*Sd[:, indexAtoms].T)
        Hw = Hwd - W[:, indexAtoms].dot(Z[:, indexAtoms].T*Sd[:, indexAtoms].T)          
    return Sd_avg,Sdv,Z_avg
def sample_Pi(train_label, Z,Pid,Pi_C, a0, b0,Xdclist):
    sumZ=train_label.dot(Z)
    K = Z.shape[1]
    Pi_C = np.random.beta((sumZ + (a0*1.0 / K)), ((b0 * 1.0*(K - 1) / K) + (np.sum(train_label, 1)[:,np.newaxis] - sumZ)))
    Pid=train_label.T.dot(Pi_C)
    return Pid,Pi_C
def sample_Pi_to_VI_Input(train_label, Z,a0, b0):
    N=Z.shape[0]
    K = Z.shape[1]
    sumZ=train_label.dot(Z)
    elpha=train_label.dot(Z) + (a0 / K)
    beta=((b0 * (K - 1) / K) + N - sumZ)
    Ln=(scipy.special.digamma(elpha.T) - scipy.special.digamma(elpha.T+beta.T)).dot(train_label)
    Ln_1 = (scipy.special.digamma(beta.T) - scipy.special.digamma(elpha.T+beta.T)).dot(train_label)
    return elpha,beta,Ln,Ln_1
def sample_g_s(train_label, S, c0, d0, Z, g_s, Xdclist):
    a1 = c0 + 0.5 * (np.sum(train_label,1))[:,np.newaxis] * Z.shape[1]
    a2 = d0 + 0.5 * (np.sum((train_label.dot(S*S)), 1))[:, np.newaxis]
    d=np.random.gamma(a1, 1. / a2)
    g_s =train_label.T.dot(d)        
    return g_s
def sample_g_s_to_VI_Input(train_label, S,c0, d0, Z):
    L_s=np.zeros(Z.T.shape, dtype=np.float64)
    a1 = c0 + 0.5 * (np.sum(train_label,1))[:,np.newaxis] * Z.shape[1]
    a2 = d0 + 0.5 * (np.sum((train_label.dot(S*S)), 1))[:, np.newaxis]
    L_s[:,:]= (np.sum((a1/a2)*train_label,0)[np.newaxis,:]) 
    L_s_a=a1-1
    L_s_b= -a2
    return L_s_a, L_s_b,L_s
def sample_g_eps(X_k, e0, f0):
    e = e0 + 0.5 * X_k.shape[0] * X_k.shape[1]
    f = f0 + 0.5 * np.sum(X_k ** 2)
    g_eps = np.random.gamma(e, 1. / f)
    return g_eps
def sample_g_eps_to_VI_Input(X_k, e0, f0):
    e = e0 + 0.5 * X_k.shape[0] * X_k.shape[1]
    f = f0 + 0.5 * np.sum(X_k ** 2)
    L_y_a=e-1
    L_y_b=-f
    return L_y_a,L_y_b

def params_intitializer_gibb_sampling(training_samples, train_label, Dinit, Winit, Alpha_init, Beta_init, pars,gibi,sparsity,g_eps_d,g_eps_w,g_sdd,batchsize,pruneDict, Xdclist,classItemsCount):
    D = Dinit
    W = Winit
    K = pars['K']
    a0 = pars['a0']
    b0 = pars['b0']
    c0 = pars['c0']
    d0 = pars['d0']
    e0 = pars['e0']
    f0 = pars['f0']
    L_y=g_eps_d
    L_h=g_eps_w
    L_y_s=L_y
    L_h_s=L_h
    Xd = np.copy(training_samples)
    Hw = np.copy(train_label)
    classes, N = Hw.shape
    g_sd = g_sdd*np.ones((Xd.shape[1],1), dtype=np.float64)
    classes, N = Hw.shape
    Pid = 0.5 * np.ones((Xd.shape[1], D.shape[1]), dtype=np.float64)
    Pi = 0.5 * np.ones((classes, K), dtype=np.float64)
    Pi_C = np.copy(Pi)
    Sd = Alpha_init
    Z = np.copy(Sd)
    Z[Z != 0] = 1
    Z = Z.T
    Sd = Sd.T
    Xd = training_samples - D.dot(Z.T*Sd.T)
    Hw = train_label - W.dot(Z.T*Sd.T)
    print('\nBayesian Inference using Gibbs sampling for SVI parameters initialization.......\n')
    DLiterations=gibi
    for iter in range(DLiterations):  #
        if iter==DLiterations-1:
            D_avg,Dv,W_avg,Wv = sample_D_to_VI_Input(Xd, D, Sd, g_eps_d, Z, Hw, W,g_eps_w,batchsize)
            Sd_avg,Sdv,Z_avg = sample_ZS_to_VI_Input(Xd, D, Sd, Z, Pid, g_sd, g_eps_d, Hw, W,
                                                                g_eps_w,batchsize)
            elpha1,beta1,Ln, Ln_1 = sample_Pi_to_VI_Input(train_label,Z,a0, b0)
            Pi_C=Pi
            L_y_a, L_y_b = sample_g_eps_to_VI_Input(Xd, e0, f0)
            L_h_a, L_h_b = sample_g_eps_to_VI_Input(Hw, e0, f0)
            L_s_a, L_s_b,L_s= sample_g_s_to_VI_Input(train_label, Sd,c0, d0, Z)
            pars.update(
                {'elpha': elpha1.T, 'beta': beta1.T,'L_y_a':L_y_a, 'L_y_b':L_y_b,'L_h_a':L_h_a,'L_h_b':L_h_b,'L_s_a':L_s_a, 'L_s_b':L_s_b,'L_s':L_s,
                 'D': D_avg, 'Dv': Dv, 'W': W_avg, 'Wv': Wv,'Ln': Ln,
                 'Ln_1': Ln_1,'Sd': Sd_avg.T, 'Sdv': Sdv.T,
                 'Z': Z_avg.T})
            break
        else:
            Xd, D, Hw, W = sample_D(Xd, D, Sd, g_eps_d, Z, Hw, W, Sw, g_eps_w,batchsize)
            Xd, Hw, Sd, Sw, Z = sample_ZS(Xd, D, Sd, Z, Pid, g_sd, g_eps_d, Hw, W, Sw,
                                                                 g_sw, g_eps_w,batchsize)
            
            Pid, Pi_C=sample_Pi(train_label, Z,Pid,Pi_C,a0,b0,Xdclist)
            Pi=Pi_C
            g_eps_d = sample_g_eps(Xd, e0, f0)
            g_eps_w = sample_g_eps(Hw, e0, f0)

            g_sd = sample_g_s(train_label, Sd, c0, d0, Z, g_sd,Xdclist)
        Pidex = [];
        if pruneDict==1 and iter >0:
            Pidex = [x0 for x0, val in enumerate(np.sum(Pi_C, axis=0)) if val > 1.0e-6]
            Pi_C1 = Pi_C
            k0 = 1e-6
            D=D[:,Pidex]
            W=W[:,Pidex]
            K = len(D[0, :]);
            Z = Z[:, Pidex]
            Pi_C=Pi_C[:,Pidex]
            Pid=Pid[:,Pidex]
            Pi=Pi[:,Pidex]
            Sd = Sd[:, Pidex]
            Sw = Sw[:, Pidex]
        print('Gibb sampling iter #:' + str(iter) + '   Dict. size:' + str(len(D[0, :])))
    return pars
################################End of Gibb sampling functions#############################################################################


################################Variational Inference functions#############################################################################
def SVI_D_W_Learning(N,Xd,D,Dv,Hd,W,Wv,Sd,Sdv,Z,batchsizee,L_y_a, L_y_b, L_h_a, L_h_b): # last ok editing
    K = D.shape[1]
    L_y=-(L_y_a+1)/L_y_b
    L_h=-(L_h_a+1)/L_h_b
    batchsize=0
    indexAtoms=list()
    if batchsizee<=0 or batchsizee>K:
        batchsize=K
    else:
        batchsize=batchsizee
    for i in range(0,K, batchsize):
        if (i+batchsize)<=K:
            indexAtoms=list(range(i,i+batchsize))
        else:
            indexAtoms=list(range(i,K))
        Xdd = Xd + (-0.5*D[:,indexAtoms]/Dv[:,indexAtoms]).dot(Z[indexAtoms,:]*(-0.5*Sd[indexAtoms,:]/Sdv[indexAtoms,:]))
        Hdd = Hd + (-0.5*W[:,indexAtoms]/Wv[:,indexAtoms]).dot(Z[indexAtoms,:]*(-0.5*Sd[indexAtoms,:]/Sdv[indexAtoms,:]))
        F = Z[indexAtoms,:] * (-0.5*Sd[indexAtoms, :]/Sdv[indexAtoms, :])
        A1 = (Z[indexAtoms,:]) * (((-0.5*Sd[indexAtoms, :]/Sdv[indexAtoms, :]) ** 2) + (-0.5/Sdv[indexAtoms, :]))
        Sds=np.sum(A1.T,0)[np.newaxis,:]
        A1 = N*np.sum(A1,1)/Xd.shape[1]
        if batchsize==1:
            B=L_y*((F.dot(Xdd.T)).T)
        else:
            B=L_y*((F.dot(Xd.T)).T + ((-0.5*D[:,indexAtoms]/Dv[:,indexAtoms])*Sds))
        A = -0.5* (L_y * A1 + D.shape[0])
        Dv[:, indexAtoms] =A
        D[:,indexAtoms]=N*B/Xd.shape[1]
        if batchsize==1:
            B=L_h*((F.dot(Hdd.T)).T)
        else:
            B=L_h*((F.dot(Hd.T)).T + ((-0.5*W[:,indexAtoms]/Wv[:,indexAtoms])*Sds))
        A = -0.5* (L_h * A1 + W.shape[0])
        Wv[:, indexAtoms] = A #-1.0/(2*A)
        W[:,indexAtoms]= N*B/Xd.shape[1] #-B/(2*A)
        Xd = Xdd - (-0.5*D[:,indexAtoms]/Dv[:,indexAtoms]).dot(Z[indexAtoms,:]*(-0.5*Sd[indexAtoms,:]/Sdv[indexAtoms,:]))
        Hd = Hdd - (-0.5*W[:,indexAtoms]/Wv[:,indexAtoms]).dot(Z[indexAtoms,:]*(-0.5*Sd[indexAtoms,:]/Sdv[indexAtoms,:]))
    return Xd,Hd,D,W,Dv,Wv
def SVI_Z_S_T_Learning(train_label,Xd,D,Dv,Hd,W,Wv,Sd,Sdv,Z,Zv,Ln,Ln_1,batchsizee,L_y_a, L_y_b, L_h_a, L_h_b,L_s_a, L_s_b): #Final editing
    K = D.shape[1];
    L_y=-(L_y_a+1)/L_y_b
    L_h=-(L_h_a+1)/L_h_b
    batchsize=0
    indexAtoms=list()
    if batchsizee<=0 or batchsizee>K:
        batchsize=K
    else:
        batchsize=batchsizee
    for i in range(0,K, batchsize):
        if (i+batchsize)<=K:
            indexAtoms=list(range(i,i+batchsize))
        else:
            indexAtoms=list(range(i,K)) #+list(range(K-i))
        Xdd = Xd + (-0.5*D[:,indexAtoms]/Dv[:,indexAtoms]).dot(Z[indexAtoms,:]*(-0.5*Sd[indexAtoms,:]/Sdv[indexAtoms, :]))
        Hdd = Hd + (-0.5*W[:,indexAtoms]/Wv[:,indexAtoms]).dot(Z[indexAtoms,:]*(-0.5*Sd[indexAtoms,:]/Sdv[indexAtoms, :]))
        Dv_sum = np.sum((-0.5*D[:,indexAtoms]/Dv[:,indexAtoms]) ** 2 + (-0.5/Dv[:,indexAtoms]),0)[np.newaxis,:]
        Wv_sum = np.sum((-0.5*W[:,indexAtoms]/Wv[:,indexAtoms]) ** 2 + (-0.5/Wv[:,indexAtoms]),0)[np.newaxis,:]
        Ass=L_y*Z[indexAtoms,:]*Dv_sum.T+L_h*Z[indexAtoms,:]*Wv_sum.T;
        DTD = np.sum((-0.5*D[:,indexAtoms]/Dv[:,indexAtoms]) ** 2,0)[np.newaxis,:]
        WTW = np.sum((-0.5*W[:,indexAtoms]/Wv[:,indexAtoms]) ** 2,0)[np.newaxis,:]
        if batchsize == 1:
            XdTD=((Xdd.T).dot((-0.5*D[:,indexAtoms]/Dv[:,indexAtoms]))).T
            HdTW=((Hdd.T).dot((-0.5*W[:,indexAtoms]/Wv[:,indexAtoms]))).T
        else:
            XdTD=(((Xd.T).dot((-0.5*D[:,indexAtoms]/Dv[:,indexAtoms])))+((Z[indexAtoms,:].T*(-0.5*Sd[indexAtoms,:]/Sdv[indexAtoms, :]).T)*DTD)).T
            HdTW=(((Hd.T).dot((-0.5*W[:,indexAtoms]/Wv[:,indexAtoms])))+((Z[indexAtoms,:].T*(-0.5*Sd[indexAtoms,:]/Sdv[indexAtoms, :]).T)*WTW)).T
        As=L_y*Z[indexAtoms,:]*XdTD+L_h * Z[indexAtoms, :] * HdTW;
        Ls=-((np.sum((L_s_a)*train_label,0)[np.newaxis,:])+1)/(np.sum((L_s_b)*train_label,0)[np.newaxis,:])
        Bs = -0.5*(Ass + Ls)
        Sd[indexAtoms,:] = As
        Sdv[indexAtoms,:] = Bs
        A=-0.5*L_y*((Dv_sum.T*((-0.5*Sd[indexAtoms,:]/Sdv[indexAtoms, :])**2+(-0.5/Sdv[indexAtoms, :])))-(2*(-0.5*Sd[indexAtoms,:]/Sdv[indexAtoms, :])*XdTD))
        B=-0.5*L_h*((Wv_sum.T*((-0.5*Sd[indexAtoms,:]/Sdv[indexAtoms, :])**2+(-0.5/Sdv[indexAtoms, :])))-(2*(-0.5*Sd[indexAtoms,:]/Sdv[indexAtoms, :])*HdTW))
        Z[indexAtoms,:] = Ln[indexAtoms,:] + A + B - Ln_1[indexAtoms,:]
        Zv[indexAtoms,:]=np.copy(Z[indexAtoms,:])
        Z[indexAtoms,:]=1.0/(1+np.exp(-Z[indexAtoms,:]))
        Xd = Xdd - (-0.5*D[:,indexAtoms]/Dv[:,indexAtoms]).dot(Z[indexAtoms,:]*(-0.5*Sd[indexAtoms,:]/Sdv[indexAtoms, :]))
        Hd = Hdd - (-0.5*W[:,indexAtoms]/Wv[:,indexAtoms]).dot(Z[indexAtoms,:]*(-0.5*Sd[indexAtoms,:]/Sdv[indexAtoms, :]))
    return Xd,Hd,Z,Zv,Sd,Sdv

def SVI_Pi_Learning(N,labels,Z,a0,b0,elpha,beta,classesItemsCount):
    K = Z.shape[0];
    elpha=(classesItemsCount)*(Z.dot(labels.T))+ a0 / K
    beta=(classesItemsCount-classesItemsCount*(Z.dot(labels.T)))+ b0 * (K - 1) / K
    Pics=np.random.beta(elpha,beta)    
    return Pics,elpha,beta
def SVI_Ly_Lh_Learning(N,Xd,D,Dv,Sd,Sdv,Hd,W,Wv,Z,e0,f0):
    a = np.sum(Xd ** 2, 0)[newaxis, :]
    b=np.sum(((-0.5*D/Dv)**2+(-0.5/Dv)),0)[newaxis,:].dot(Z*((-0.5*Sd/Sdv)**2+(-0.5/Sdv)))
    c=np.sum((-0.5*D/Dv)**2,0)[newaxis,:].dot(Z*(-0.5*Sd/Sdv)**2)
    a2=0.5*N*np.sum((a+b-c))/Xd.shape[1]+f0
    a1=(0.5*N*1*Xd.shape[0]+e0-1)
    L_y_a=a1
    L_y_b=-a2;
    a = np.sum(Hd ** 2, 0)[newaxis, :];
    b = np.sum(((-0.5*W/Wv)** 2 + (-0.5/Wv)), 0)[newaxis,:].dot(Z * ((-0.5*Sd/Sdv)** 2 + (-0.5/Sdv)));
    c = np.sum((-0.5*W/Wv) ** 2, 0)[newaxis,:].dot(Z *(-0.5*Sd/Sdv) ** 2);
    a2 = 0.5*N*np.sum((a + b - c))/Hd.shape[1] + f0;
    a1 = (0.5*N*1 * Hd.shape[0] + e0-1);
    L_h_a=a1
    L_h_b = -a2
    return L_y_a,L_y_b,L_h_a,L_h_b
def SVI_Ls_Learning(N,labels,Sd,Sdv,c0,d0,L_s_a,L_s_b,classItemsCount):
    a1=(c0+0.5*((classItemsCount).T)*Sdv.shape[0])
    a2=(0.5*((classItemsCount).T)*(np.sum(labels.dot(((-0.5*Sd/Sdv)**2+(-0.5/Sdv)).T),1)[:,np.newaxis])+d0)
    L_s_a=a1-1
    L_s_b=-a2
    return L_s_a,L_s_b
def SVI(loadedDB, batchsize,patchLength, DBdisc,dicSize,iiii,training_samples, train_label, test, test_label,p0,kk,k0,mi,gibi,sparsity,a0,b0,c0,d0,e0,f0,g_eps_d,g_eps_w,g_sd,dataIndex):
    starttime = time.time()
    test = sk.normalize(test, norm='l2', axis=0)
    training_samples = sk.normalize(training_samples, norm='l2', axis=0) 
    if dicSize!=0:
        initDictSize=dicSize
    else:
        initDictSize = np.math.floor(1.25 * training_samples.shape[1])
    training_samples = sk.normalize(training_samples, norm='l2', axis=0)
    Dinit, label_init = initDSeqSelection(training_samples, train_label, initDictSize)
    Alpha_init = ompAlgo(Dinit, training_samples, sparsity)
    Winit = initClassifier(Alpha_init, train_label)
    Beta_init = np.copy(Alpha_init)
    classItemsCount=np.sum(train_label,1)[np.newaxis,:]
    ss=np.min(classItemsCount)
    if a0!=0:
        a0=a0
    else:
        a0=ss/4.
    if b0!=0:
        b0=b0
    else:
        b0=ss/4.
    Xdclist=list()
    for c in range(train_label.shape[0]):
        Xdc=[x0 for x0,val in enumerate(train_label[c,:]) if val!=0]
        Xdclist=Xdclist+[Xdc]
    K = initDictSize
    pars = {'a0': a0, 'b0': b0, 'c0': c0, 'd0': d0, 'e0': e0, 'f0': f0, 'K': K}
    abc = params_intitializer_gibb_sampling(training_samples, train_label, Dinit, Winit, Alpha_init, Beta_init,pars,gibi,sparsity,g_eps_d,g_eps_w,g_sd,1,iiii,Xdclist,classItemsCount)
    K = abc['K']
    D = abc['D']
    W = abc['W']
    Dv = abc['Dv']
    Wv = abc['Wv']
    elpha=abc['elpha']
    beta=abc['beta']
    Pics=np.random.beta(elpha,beta)
    Sd = abc['Sd']
    Sdv = abc['Sdv']
    Z = abc['Z']
    Zv=np.copy(Z)
    Xd = training_samples - D.dot(Z * Sd)
    Hd = train_label - W.dot(Z * Sd)
    D=D/Dv
    Dv=-0.5/Dv
    W=W/Wv
    Wv=-0.5/Wv
    Sd=Sd/Sdv
    Sdv=-0.5/Sdv
    print("\n\n########################################################################################\n")
    print("----Stochastic Variational Inference.......\n")
    print('Before training:' + '   Dict. size:' + str(D.shape[1]))
    L_s_a = abc['L_s_a']
    L_s_b = abc['L_s_b']
    L_y_a=abc['L_y_a']
    L_y_b=abc['L_y_b']
    L_h_a=abc['L_h_a']
    L_h_b=abc['L_h_b']
    previous=0.0
    t=-1
    xpnts=[]
    ypnts=[]
    starttime = time.time()
    while t<mi:
        t=t+1
        pt=(p0+t+1)**kk
        Sdp=np.copy(Sd)
        Sdvp=np.copy(Sdv)
        Dp =np.copy(D)
        Dvp = np.copy(Dv)
        Wp = np.copy(W)
        Wvp = np.copy(Wv)
        elphap = np.copy(elpha)
        betap = np.copy(beta)
        L_h_ap = L_h_a
        L_h_bp = L_h_b
        L_y_ap = L_y_a
        L_y_bp = L_y_b
        L_s_ap = np.copy(L_s_a)
        L_s_bp = np.copy(L_s_b)
        if t !=0:
            Zp=np.copy(Zv)
        n=list()
        labelsSparsen=0
        for c in range(len(Xdclist)):
            Xdc=Xdclist[c]
            ls=rd.sample(Xdc,1)
            if c==0:
                labelsSparsen=train_label[:,ls]
            else:
                labelsSparsen=np.hstack((labelsSparsen,train_label[:,ls]))
            n=n+ls
        Ln=(scipy.special.digamma(elpha) - scipy.special.digamma(elpha+beta)).dot(train_label)
        Ln_1 = (scipy.special.digamma(beta) - scipy.special.digamma(elpha+beta)).dot(train_label) 
        Xd[:,n], Hd[:,n], Z[:,n],Zv[:,n],Sd[:,n], Sdv[:,n] = SVI_Z_S_T_Learning(train_label[:,n],Xd[:,n], D, Dv, Hd[:,n], W, Wv, Sd[:,n], Sdv[:,n], Z[:,n],Zv[:,n],Ln[:,n], Ln_1[:,n],batchsize,L_y_a, L_y_b, L_h_a, L_h_b,L_s_a, L_s_b)
        Xd[:,n],Hd[:,n],D, W, Dv,Wv = SVI_D_W_Learning(Xd.shape[1],Xd[:,n], D,Dv,Hd[:,n], W, Wv,Sd[:,n], Sdv[:,n], Z[:,n],batchsize,L_y_a, L_y_b, L_h_a, L_h_b)
        L_y_a,L_y_b,L_h_a,L_h_b = SVI_Ly_Lh_Learning(Xd.shape[1],Xd[:,n], D, Dv, Sd[:,n], Sdv[:,n],Hd[:,n], W, Wv, Z[:,n], e0, f0)
        Pics,elpha,beta = SVI_Pi_Learning(Xd.shape[1],train_label[:,n],Z[:,n], a0, b0,elpha,beta,classItemsCount)
        L_s_a,L_s_b= SVI_Ls_Learning(Xd.shape[1],train_label[:,n],Sd[:,n], Sdv[:,n], c0, d0,L_s_a,L_s_b,classItemsCount)
        
        D = (1 - pt) * Dp + pt * D
        Dv = (1 - pt) * Dvp + pt * Dv
        W = (1 - pt) * Wp + pt * W
        Wv = (1 - pt) * Wvp + pt * Wv
        elpha = (1 - pt) * elphap + pt * elpha
        beta = (1 - pt) * betap + pt * beta
        L_y_a = (1 - pt) * L_y_ap + pt * L_y_a
        L_y_b = (1 - pt) * L_y_bp + pt * L_y_b
        L_h_a = (1 - pt) * L_h_ap + pt * L_h_a
        L_h_b = (1 - pt) * L_h_bp + pt * L_h_b
        L_s_a = (1 - pt) * L_s_ap + pt * L_s_a
        L_s_b = (1 - pt) * L_s_bp + pt * L_s_b
        #ELBO Computing after each 20 iterations
        if t%20==0:
            obj=0.0
            obj=np.sum((D-Dp)*(-Dp/(2*Dvp))+(Dv-Dvp)*(Dp**2/(4*Dvp**2)-1.0/(2*Dvp))-(Dp**2/(4*Dvp)+0.5*np.log(-2*Dvp)))
            obj=obj+np.sum((W-Wp)*(-Wp/(2*Wvp))+(Wv-Wvp)*(Wp**2/(4*Wvp**2)-1.0/(2*Wvp))-(Wp**2/(4*Wvp)+0.5*np.log(-2*Wvp)))
            #obj=obj+np.sum(((elpha-elphap)*(scipy.special.digamma(elphap+1)-scipy.special.digamma(elphap+betap+2))+(scipy.special.loggamma(elphap+1))+(scipy.special.loggamma(betap+1))-(scipy.special.loggamma(elphap+betap+2))))
            #obj=obj+np.sum(((beta-betap)*(scipy.special.digamma(betap+1)-scipy.special.digamma(elphap+betap+2))))
            obj=obj+np.sum(((elpha-elphap)*(scipy.special.digamma(elphap)-scipy.special.digamma(elphap+betap))+(scipy.special.loggamma(elphap))+(scipy.special.loggamma(betap))-(scipy.special.loggamma(elphap+betap))))
            obj=obj+np.sum(((beta-betap)*(scipy.special.digamma(betap)-scipy.special.digamma(elphap+betap))))
            obj=obj+np.sum((Sd-Sdp)*(-Sdp/(2*Sdvp))+(Sdv-Sdvp)*(Sdp**2/(4*Sdvp**2)-1.0/(2*Sdvp))-(Sdp**2/(4*Sdvp)+0.5*np.log(-2*Sdvp)))
            obj=obj+(L_y_a-L_y_ap)*(scipy.special.digamma(L_y_ap+1)-np.log(-L_y_bp))+(L_y_b-L_y_bp)*(-(L_y_ap+1)/L_y_bp) + ((scipy.special.loggamma(L_y_ap+1))-(L_y_ap+1)*np.log(-L_y_bp))
            obj=obj+(L_h_a-L_h_ap)*(scipy.special.digamma(L_h_ap+1)-np.log(-L_h_bp))+(L_h_b-L_h_bp)*(-(L_h_ap+1)/L_h_bp) + ((scipy.special.loggamma(L_h_ap+1))-(L_h_ap+1)*np.log(-L_h_bp))
            obj=obj+np.sum((L_s_a-L_s_ap)*(scipy.special.digamma(L_s_ap+1)-np.log(-L_s_bp))+(L_s_b-L_s_bp)*(-(L_s_ap+1)/L_s_bp) + ((scipy.special.loggamma(L_s_ap+1))-(L_s_ap+1)*np.log(-L_s_bp)))
            if t!=0:
                obj=obj+np.sum((Zv-Zp)*(1.0/(1+np.exp(-Zp)))+Zp)
                ypnts=ypnts+[obj]
                xpnts=xpnts+[t]
            print(f"\n****Outcomes of Stochastic Variational Inference for {DBdisc}****")
            print(f"Iteration No = {(t+1):3d} off {mi:3d}")
            print(f"[Initial dictionary size: {initDictSize:4d} atoms]")
            print(f"[Current Dictionnary size: {D.shape[1]:4d} atoms]")
            print(f"Value of ELBO  = {(obj):8.2e}")
            print(f"Rate of change of ELBO  = {(obj-previous):8.2e}")
        previous=obj
        
        ######################################################################################3
        #Code for pruning of dictionary atoms
        ps=np.sum(Pics,axis=1)
        if t<mi and iiii != 0 and np.min(ps)<=k0 and np.max(ps)>k0:  # and t!=(Ntms*(lt)-1) :  # t%lt==0:  
            Pidex = [x0 for x0, val in enumerate(ps) if val > k0]
            Pidexr=list(set(range(D.shape[1])) - set(Pidex))
            Xd = Xd + (-0.5*D[:,Pidexr]/Dv[:,Pidexr]).dot(Z[Pidexr,:]*(-0.5*Sd[Pidexr,:]/Sdv[Pidexr, :]))
            Hd = Hd + (-0.5*W[:,Pidexr]/Wv[:,Pidexr]).dot(Z[Pidexr,:]*(-0.5*Sd[Pidexr,:]/Sdv[Pidexr, :]))
            D = D[:,Pidex]
            W = W[:,Pidex]
            Dv = Dv[:,Pidex]
            Wv = Wv[:,Pidex]
            Z = Z[Pidex, :]
            Zv = Zv[Pidex, :]
            elpha=elpha[Pidex,:]
            beta=beta[Pidex,:]
            Sd = Sd[Pidex, :]
            Sdv = Sdv[Pidex, :]
    # show the trend of ELBO with each iteration
    # plt.plot(xpnts, ypnts)
    # plt.show()
    D = ((-0.5*D/Dv) + np.random.randn(D.shape[0], D.shape[1]) * np.sqrt(-0.5/Dv))
    W = ((-0.5*W/Wv) + np.random.randn(W.shape[0], W.shape[1]) * np.sqrt(-0.5/Wv))
    finalDictSize=D.shape[1]
    endtime = time.time()
    trainTime = endtime - starttime
    print('\nClassification...\n')
    starttime = time.time()
    data3D=loadedDB['data3D']
    data_gt2D=loadedDB['data_gt2D']
    OrgIndex=loadedDB['OrgIndex']
    ColumnLength=loadedDB['ColumnLength']
    kCoff, classWiseAccuracy,accuracy  = classification(D, W, test, test_label, sparsity,dataIndex, OrgIndex,data3D,ColumnLength,data_gt2D,patchLength)
    endtime = time.time()
    testTime = endtime - starttime
    algosettings = "Date and time = " + str(datetime.now()) + ", p0 = " + str(p0) + ", k = " + str(kk) + ", k0 = " + str(k0) + ", a0 = " + str(a0) + ", b0 = " + str(b0) + ", c0 = " + str(c0) + ", d0 = " + str(d0) + ", e0 = " + str(e0) + ", f0 = " + str(f0) + ", sparsity = " + str(sparsity) + ", Iterations  = " + str(t+1)
    return kCoff, classWiseAccuracy,algosettings,(t+1),accuracy,trainTime, trainTime/train_label.shape[1], testTime / test_label.shape[1], initDictSize, finalDictSize

#*********************INSTRUCTIONS**************************
# 1--This code has been written for Stochastic Variational inference.
# 3--Output: after each 20th iteration, it displays main training parameters like ELBO, current size of dictionary, iteration number etc.
#    However, overall results will be written to output files for respective datasets.
# 4--Please note that all packages listed at the top of this file must be installed before running this algorithm
# 5---Our algorithm performs sampling of parameters while iterating through atoms one by one in each of the main iterations.
#     We have designed and vetorized the code in such a way that we can treat atoms in groups that reduces the number of iterations at atoms level. We have named this variable as "batchsize" in the code.
#     We treat all atoms as one group and vectorization updates atoms in one iteration at atoms update stage.
#Code_DPath="./HyperSpectral_Classification/IEEE_ACCESS/"
Code_DPath="./"
f=0
while f<3:
    f=f+1
    if f==1:
        DBdisc="SVI training on Pines database"
        resultFile="SVI_Pines_Result.txt"
        dataBaseName="Pines"
        p0 = 1 
        k = -1.0
        k0=1e-100 #(Threshold for pruning of dictionary atoms. It is effective if pruning is on)
        mi = 500 #(Number of iterations)
        gibi=1     #(Number of iterations of Gibbs Sampling for initilization of this algorithm--One iteration means without initilizing using Gibbs Sampling)--We can initilize with or without Gibbs sampling.
        sparsity=25
        #Hyper parameters as explained in the paper
        a0=0
        b0=0
        c0=1.e-6
        d0=1.e-6
        e0=1.e-6
        f0=1.e-6
        g_eps_d = 1.e+12
        g_eps_w = 1.e+12
        g_sd = 1
        g_sw = 1
        percentTrainSamples=0.1 #(Percentage of training samples selelction from database pixels)
        patchLength=6# 6 #(patchLength*patchLength=patch size around a pixel during classification)
        batchsize=0 # Number of dictionary atoms to be trained as one group (0 for full dictionary atoms)--We set it at 0. Note that Variation inference algorithm gets stuck at local minimum and does not get escaped but Stochastic variational inference works fine with training of multiple dictionary atoms at one time.
        pruneDict=0 #(1 if pruning allowed, 0 if not)--We can perform pruning of dictionary atoms after each iteration to reduce the size of dictionary. We have used 0 value in our experiments
        dicSize=0 # Dictionary Size (0 for auto selection)--We can perform by fixing dictionary size in number of atoms
        q=SVI_main(pruneDict,dicSize,Code_DPath+'Data/Processed_Data/Indian_pines3.npz',Code_DPath,resultFile,DBdisc,dataBaseName,p0,k,k0,mi,gibi,sparsity,a0,b0,c0,d0,e0,f0,g_eps_d,g_eps_w,g_sd,batchsize,percentTrainSamples,patchLength,foperation="a+")
    if f==2:
        DBdisc="SVI training on Salinas database"
        resultFile="SVI_Salinas_Result.txt"
        dataBaseName="Salinas"
        p0 = 1 
        k = -1.0
        k0=1e-30 
        mi = 500
        gibi=1
        sparsity=25
        #Hyper parameters as explained in the paper
        a0=0
        b0=0
        c0=1.e-6
        d0=1.e-6
        e0=1.e-6
        f0=1.e-6
        g_eps_d = 1.e+12 
        g_eps_w = 1.e+12 
        g_sd = 1 
        g_sw = 1
        percentTrainSamples=0.01
        patchLength=5
        batchsize=0 
        pruneDict=0
        dicSize=0
        q=SVI_main(pruneDict,dicSize,Code_DPath+'Data/Processed_Data/Salinas3.npz',Code_DPath,resultFile,DBdisc,dataBaseName,p0,k,k0,mi,gibi,sparsity,a0,b0,c0,d0,e0,f0,g_eps_d,g_eps_w,g_sd,batchsize,percentTrainSamples,patchLength,foperation="a+")
    if f==3:
        DBdisc="SVI training on PaviaU Database"
        resultFile="SVI_PaviaU_Result.txt"
        dataBaseName="Pavia"
        p0 = 1
        k = -1.0
        k0=1.e-30
        mi = 500
        gibi=1
        sparsity=40
        a0=0
        b0=0
        c0=1.e-6
        d0=1.e-6
        e0=1.e-6
        f0=1.e-6
        g_eps_d = 1.e+9
        g_eps_w = 1.e+9
        g_sd = 1
        g_sw = 1
        percentTrainSamples=0.1
        patchLength=5
        batchsize=0
        pruneDict=0
        dicSize=0
        q=SVI_main(pruneDict,dicSize,Code_DPath+'Data/Processed_Data/PaviaU3.npz',Code_DPath,resultFile,DBdisc,dataBaseName,p0,k,k0,mi,gibi,sparsity,a0,b0,c0,d0,e0,f0,g_eps_d,g_eps_w,g_sd,batchsize,percentTrainSamples,patchLength,foperation="a+")
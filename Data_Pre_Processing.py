import numpy as np
import gzip
import os
import random
import scipy
import scipy.io as sio
def readData(dataFileName,dictFileName, gt_fileName, dict_gt_fileName, savedFileName, numClasses,Code_DPath="./Hyperspectral_Classification/IEEE_ACCESS/"):
    #Code_DPath="/HyperSpectral_Classification/IEEE_ACCESS/"
    Code_DPath="/"
    cwd = os.getcwd()
    path = cwd + Code_DPath + "Data/Raw_Data/" + dataFileName + ".mat"
    mat=scipy.io.loadmat(path)
    features1=mat[dictFileName]
    ColumnLength=features1.shape[1]
    data3D=features1.transpose(2,0,1)
    features1=features1.transpose(2,0,1).reshape(features1.shape[2],-1)
    path = cwd + Code_DPath + "Data/Raw_Data/" + gt_fileName + ".mat"
    mat=scipy.io.loadmat(path)
    pavia_gt=mat[dict_gt_fileName]
    data_gt2D=np.copy(pavia_gt)
    pavia_gt=pavia_gt.reshape(-1)
    lc=np.zeros((numClasses,1),dtype=np.int8)
    lllc=np.zeros((numClasses,1),dtype=np.int32)
    ff=0
    indorg=-1
    PaviaUIndexlist=list()
    for i in pavia_gt:
        indorg=indorg+1
        llc=np.copy(lc)
        if i>0:
            PaviaUIndexlist=PaviaUIndexlist+[indorg]
            llc[i-1,0]=1
            lllc[i-1,0]=lllc[i-1,0]+1
            if ff==0:
                labels=llc
                ff=1
            else:
                labels=np.hstack([labels,llc])
    features=np.array((features1.shape[0],labels.shape[1]),dtype=np.float32)
    features=features1[:,PaviaUIndexlist]
    path = cwd + Code_DPath + "Data/Processed_Data/" + savedFileName
    np.savez(path,features=features,labels=labels,data3D=data3D,data_gt2D=data_gt2D, OrgIndex=PaviaUIndexlist,ColumnLength=ColumnLength)
    print(f" {dataFileName} processed successfully")
    return features, labels
readData("PaviaU","paviaU",  "PaviaU_gt","paviaU_gt",  "PaviaU3", 9)
readData("Indian_pines","indian_pines","Indian_pines_gt","indian_pines_gt",  "Indian_pines3", 16)
readData("Salinas_corrected","salinas_corrected",  "Salinas_gt","salinas_gt",  "Salinas3", 16)
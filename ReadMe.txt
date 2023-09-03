(Application of Stochastic Variational Inference for Dictionary Learning)
This code is the implementation of Stochastic Variational Inference in the domain of dictionary Learning
It consistes of forllowing folders and files.
Root (This is any folder on the computer to be considered as root folder in which the code will be copied)
 ---Data (This folder contains raw and processed data)
     ---Raw_Data (This folder contains .mat files of datasets downloaded from their locations)
     ---Processed_Data (This folder contains processed data to be used in our code)
 Data_Pre_Processing.py (This file contains the code for pre-processing the Raw Data. The processed data is saved in Processed_Data folder)
 SVI.py (This file is the code file for dictionary and classifier learning)
How to setup and run the code?
1--Download hyperspectral images along with ground truths available at (https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Pavia_Centre_and_University) and place them in the folder "Raw_Data". 
   Pavia University Image:-> Download the image (PaviaU.mat) from (https://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat) and download the ground truth (PaviaU_gt.mat) from (https://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat)
   Indian Pines:-> Download the image (Indian_pines.mat) from (https://www.ehu.eus/ccwintco/uploads/2/22/Indian_pines.mat) and download the ground truth (Indian_pines_gt.mat) from (https://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat)
   Salinas:-> Download the image (Salinas_corrected.mat) from (https://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat) and download the ground truth (Salinas_corrected.mat) from (https://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat)
2--Run Data_Pre_Processing.py code to generate the processed data with files names "PaviaU3.npz", "Salinas3.npz", and "Indian_pines3.npz" placed in "Processed_Data" folder
3--Run the main code file "SVI.py". The code will perform the experiments on three processed datasets and will place the results in 
   three respective files named "SVI_Pines_Result.txt", "SVI_Salinas_Result.txt", and "SVI_PaviaU_Result.txt" in root folder
Note:Our algorithm performs sampling of parameters while iterating through atoms one by one in each of the main iterations.
     We have designed and vetorized the code in such a way that we can update the atoms in groups that reduces the number of iterations at atoms level. We have named this variable as "batchsize" in the code that represents number of atoms in a group.
     Its value equal to 0 means all the atoms will be iterated once, updating them a single group.
     The results of our approach SVI do not disturb if we update all the atoms as a single group in each main iteration. Other approaches do not follow this.

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
from IPython.display import display, HTML

root_dir='Results_Random_Hyp'
file_name='val_loss.txt'
Exp_directories = next(os.walk(root_dir))[1]
print 'Exp Directories:', Exp_directories
Trial_directories = sorted(next(os.walk(root_dir + '/' + Exp_directories[1]))[1])
print 'Trial :', Trial_directories
best_losses=np.zeros(len(Exp_directories)*len(Trial_directories))
count=0
for exp_dir in Exp_directories:
    for trial in Trial_directories:
        currFile = open(root_dir + '/' + exp_dir + '/' + trial + '/'+ file_name , "r")
        lines = currFile.readlines()
        lines = [float(x.split()[0]) for x in lines]
        # print(lines)
        best_losses[count]=min(lines)
        count+=1
print 'best_losses:', best_losses
trial_nums=[2, 4, 8, 16]

# data is dcitionary of all the data for particular trial
data ={}
totalExp_size = len(best_losses)

for trial in trial_nums:
    numData = int(totalExp_size/trial) # number of datapoints for that particular trial
    data[trial] = np.zeros(numData)
    for i in range(numData):
        minLoss = min(best_losses[i*trial: (i+1)*trial])
        data[trial][i] = minLoss

# trial1_box = pd.DataFrame([data[2],data[4],data[8]] , columns = ['2','4','8'])
# trial_box1 = pd.DataFrame(data[2] , columns = ['2'])
# trial_box2 = pd.DataFrame(data[4] , columns = ['4'])
# trial_box3 = pd.DataFrame(data[8] , columns = ['8'])

data_list = [data[x] for x in sorted(data)]
print('data_list: ', data_list)

fig, ax = plt.subplots()
points_16=data_list[-1]
points_8=data_list[-2]
best_grid_loss= 0.14603358960390547 #best loss for grid loss 24 trials
ax.plot([0., 4.5], [best_grid_loss, best_grid_loss], "b--")
data_list[2]=[]
data_list[3]=[]
ax.boxplot(data_list)
ax.scatter(np.ones(len(points_8))*3, points_8, c='r',marker="+")
ax.scatter(np.ones(len(points_16))*4, points_16,c='r', marker="+")
ax.set_xticklabels(trial_nums,fontsize=12)
ax.set_title('Model performance hyper-parameter optimization')
ax.set_xlabel('Number of trials')
ax.set_ylabel('Validation Loss')
ax.legend(['Grid Search'])

plt.show()
# print('data',data)
# trial_box_all = pd.DataFrame(data_list , columns = ['2','4','8'])
# data = [data, d2, d2[::2, 0]]
# multiple box plots on one figure
# plt.figure()
# plt.boxplot(data)

# trial_box1.plot.box()
# trial_box2.plot.box()
# trial_box3.plot.box()







# for i, val in enumerate(expList):
#   #creating folders for each experiment
#   foldExpList.append(root_dir + '/' + 'Exp' + str(val))
#   if not os.path.exists(foldExpList[i]):
#       os.mkdir(foldExpList[i])
#
# currFile = open(labelPath + '/' + 'orientation_labels.txt', "r")

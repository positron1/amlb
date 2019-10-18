##################################################################################
########## Author: Yonglin Zhu
########## Email: zhuygln@gmail.com
##################################################################################
import json
import jsonpickle
from sas7bdat import SAS7BDAT
import os
import sys
import logging
import optparse
from utils import *
from postprocessing import *
from runbench import *
import secrets 
from DateTime import DateTime
import time
##################################################################################
##################################################################################
debugmode = True 
if debugmode:
  pass
else:
  orig_stdout = sys.stdout
<<<<<<< HEAD
current_time = DateTime(time.time(), 'US/Eastern')

numeric_features =[]
categorical_features =[]
dirt = '/root/data/'
task ='it'
if task=='bt':
   datalist = glob.glob(dirt+"opentest/*sas7bdat")
   metalist = glob.glob(dirt+"meta/*csv")
   datalist = remove_dirt(datalist,dirt+'/opentest/')
   metalist = remove_dirt(metalist,dirt+'/meta/')
elif task=='bre':
   datalist = glob.glob(dirt+"binaryRareEvent/data/*sas7bdat")
   metalist = glob.glob(dirt+"binaryRareEvent/meta/*csv")
   datalist = remove_dirt(datalist,dirt+'/binaryRareEvent/data/')
   metalist = remove_dirt(metalist,dirt+'/binaryRareEvent/meta/')
   dirt = dirt+'binaryRareEvent/'
elif task =='it':
   datalist = glob.glob(dirt+"intervalTarget/data/*sas7bdat")
   metalist = glob.glob(dirt+"intervalTarget/meta/*csv")
   datalist = remove_dirt(datalist,dirt+'/intervalTarget/data/')
   metalist = remove_dirt(metalist,dirt+'/intervalTarget/meta/')
   dirt = dirt+'intervalTarget/'

print(datalist)
print(metalist)
fitmetrics = autosklearn.metrics.mean_squared_error
datalist =sorted(datalist)
metalist = sorted(metalist)
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
runlist =['0','1','2','3','4','10','11','12','13','14']
runlist =['12','13','14']
rep= 1 
timelist = [900]
foldlist = [0]

prep = True 
runlist = ['3']
runlist = ['4']

runlist =['4','5','6','7']
runlist =['6','7']
runlist =['0','1','2','3','4','5','6','7']
timestamp = str(current_time.year()) + str(current_time.aMonth())+ str(current_time.day()) + \
        str(current_time.h_24()) + str(current_time.minute())  + str(time.time())[:2]
logfile = open('results/log_'+str(len(runlist))+'dataset'+str(timelist[0])+str(foldlist[0])+"rep"+str(rep)+str(timestamp)+".txt",'w')
=======
>>>>>>> 11b2b413a3c345785f4bcde1a2958dae4c2aae03

##################################################################################
###             Inputs                                  #
#################################################################################
task ='it' # interval target task
prep = False # Data preprocessing with meta data
dirt = '/root/data/' # dataset directory
task_token = secrets.token_hex(8) # generate unique token for this run
#################################################################################
runlist=['5','7'] # dataset id #
rep= 2 # repetition 
foldlist = [0] # 0: single validation, no cross validation
timelist =[100] # time limit for training in seconds
#################################################################################
############## Initial setup
#################################################################################
dirt,logfile,datalist,metalist,timestamp,fitmetrics = init(dirt,task,runlist,timelist,foldlist,rep,task_token)
if debugmode:
  pass
else:
  sys.stdout = logfile
#################################################################################
########## runing ...
#################################################################################
print(datalist,metalist)
for im,meta in enumerate(metalist):
    myid = meta.split('_')[0]
    if myid[2:] in runlist:
      print(myid[2:])
      framework = 'autosklearn'
      ncore = 4
      dataset = datalist[im]# "uci_bank_marketing_pd"
      print("\ndataset:\t",dataset)
      print("\nmetadata information:\t",meta)
      try:
        runbenchmark(prep,dataset,framework,foldlist,ncore,timelist,dirt,meta,fitmetrics,rep,logfile,task_token)
      except:
        traceback.print_exc(file=sys.stdout)
        print('Failed:\t',myid,dataset)
        continue
#################################################################################
########## Summary of results
#################################################################################
for lf,locfold in enumerate(timelist):
    ldirt='results/'+str(locfold)+'s/'
    dataname,auclist,loglosslist,acclist=get_results_reg(ldirt,timestamp,task_token)

#################################################################################
########## The End
#################################################################################
if debugmode:
  pass
else:   
  sys.stdout = orig_stdout
  logfile.close()

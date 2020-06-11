##################################################################################
# Author: Yonglin Zhu
########## Email: zhuygln@gmail.com
##################################################################################
from utils import *
from postprocessing import *
from runbench import *
from sklearn.metrics import roc_auc_score, accuracy_score
from util_lib import *

##################################################################################
logmode = False
if logmode:
    orig_stdout = sys.stdout

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")
import os


ssh = createSSHClient(server, port, user, password)
scp = SCPClient(ssh.get_transport())
##################################################################################
###             Inputs                                  #
#################################################################################

framework = "autosklearn"
metalearning = True  # default for autosklearn
framework = "tpot"
metalearning = False
task = "bt"  # interval target task
prep = False  # Data preprocessing with meta data
dirt = "/root/data/"  # dataset directory
outputdir = "./results/"
task_token = secrets.token_hex(8)  # generate unique token for this run
#################################################################################
runlist = ["14"]  # dataset id #
rep = 2  # repetition
corelist = [1]
foldlist = [0]  # 0: single validation, no cross validation
timelist = [3600]  # time limit for training in seconds
#################################################################################
# Initial setup
#################################################################################
dirt, taskname, logfile, csvdatalist, sasdatalist, metalist, timestamp, fitmetrics = init(
    dirt, task, runlist, timelist, foldlist, rep, task_token
)
metadataid = get_id(metalist)
csvdataid = get_id(csvdatalist)
sasdataid = get_id(sasdatalist)
print("csvdataid\n", csvdataid)
print("csvdataid\n", len(csvdataid))
print("sasdataid\n", sasdataid)
print("sasdataid\n", len(sasdataid))
print("metadataid\n", metadataid)
print("metadataid\n", len(metadataid))

if logmode:
    sys.stdout = logfile
#################################################################################
# runing ...
#################################################################################

runnamelist = sasdatalist

for dataname in runnamelist:
    dataset, meta = check_dataset(
        dataname, csvdatalist, sasdatalist, metalist
    )  #: check_id(ind,csvdataid,csvdatalist,sasdataid,sasdatalist,metadataid,metalist)
    try:
        print(task)
        runbenchmark(
            task,
            taskname,
            metalearning,
            prep,
            dataset,
            framework,
            foldlist,
            corelist,
            timelist,
            dirt,
            meta,
            fitmetrics,
            rep,
            logfile,
            outputdir,
            task_token,
        )
    except:
        print("Failed:\t", dataname)  # ,dataset)
        traceback.print_exc(file=sys.stdout)
        continue
#################################################################################
# Summary of results
#################################################################################




for lf, locfold in enumerate(timelist):
    ldirt = "results/" + str(locfold) + "s/"
    dataname, auclist, loglosslist, acclist = get_results_clf(
        ldirt, timestamp, task_token
    )
    benchmark_results = compile_results(
        ldirt, timestamp, task_token, taskname
    )
    scp.put(benchmark_results,'/nfs/rdcx200/ifs/EM_data/All_Users/ramyne/dataForBenchmarking/auto_benchmark_results/opensource_benchmark_'
            +task_token+'.csv')

if logmode:
    sys.stdout = orig_stdout
    logfile.close()

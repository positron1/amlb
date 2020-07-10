import pandas as pd
import glob
import numpy as np
import glob
import os
import json
from datetime import datetime


def check(Datasetname, shapetrain, shapetest, para, timespend, results):
    return True


def get_results_clf(dirt, date, task_token):
    # Create target Directory if don't exist
    if not os.path.exists(dirt):
        os.mkdir(dirt)
        print("Directory ", dirt,  " Newly Created ")
    plotfile = open(dirt + task_token + "_summary.csv", "w")
    plotfile.write(
        "Dataset_name,data_id,Folds,Run_id,date,Logloss,AUC,f1,ACC,Time_limit,Timespend,Cores,Fitmetric,Framework,train_size,test_size,models\n"
    )
    sep = "_"
    if True:
        dataname = []
        auclist = []
        loglosslist = []
        acclist = []
        datalist = glob.glob(dirt + "result*" + task_token + "*")
        print(datalist)
        for dataresult in datalist:
            getname = True
            gettime = True
            temp = []

            resultfile = dataresult.split("_")
            print(resultfile)
            outjson = json.load(open(dataresult))
            resultdf = pd.DataFrame()
            print(outjson)
            col = True
            for k, v in outjson.items():
                if k == "data":
                    Datasetname = v  # sep.join(v.split("_")[1:-1]) + "_p"
                    data_id = v
                if k == "shape":
                    shapetrain = v["ytrain"]
                    shapetest = v["ytest"]
                if k == "para":
                    para = v
                if k == "timespend":
                    timespend = v
                if k == "results":
                    results = v
                if k == "targetname":
                    target = v
                if k == "models":
                    fmodels = v
                print(k, v)
            if check(Datasetname, shapetrain, shapetest, para, timespend, results):
                plotfile.write(str(Datasetname) + ",")
                plotfile.write(str(data_id) + ",")
                plotfile.write(str(para["folds"]) + ",")
                plotfile.write(
                    str(resultfile[1][7:] + "_" +
                        resultfile[-1].split(".")[0]) + ","
                )
                plotfile.write(str(date) + ",")
                plotfile.write(str(results["logloss"]) + ",")
                print(data_id, results["logloss"])
                plotfile.write(str(results["AUC"]) + ",")
                plotfile.write(str(results["f1"]) + ",")
                plotfile.write(str(results["ACC"]) + ",")
                plotfile.write(str(para["time"]) + ",")
                plotfile.write(str(timespend) + ",")
                plotfile.write(str(para["cores"]) + ",")
                plotfile.write(str(para["fitmetrics"]) + ",")
                plotfile.write(str(para["framework"]) + ",")
                plotfile.write(str(shapetrain[0]) + ",")
                plotfile.write(str(shapetest[0]) + ",")
                plotfile.write(str(fmodels))
                plotfile.write("\n")
    plotfile.close()
    return dataname, auclist, loglosslist, acclist


def compile_results(dirt, task_token, taskname, date=''):
    # Create target Directory if don't exist
    # DATASET=          // data set name (str)
    # TARGET=           // target variable name (str)
    # SUITE=            // binaryTarget, intervalTarget, nominalTarget, binaryRareEvent (str)
    # AUTOML_MODE=      // MLPA-Std, MLPA-Enh, AUTOSKLEARN, TPOT (str)
    # AUTOML_TIME=      // run time in mts specified by user (int)
    # AUTOML_CRITERIA=  // Type of loss (MCLL, ASE etc)
    # RUN_TAG=          // Tag for the run
    # RUN_STATISTIC=    // MCLL or ASE statistic (float)
    # RUN_MSG='NONE'    // short error msg (str)
    # RUN_DATETIME=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #                   // run time (str) in format 2020-05-22 18:45:33
    # RUN_DURATION=     // run duration in mts
    # NUMOBS            // size of the test partition
    # RUN_MODEL     // name of the winning model

    if not os.path.exists(dirt):
        os.mkdir(dirt)
        print("Directory ", dirt,  " Newly Created ")
    osbenchmarkfile = dirt + '/opensource_benchmark' + task_token + ".csv"
    bm_results = open(osbenchmarkfile, "w")
    bm_results.write(
        "DATASET,TARGET,SUITE,AUTOML_MODE,AUTOML_TIME,AUTOML_CRITERIA,RUN_TAG,RUN_STATISTIC,RUN_MSG,RUN_DATETIME,RUN_DURATION,NUMOBS,RUN_MODEL\n")
    sum_results = pd.DataFrame(columns=["DATASET","TARGET","SUITE","AUTOML_MODE","AUTOML_TIME","AUTOML_CRITERIA","RUN_TAG","RUN_STATISTIC","RUN_MSG","RUN_DATETIME","RUN_DURATION","NUMOBS","RUN_MODEL"])
    print("sum_results")
    print(sum_results)
    dataname = []
    auclist = []
    loglosslist = []
    acclist = []
    datalist = glob.glob(dirt + "result*" + task_token + "*")
    print(datalist)
    for dataresult in datalist:
        getname = True
        gettime = True
        temp = []
        resultfile = dataresult.split("_")
        print(resultfile)
        outjson = json.load(open(dataresult))
        resultdf = pd.DataFrame()
        print(outjson)
        col = True
        for k, v in outjson.items():
            if k == "data":
                Datasetname = v  # sep.join(v.split("_")[1:-1]) + "_p"
                data_id = v
            if k == "shape":
                shapetrain = v["ytrain"]
                shapetest = v["ytest"]
            if k == "para":
                para = v
            if k == "timespend":
                timespend = v
            if k == "results":
                results = v
            if k == "targetname":
                target = v
            if k == "models":
                fmodel = v
            print(k, v)
        if check(Datasetname, shapetrain, shapetest, para, timespend, results):
            # ("DATASET,TARGET,SUITE,AUTOML_MODE,AUTOML_TIME,AUTOML_CRITERIA,RUN_TAG,RUN_STATISTIC,RUN_MSG,RUN_DATETIME,RUN_DURATION,NUMOBS,RUN_MODEL\n")
            result_dict = dict()
            bm_results.write(Datasetname+",")  # DATASET
            result_dict['DATASET'] = Datasetname
            bm_results.write(str(target)+",")  # TARGET
            result_dict['TARGET'] = target
            bm_results.write(str(taskname)+",")  # SUITE
            result_dict['SUITE'] = taskname

            bm_results.write(str(para["framework"]).upper() + ",")
            result_dict['AUTOML_MODE'] = str(para["framework"]).upper()
            bm_results.write(str(para["time"]) + ",")
            result_dict['AUTOML_TIME'] =para["time"]
            if para["fitmetrics"] == 'log_loss':
                bm_results.write("MCLL,")  # MCLL
                result_dict['AUTOML_CRITERIA'] = 'MCLL'
            elif para["fitmetrics"] == 'ACC':

                bm_results.write("ASE,")
                result_dict['AUTOML_CRITERIA'] = "ASE"
            else:
                bm_results.write("AUC,")
                result_dict['AUTOML_CRITERIA'] = 'AUC'
                print("AUC as main statistics")

            bm_results.write(str(task_token)+",")
            result_dict['RUN_TAG'] = task_token
            if para["fitmetrics"] == 'log_loss':
                bm_results.write(str(results["logloss"]) + ",")  # MCLL
                result_dict['RUN_STATISTIC'] = results["logloss"]
            elif para["fitmetrics"] == 'ACC':
                bm_results.write(str(results["ACC"]) + ",")
                result_dict['RUN_STATISTIC'] = results["ACC"]
            else:
                bm_results.write(str(results["AUC"]) + ",")
                result_dict['RUN_STATISTIC'] = results["AUC"]
            bm_results.write("NONE,")
            result_dict['RUN_MSG'] = "NONE"
            date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            bm_results.write(str(date) + ",")  # DATETIME,
            result_dict['RUN_DATETIME'] = date

            bm_results.write(str(float(timespend)/60) + ",")  # DURATION
            result_dict['RUN_DURATION'] = float(timespend)/60.0
            bm_results.write(str(shapetest[0]) + ",")
            result_dict['NUMOBS'] = shapetest[0]

            bm_results.write(str(fmodel)+'\n')
            result_dict['RUN_MODEL'] = str(fmodel)
        sum_results = sum_results.append(result_dict,ignore_index=True)
        print("results_dict",result_dict)
    bm_results.close
    print("sum_results")

    print(sum_results.head())
    datasetlist = sum_results['DATASET'].drop_duplicates()
    print("datasetlist\n",datasetlist)
    if not os.path.exists('results/opensource_benchmark.csv'):
        gresults = open(dirt + '/opensource_benchmark.csv', "w")
        gresults.write("DATASET,TARGET,SUITE,AUTOML_MODE,AUTOML_TIME,AUTOML_CRITERIA,RUN_TAG,RUN_STATISTIC,RUN_MSG,RUN_DATETIME,RUN_DURATION,NUMOBS,RUN_MODEL\n")
        print("new opensource_benchmark file")
    print(datasetlist.values.tolist())
    for i,d in enumerate(datasetlist):
        print(i,d)
        print(sum_results.columns.values)
        print("dataset subset\n",sum_results[sum_results["DATASET"]==str(d)])
        print(sum_results[sum_results['DATASET']==str(d)].columns.values)
        df = sum_results[sum_results['DATASET']==str(d)].sort_values(by='RUN_STATISTIC')
        df = df.groupby(['DATASET'], as_index=False).agg({'TARGET': 'first', 'SUITE': 'first', 'AUTOML_MODE': 'first', 'AUTOML_TIME': 'first', 'AUTOML_CRITERIA': 'first',
                                                                                       'RUN_TAG': 'first', 'RUN_STATISTIC': 'first', 'RUN_MSG': 'first', 'RUN_DATETIME': 'first', 'RUN_DURATION': 'first', 'NUMOBS': 'first', 'RUN_MODEL': 'first'})
        print("\nAppending results\n",df)

        df.to_csv(dirt + '/opensource_benchmark.csv', mode='a', header=True, index=False)
    #sum_results.astype({''})
    # group_results = sum_results.groupby(['DATASET'], as_index=False).agg({'TARGET': 'first', 'SUITE': 'first', 'AUTOML_MODE': 'first', 'AUTOML_TIME': 'first', 'AUTOML_CRITERIA': 'first',
    #                                                                                    'RUN_TAG': 'first', 'RUN_STATISTIC': 'mean', 'RUN_MSG': 'first', 'RUN_DATETIME': 'first', 'RUN_DURATION': 'mean', 'NUMOBS': 'first', 'RUN_MODEL': 'first'})
    # # ,'ERROR':'first','MCLL':'mean','MLPA_FOLDER':'first','MODELING_MODE':'first','SAMPLING_ENABLED':'first','SUITE':'first','SUITE_TYPE':'first','TAG':'first','TARGET':'first','NOTE':'first'})
    # print(group_results)
    # if os.path.exists('results/opensource_benchmark.csv'):

    #     group_results.to_csv(dirt + '/opensource_benchmark.csv',
    #                          mode='a', header=False, index=False)
    # else:

    #     group_results.to_csv(dirt + '/opensource_benchmark.csv',
    #                          mode='a', header=True, index=False)



def get_results_reg(dirt, date, task_token):
    # print(dirt,task_token,task_token[2:4])
    plotfile = open(dirt + task_token + "_summary.csv", "w")
    plotfile.write(
        "Dataset_name,data_id,Folds,Run_id,date,R2,MSE,MAE1,MAE2,Time_limit,Timespend,Cores,Fitmetric,Framework,train_size,test_size\n"
    )
    # time\tmetrix1\tmetrix2\tcore\tfold\tframework\tlogloss\tAUC\tf1\tACC\n')
    sep = "_"
    if True:
        dataname = []
        auclist = []
        loglosslist = []
        acclist = []
        datalist = glob.glob(dirt + "result*" + task_token + "*json")
        print(datalist)
        for dataresult in datalist:
            getname = True
            gettime = True
            temp = []

            resultfile = dataresult.split("_")
            print(resultfile)
            outjson = json.load(open(dataresult))
            resultdf = pd.DataFrame()
            print(outjson)
            col = True
            for k, v in outjson.items():
                if k == "data":
                    # Datasetname = sep.join(v.split("_")[1:-1]) + "_p"
                    # data_id = v.split("_")[0]
                    Datasetname = sep.join(v.split("_")[1:-1]) + "_p"
                    data_id = v
                if k == "shape":
                    shapetrain = v["ytrain"]
                    shapetest = v["ytest"]
                if k == "para":
                    para = v
                if k == "timespend":
                    timespend = v
                if k == "results":
                    results = v
                print(k, v)
            # check(Datasetname,shapetrain,shapetest,para,timespend,results):
            if True:
                plotfile.write(str(Datasetname) + ",")
                plotfile.write(str(data_id) + ",")
                plotfile.write(str(para["folds"]) + ",")
                plotfile.write(
                    str(resultfile[1][7:] + "_" +
                        resultfile[-1].split(".")[0]) + ","
                )
                plotfile.write(str(date) + ",")
                plotfile.write(str(results["r2"]) + ",")
                print(data_id, results["r2"])
                plotfile.write(str(results["MSE"]) + ",")
                plotfile.write(str(results["MAE1"]) + ",")
                plotfile.write(str(results["MAE2"]) + ",")
                plotfile.write(str(para["time"]) + ",")
                plotfile.write(str(timespend) + ",")
                plotfile.write(str(para["cores"]) + ",")
                plotfile.write(str(para["fitmetrics"]) + ",")
                plotfile.write(str(para["framework"]) + ",")
                plotfile.write(str(shapetrain[0]) + ",")
                plotfile.write(str(shapetest[0]) + ",")
                plotfile.write("\n")
    plotfile.close()
    return dataname, auclist, loglosslist, acclist


if __name__ == "__main__":
    datadirt = "/root/data/"
    runlist = []
    rep = 1
    foldlist = []
    task = task_token
    outputdir = "./results/"  # '/run/user/yozhuz/automl/results/'
    task = "bt"
    timelist = [900]
    for lf, locfold in enumerate(timelist):
        ldirt = dirt + locfold + "s/"
        dataname, auclist, loglosslist, acclist = get_results_reg(ldirt, task)
        # plot_result(title,dataname,loglosslist,task_token,lf,fig,ax,ik)
        # plt.legend(loc='upper center',ncol=3,fontsize=12,shadow=True)
    # plt.savefig(title+locfold+'.png',dpi=700)

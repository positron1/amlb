import pandas as pd
import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
import json


def check(Datasetname, shapetrain, shapetest, para, timespend, results):
    return True


def get_results_clf(dirt, date, key):
    # Create target Directory if don't exist
    if not os.path.exists(dirt):
        os.mkdir(dirt)
        print("Directory " , dirt ,  " Newly Created ")
    plotfile = open(dirt + key + "_summary.csv", "w")
    plotfile.write(
        "Dataset_name,data_id,Folds,Run_id,date,Logloss,AUC,f1,ACC,Time_limit,Timespend,Cores,Fitmetric,Framework,train_size,test_size\n"
    )
    # time\tmetrix1\tmetrix2\tcore\tfold\tframework\tlogloss\tAUC\tf1\tACC\n')
    sep = "_"
    if True:
        dataname = []
        auclist = []
        loglosslist = []
        acclist = []
        datalist = glob.glob(dirt + "result*" + key + "*")
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
                    Datasetname = sep.join(v.split("_")[1:-1]) + "_p"
                    data_id = v.split("_")[0]
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
            if check(Datasetname, shapetrain, shapetest, para, timespend, results):
                plotfile.write(str(Datasetname) + ",")
                plotfile.write(str(data_id) + ",")
                plotfile.write(str(para["folds"]) + ",")
                plotfile.write(
                    str(resultfile[1][7:] + "_" + resultfile[-1].split(".")[0]) + ","
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
                plotfile.write("\n")
    plotfile.close()
    return dataname, auclist, loglosslist, acclist

    # output_json = json.load(open('C://Users/yozhuz/Documents/atoml/automl/results/1800s/id0_uci_bank_marketing_p.sas7bdat2019Aug3119815autosklearn.json'))


def get_results_reg(dirt, date, key):
    # print(dirt,key,key[2:4])
    plotfile = open(dirt + key + "_summary.csv", "w")
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
        datalist = glob.glob(dirt + "result*" + key + "*json")
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
                    Datasetname = sep.join(v.split("_")[1:-1]) + "_p"
                    data_id = v.split("_")[0]
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
            if True:  # check(Datasetname,shapetrain,shapetest,para,timespend,results):
                plotfile.write(str(Datasetname) + ",")
                plotfile.write(str(data_id) + ",")
                plotfile.write(str(para["folds"]) + ",")
                plotfile.write(
                    str(resultfile[1][7:] + "_" + resultfile[-1].split(".")[0]) + ","
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
    outputdir = "./results/"  #'/run/user/yozhuz/automl/results/'
    task = "bt"
    timelist = [900]
    for lf, locfold in enumerate(timelist):
        ldirt = dirt + locfold + "s/"
        dataname, auclist, loglosslist, acclist = get_results_reg(ldirt, task)
        # plot_result(title,dataname,loglosslist,key,lf,fig,ax,ik)
        # plt.legend(loc='upper center',ncol=3,fontsize=12,shadow=True)
    # plt.savefig(title+locfold+'.png',dpi=700)


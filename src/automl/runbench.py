from numpy.lib.twodim_base import mask_indices
import serialize_sk as sr
import autosklearn.classification
import autosklearn.regression
import sklearn.datasets
from tpot import TPOTClassifier
from tpot import TPOTRegressor

###### Read in data
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, vstack
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn import preprocessing
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
    roc_auc_score,
    accuracy_score,
    log_loss,
    f1_score,
)
from sklearn.externals import joblib
import pickle
import re

from utils import *
##################################################
if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")


def getfitmetrics(fitmetrics):
    if fitmetrics == autosklearn.metrics.roc_auc:
        return "AUC"
    elif fitmetrics == autosklearn.metrics.log_loss:
        return "LogLoss"


def savemodel(timeforjob, resultfile, automl):
    resultfileout = open(
        "results/" + str(timeforjob) + "s/finalmodels" + resultfile, "w"
    )
    resultfileout.write("show_models")
    resultfileout.write(str(automl.show_models()))
    resultfileout.write("\ncv_results\n")
    resultfileout.write(str(automl.cv_results_))
    resultfileout.close()


def metric(task, y_test, y_pred, y_pred_prob):
    metrics = dict()
    if task == 'it':
        metrics['r2'] = r2_score(y_test, y_pred)
        metrics['MSE'] = mean_squared_error(y_test, y_pred)
        metrics['MAE1'] = mean_absolute_error(y_test, y_pred)
        metrics['MAE2'] = median_absolute_error(y_test, y_pred)
    elif task == 'bre' or task == 'bt':
        metrics['logloss'] = log_loss(y_test, y_pred_prob)
        metrics['AUC'] = roc_auc_score(y_test, y_pred)
        metrics['f1'] = f1_score(y_test, y_pred)
        metrics['ACC'] = accuracy_score(y_test, y_pred)
    return metrics


def autoprep(dirt, dataset, targetname):
    if targetname:
        targetname = dataset
    else:
        dirt = dataset
        # use the last one as target and print it out
    return nfeatures, cfeatures, target


def autoreg(
    metalearning,
    framework,
    feat_type,
    timeforjob,
    foldn,
    ncore,
    X_train,
    y_train,
    fitmetrics,
):
    if metalearning:
        metan = 25
        en_size = 50
    else:
        metan = 0
        en_size = 1
    print("meta learning\t", metalearning, en_size, metan)
    print(X_train, y_train)
    if foldn == 0:
        automl = autosklearn.regression.AutoSklearnRegressor(
            time_left_for_this_task=timeforjob,
            resampling_strategy="holdout",
            resampling_strategy_arguments={"train_size": float(5 / 7)},
            ensemble_memory_limit=20720,
            ensemble_size=en_size,
            initial_configurations_via_metalearning=metan,
            seed=1,
            n_jobs=ncore,
            delete_tmp_folder_after_terminate=True,
            ml_memory_limit=20720 * 2,
            per_run_time_limit=timeforjob,
        )
        if len(feat_type) > 0:
            automl.fit(
                X_train.copy(), y_train.copy(), metric=fitmetrics, feat_type=feat_type
            )
        else:
            automl.fit(
                X_train.copy(), y_train.copy()
            )  # ,metric=fitmetrics)#,feat_type = feat_type)
    #        automl.fit(X_train.copy(), y_train.copy(),metric=fitmetrics,feat_type=feat_type)
    #        automl.refit(X_train.copy(), y_train.copy())#,feat_type=feat_type)#,metric=autosklearn.metrics.roc_auc)
    else:
        automl = autosklearn.regression.AutoSklearnRegressor(
            time_left_for_this_task=timeforjob,
            delete_tmp_folder_after_terminate=True,
            seed=1,
            ensemble_size=en_size,
            initial_configurations_via_metalearning=metan,
            per_run_time_limit=timeforjob,
            ensemble_memory_limit=20720,
            ml_memory_limit=20720 * 2,
            resampling_strategy_arguments={"folds": int(foldn)},
            resampling_strategy="cv",
            n_jobs=ncore,
        )
        if len(feat_type) > 0:
            automl.fit(
                X_train.copy(), y_train.copy(), metric=fitmetrics, feat_type=feat_type
            )
        else:
            automl.fit(
                X_train.copy(), y_train.copy()
            )  # ,metric=fitmetrics)#,feat_type = feat_type)

        automl.refit(
            X_train.copy(), y_train.copy()
        )  # ,feat_type = feat_type)#,metric=autosklearn.metrics.roc_auc)
    return automl


def autoclf(
    metalearning,
    framework,
    feat_type,
    timeforjob,
    foldn,
    ncore,
    X_train,
    y_train,
    fitmetrics,
):
    if metalearning:
        metan = 25
        en_size = 50
    else:
        metan = 0
        en_size = 1
    print("meta learning\t", metalearning, en_size, metan)
    if foldn == 0:
        automl = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=timeforjob,
            per_run_time_limit=timeforjob,
            delete_tmp_folder_after_terminate=True,
            ensemble_size=en_size,
            initial_configurations_via_metalearning=metan,
            seed=1,
            ensemble_memory_limit=20720,
            resampling_strategy="holdout",
            ml_memory_limit=20720 * 2,
            resampling_strategy_arguments={"train_size": float(5 / 7)},
            n_jobs=ncore,
        )
        if len(feat_type) > 0:
            automl.fit(
                X_train.copy(), y_train.copy(), metric=fitmetrics, feat_type=feat_type
            )
        else:
            automl.fit(
                X_train.copy(), y_train.copy(), metric=fitmetrics
            )  # ,feat_type = feat_type)

    else:
        automl = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=timeforjob,
            delete_tmp_folder_after_terminate=True,
            seed=1,
            per_run_time_limit=timeforjob,
            ensemble_memory_limit=20720,
            ml_memory_limit=20720 * 2,
            resampling_strategy_arguments={"folds": int(foldn)},
            resampling_strategy="cv",
            ensemble_size=en_size,
            initial_configurations_via_metalearning=metan,
            n_jobs=ncore,
        )
        if len(feat_type) > 0:
            automl.fit(
                X_train.copy(), y_train.copy(), metric=fitmetrics, feat_type=feat_type
            )
        else:
            automl.fit(
                X_train.copy(), y_train.copy(), metric=fitmetrics
            )  # ,feat_type = feat_type)

        automl.refit(
            X_train.copy(), y_train.copy()
        )  # ,feat_type = feat_type)#,metric=autosklearn.metrics.roc_auc)
    return automl


def get_pipe(pipe_str):
    try:
        outer = re.compile("\(.+\)")
        m = outer.search(pipe_str)
        print(m)
        innerre = re.compile("\{(.+)},{(.+)\}")
        regex = r"\{(.*?)\}"
        matches = re.finditer(regex, m.group(), re.MULTILINE | re.DOTALL)
        fm = 'Dummy Classifier'
        for match in matches:
            print("\n")
            x = eval("{" + match.group(1) + "}")
            print(x)
            try:
                fm = x['classifier:__choice__']
            except:
                pass
    except:
        m = pipe_str[pipe_str.find('('):-1]
        print(m)
        matches = m.split('}')[0][2:]
        print(matches)
        fm = 'Dummy Classifier'
        x = eval("{" + matches + "}")
        print(x)
        try:
            fm = x['classifier:__choice__']
        except:
            pass
    return fm


def get_fensemble(fmodel):
    fensemble = dict()
    print(fmodel)
    for i, n in enumerate(fmodel):
        f, c = n
        print(str(f), str(c))
        fensemble[get_pipe(str(c))+str(i)] = float(f)
    return fensemble


def get_run_info(
    metalearning,
    automl,
    dataset,
    shape,
    timeforjob,
    ncore,
    foldn,
    framework,
    resultsfile,
    fitmetrics,
    metrics,
    timespend,
    prepb,
    outputdir,
    target,
):
    runs = dict()
    runs["data"] = str(dataset)
    runs["shape"] = dict()
    runs["shape"]["xtrain"] = shape[0]
    runs["shape"]["ytrain"] = shape[1]
    runs["shape"]["xtest"] = shape[2]
    runs["shape"]["ytest"] = shape[3]
    runs["para"] = dict()
    runs["para"]["prep_data"] = prepb
    runs["para"]["time"] = timeforjob
    runs["para"]["fitmetrics"] = str(fitmetrics)
    runs["para"]["refitmetrics"] = "def"
    runs["para"]["cores"] = str(ncore)
    runs["para"]["folds"] = str(foldn)
    runs["para"]["framework"] = str(framework)
    runs["timespend"] = timespend
    runs["results"] = dict(metrics)
    runs["metalearning"] = metalearning
    runs["targetname"] = target
    if framework == 'autosklearn':
        runs["models"] = get_fensemble(automl.get_models_with_weights())

    elif framework == 'tpot':
        print("pareto_front_fitted_pipeline_",automl.pareto_front_fitted_pipelines_)
        print("fitted_pipeline_",automl.fitted_pipeline_)
        runs["models"] = automl.fitted_pipeline_.steps[-1][0]

    jsonf = json.dumps(runs)
    print(runs)

    #    tpot = 
    f = open(
        outputdir
        + "/"
        + str(timeforjob)
        + "s/result_"
        + str(getfitmetrics(fitmetrics))
        + resultsfile
        + ".json",
        "w",
    )
    #savemodel(timeforjob, resultsfile, automl)
    f.write(jsonf)
    f.close()


def save_prob(timeforjob, dataset, resultsfile, foldn, y_pred, y_pred_prob, outputdir):
    if not os.path.exists(dirt):
        os.mkdir(dirt)
        print("Directory ", dirt,  " Newly Created ")
    briefout = open(
        outputdir
        + "/"
        + str(timeforjob)
        + "s/"
        #+ dataset
        + resultsfile
        + str(foldn)
        + "fresult.csv",
        "w",
    )
    briefout.write("#ypred\typred_prob\n")
    for i, y in enumerate(y_pred):
        briefout.write(str(y) + "\t" + str(y_pred_prob[i]) + "\n")
    briefout.close()


def autoframe(
    task,
    metalearning,
    prepb,
    feat_type,
    resultsfile,
    X_train,
    y_train,
    X_test,
    y_test,
    dataset,
    framework,
    foldn,
    ncore,
    timeforjob,
    dirt,
    meta,
    fitmetrics,
    outputdir,
    target,
):

    shape = []
    shape = [X_train.shape, y_train.shape, X_test.shape, y_test.shape]
    start = time.time()
    if framework == 'autosklearn':
        if task == "bt" or task == "bre":
            automl = autoclf(
                metalearning,
                framework,
                feat_type,
                timeforjob,
                foldn,
                ncore,
                X_train,
                y_train,
                fitmetrics,
            )
            y_pred_prob = automl.predict_proba(X_test)
        elif task == "it":
            automl = autoreg(
                metalearning,
                framework,
                feat_type,
                timeforjob,
                foldn,
                ncore,
                X_train,
                y_train,
                fitmetrics,
            )
            y_pred_prob = []
        y_pred = automl.predict(X_test)

        ###################################################################
    elif framework == 'tpot':
        if task == "bt" or task == "bre":
            tpot = TPOTClassifier(
                max_time_mins=int(timeforjob/60), max_eval_time_mins=float(timeforjob/60), n_jobs=ncore, verbosity=2)
            tpot.fit(X_train, y_train)
            y_pred_prob = tpot.predict_proba(X_test)
        elif task == "it":
            tpot = TPOTRegressor(
                generations=5, population_size=50, verbosity=2)
            y_pred_prob = []
        automl = tpot

        y_pred = tpot.predict(X_test)
        print(tpot.score(X_test, y_test))

    end = time.time()
    timespend = float(end - start)
    ###################################################################
    ###################################################################
    save_prob(timeforjob, dataset, resultsfile,
              foldn, y_pred, y_pred_prob, outputdir)
    metrics = metric(task, y_test, y_pred, y_pred_prob)
    print(dataset)
    get_run_info(
        metalearning,
        automl,
        dataset,
        shape,
        timeforjob,
        ncore,
        foldn,
        framework,
        resultsfile,
        fitmetrics,
        metrics,
        timespend,
        prepb,
        outputdir,
        target,
    )


def shortname(dirt, dataset, meta):
    print("\n\ndataset\t", dataset)
    if str(dataset[-2:]) == '_p':
        myid = dataset[:-2]
    else:
        myid = dataset
    print('\nmyid:\t', myid, '\n')

    if meta[-2:] == '_p':
        meta = meta[:-2]
    print(dirt + "tmp_metadata/" + meta+'_meta.csv')
    return dataset, meta, myid


def preprocessing(dirt, meta, prepb, dataset, taskname):
        # if there is meta info, read inputs and targets, if not, figure it out.
    if os.path.exists(dirt+"/tmp_metadata/" + meta+'_meta.csv'):
        print("\nMETA data file at:\t" + "tmp_metadata/" + meta+'_meta.csv')
        if prepb:
            nfeatures, cfeatures, target = meta_info(dirt, meta, prepb)
            inputs = nfeatures+cfeatures
        else:
            inputs, target = meta_info(dirt, meta, prepb)
            nfeatures = []
            cfeatures = []

        data, X, y, X_train, y_train, X_test, y_test, feat_type = prep(
            prepb,
            dataset,
            taskname,
            dirt,
            nfeatures,
            cfeatures,
            inputs,
            target,
            delim=",",
            indexdrop=False,)
    else:
        print("Error")
        sys.exit()
    return data, X, y, X_train, y_train, X_test, y_test, feat_type, target


def runbenchmark(
    task,
    taskname,
    metalearning,
    prepb,
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
):
    dataset, meta, myid = shortname(dirt, dataset, meta)
    feat_type = []

    try:
        data, X, y, X_train, y_train, X_test, y_test, feat_type, target = preprocessing(
            dirt, meta, prepb, dataset, taskname)
        for timeforjob in timelist:
            for ncore in corelist:
                for foldn in foldlist:
                    for rp in range(rep):
                        current_time = DateTime(time.time(), "US/Eastern")
                        resultsfile = (
                            myid
                            + "_"
                            + str(framework)
                            + "_"
                            + str(foldn)
                            + "f_"
                            + str(ncore)
                            + "c_"
                            + str(timeforjob)
                            + "s_task_"
                            + str(task_token)
                            + "_rep"
                            + str(rp)
                            + "of"
                            + str(rep)
                            + "_"
                            + str(current_time.aMonth())
                            + "_"
                            + str(current_time.day())
                            + "_"
                            + str(current_time.h_24())
                            + str(current_time.minute())
                            + str(time.time())[:2]
                        )
                        # resultsfile = myid+"_"+str(framework)+'_'+str(foldn)+'f_'+str(ncore)+"c_"+str(timeforjob)+"s_"+str(current_time.year()) + str(current_time.aMonth())+ str(current_time.day()) + \
                        # str(current_time.h_24()) + str(current_time.minute())  + str(time.time())[:2]
                        print(
                            "\nstarting:\t",
                            framework,
                            "\t",
                            foldn,
                            " fold\t",
                            ncore,
                            " core\t",
                            timeforjob,
                            " seconds\n",
                            file=logfile,
                        )
                        autoframe(
                            task,
                            metalearning,
                            prepb,
                            feat_type,
                            resultsfile,
                            X_train.copy(),
                            y_train.copy(),
                            X_test.copy(),
                            y_test.copy(),
                            dataset,
                            framework,
                            foldn,
                            ncore,
                            timeforjob,
                            dirt,
                            meta,
                            fitmetrics,
                            outputdir,
                            target,
                        )

                        print(
                            "Finishing:\t",
                            framework,
                            "\t",
                            foldn,
                            " fold\t",
                            ncore,
                            " core\t",
                            timeforjob,
                            " seconds\n",
                        )
    except:
        print("\nfail in:\t", dataset)
        traceback.print_exc(file=sys.stdout)

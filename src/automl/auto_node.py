from preprocess import *
class autoframe(object,task_preprocess):
    def __init__(self, metalearning, framework, timeforjob, ncore, rp, rep, timelist, corelist, foldlist,):
        self.metalearning = metalearning
        self.framwork = framework
        self.task_token = task_preprocess.task_token
        self.timeforjob = timeforjob
        self.foldn = task_preprocess.foldn
        self.ncore = ncore

    def resultsfile(self):
        current_time = DateTime(time.time(), "US/Eastern")
        resultsfile = (
            myid
            + "_"
            + str(self.framework)
            + "_"
            + str(self.foldn)
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
            file=task_preprocess.logfile,
        )

    def autoframe(self):
        data, X, y, X_train, y_train, X_test, y_test, feat_type = task_preprocess.data_prep()
        self.task_type = task_preprocess.task_type

        shape = []
        shape = [X_train.shape, y_train.shape, X_test.shape, y_test.shape]
        start = time.time()
        if framework == 'autosklearn':
            frame_autosklearn()
            ###################################################################
        elif framework == 'tpot':
            frame_tpot()
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
            #        automl,
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
    def frame_autosklearn(self):
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
    def frame_tpot(self):
            if task == "bt" or task == "bre":
                tpot = TPOTClassifier(
                    max_time_mins=int(timeforjob/60), max_eval_time_mins=float(timeforjob/100), n_jobs=ncore, verbosity=2)
                tpot.fit(X_train, y_train)
                y_pred_prob = tpot.predict_proba(X_test)
            elif task == "it":
                tpot = TPOTRegressor(
                    generations=5, population_size=50, verbosity=2)
                y_pred_prob = []

            y_pred = tpot.predict(X_test)
            print(tpot.score(X_test, y_test))

    def save_prob(self,timeforjob, dataset, resultsfile, foldn, y_pred, y_pred_prob, outputdir):
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

    def metric(self,task, y_test, y_pred, y_pred_prob):
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

    def get_run_info(self,
        metalearning,
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
        print(runs)
        #    tpot = json.dumps(jsonpickle.encode(runs))
        jsonf = json.dumps(runs)
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


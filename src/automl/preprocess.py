
import numpy as np
import pandas as pd
import sklearn.metrics
import glob
from sas7bdat import SAS7BDAT

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import autosklearn.classification

from sklearn.compose import ColumnTransformer
import time

from DateTime import DateTime
import secrets

class benchmark_task():

    def __init__(self,dirt,task_type,timelimit,foldn,rep):
        self.path = dirt
        self.task_type = task_type
        self.timelimit = timelimit
        self.foldn = foldn
        self.rep = rep
        self.task_token = secrets.token_hex(8)
        self.get_datalist_csv()
        self.get_datalist_sas()
        self.get_metadatalist()
        self.get_time()
        self.get_task_fitmetrics()
        self.logging()# generate unique token for this task

    def get_task_fitmetrics(self,fitmetrics=False):
        """ Get Task Type and fit metrics

        Args:
            task_type ([string]): [bt/bre/it]

        Returns:
            [string,string]: [taskname,fimetrics]
        """

        if self.task_type == "bt":
            self.taskname = "binaryTarget"
            if not fitmetrics:
                fitmetrics = autosklearn.metrics.log_loss
        elif self.task_type == "bre":
            self.taskname = "binaryRareEvent"
            if not fitmetrics:
                fitmetrics = autosklearn.metrics.log_loss
        elif self.task_type == "it":
            self.taskname = "intervalTarget"
            if not fitmetrics:
                fitmetrics = autosklearn.metrics.mean_squared_error
        self.fitmetrics = fitmetrics

    def get_datalist_csv(self):
        """ Get Data Files in CSV Formats

        Returns:
            [list]: [sorted list of data files in csv format]
        """
        csvdatalist = glob.glob(self.path + self.taskname+"/*.csv")
        csvdatalist = [i[:-4] for i in csvdatalist]
        self.csvlist = sorted(csvdatalist)


    def get_datalist_sas(self):
        """ Get Data Files in sas7bdat Formats

        Returns:
            [list]: [sorted list of data files in sas format]
        """
        sasdatalist = glob.glob(self.path + self.taskname+"/*sas7bdat")
        sasdatalist = [i[:-9] for i in sasdatalist]
        self.saslist = sorted(sasdatalist)

    def get_metadatalist(self):
        """ Get Metadata file

        Returns:
            [list]: [sored list of meta data files]
        """
        metalist = glob.glob(self.path + "/tmp_metadata/*meta.csv")
        metalist = [i[:-9] for i in metalist]
        self.metalist =  sorted(metalist)

    def get_time(self):
        """ Get timestamp at the initiation of task

        Returns:
            [string]: [time at task initiation]
        """
        current_time = DateTime(time.time(), "US/Eastern")
        self.timestamp = (
            str(current_time.year())
            + str(current_time.aMonth())
            + str(current_time.day())
            + str(current_time.h_24())
            + str(current_time.minute())
            + str(time.time())[:2]
        )

    def logging(self):
        """ Logging information

        Returns:
            [string]: [path and filename]
        """
        print("working dirt\t", self.path)
        print("csv datalist\n", self.csvlist)
        print("sas datalist\n", self.saslist)
        print("metadatalit\n", self.metalist)
        self.logfile = "results/log_" + "dataset_" + str(self.timelimit) + "s_" + str(self.foldn) + "f_rep" + str(self.rep) + "_task_" + str(self.task_token)+ ".txt"
class task_preprocess(benchmark_task):
    def __init__(self,dirt,dataname,metalist,prepb,task_type,timelimit,foldn,rep):
        self.dataname = dataname
        self.path = dirt
        self.metalist = metalist
        self.prepb = prepb
        super().__init__(dirt,task_type,timelimit,foldn,rep)
        self.check_metadata()

    def check_metadata(self):
        """ Check if metadata exists

        Args:
            dataname ([string]): [dataset name]
            csvdatalist ([list]): [list of dataset names in csv format]
            sasdatalist ([list]): [list of dataset names in sas7bdat format]
            metalist ([list]): [list metadata files]

        Returns:
            [str]: [dataset name and its metadata name]
        """
        print(self.dataname)
        if self.dataname in self.metalist or self.dataname[:-2] in self.metalist:
            self.meta = self.dataname
        else:
            self.meta = '0'

    def get_shortname(self):
        """[summary]
        """
        print("\n\ndataset\t", self.dataname)
        if str(self.dataname[-2:]) == '_p':
            myid = self.dataname[:-2]
        else:
            myid = self.dataname
        print('\nmyid:\t', myid, '\n')

        if self.meta[-2:] == '_p':
            self.meta = self.meta[:-2]
        print(self.path + "tmp_metadata/" + self.meta +'_meta.csv')

    def get_meta_info(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        if self.meta[-2:] == '_p':
            self.meta = self.meta[:-2]
        dmeta = pd.read_csv(self.path + "/tmp_metadata/" + self.meta+'_meta.csv')
        target = dmeta[dmeta["ROLE"] == "TARGET"]
        targetname = target["UNAME"].tolist()[0]
        inputs = dmeta[dmeta["ROLE"] == "INPUT"]
        if self.prepb:
            cinputs = inputs[inputs["type"] == "C"]
            self.cinputname = cinputs["UNAME"].tolist()
            ninputs = inputs[inputs["type"] == "N"]
            self.ninputname = ninputs["UNAME"].tolist()
            self.inputs = self.cinputname + self.ninputname
        else:
            self.inputs = inputs["UNAME"].tolist()
            self.ninputname = []
            self.cinputname = []

        print("meta_info: inputs\n", inputs)
        self.target = targetname

    def data_prep(self, delim=",", indexdrop=False):
            # if there is meta info, read inputs and targets, if not, figure it out.
        if len(self.meta)>0:
            self.get_shortname()
            print("\nMETA data file at:\t" + "tmp_metadata/" + self.meta+'_meta.csv')
            self.get_meta_info()
            data = self.read_data_csv()
            data, X, y, X_train, y_train, X_test, y_test, feat_type = self.data_transform(data)
            return data, X, y, X_train, y_train, X_test, y_test, feat_type

    def sas_to_csv(self):
        print("\n\nReading data from", self.path + self.taskname+'/' + self.dataname)
        with SAS7BDAT(self.path + self.taskname+'/'+ self.dataname +'.sas7bdat') as f:
            df = f.to_data_frame()
        print("\n\nData description:\n\n", df.describe())

        return df

    def read_data_csv(self):
            try:
                data = pd.read_csv(self.path + '/' + self.taskname +'/' + self.dataname+'.csv',
                                   delimiter=delim)  # panda.DataFrame
            except:
                df = self.sas_to_csv()
                cols = df.columns
                print("\n\nCheck column\n\n", cols)
                data = self.data_index(df)
                data = self.data_feature_target(data)
                data.to_csv(self.path + self.taskname +'/' + self.dataname +'.csv', encoding="utf-8",
                          index=False, header=True)
                # data = pd.read_csv(self.path + '/' + self.taskname+'/' + self.dataname+'.csv',
                #                    delimiter=delim)  # panda.DataFrame
            return data

    def data_index(self,data):
            data = data.astype({"_PartInd_": "int"})
            col = data.columns.values
            print(col)
            data = data.rename(str.upper, axis='columns')
            print(col)
            print(set(data[self.target]))
            print("inputs", self.inputs)
            self.target = self.target.upper()
            return data

    def data_feature_target(self,data):
            print(self.inputs)
            index_features = ["_dmIndex_", "_PartInd_"]
            self.index_features = [i.upper() for i in index_features]
            if not self.prepb:
                self.ninputname = list(set(self.inputs) & (set(data.select_dtypes(
                    include=["number"]))-set(self.index_features)-set([self.target])))
                self.cinputname = list(set(self.inputs) & (
                    set(data.select_dtypes(exclude=["number"]))-set(self.index_features)-set([self.target])))

            data = data[data[self.target].notna()]
            data[self.cinputname] = data[self.cinputname].astype("str")
            data[self.ninputname] = data[self.ninputname].astype("float32")
            data[self.target] = data[self.target].astype("str")
            print(set(data[self.target]))
            print(
                "\nCheck numerical features:\t", self.ninputname, data[self.ninputname].dtypes
            )
            print(
                "\nCheck catogorical features:\t",
                self.cinputname,
                data[self.cinputname].dtypes,
            )
            return data

    def data_transform(self,data):

            ###############################
            index_transformer = Pipeline(
                steps=[("imputer", SimpleImputer(strategy="constant", fill_value=-1))]
            )
            numeric_transformer = Pipeline(
                steps=[("imputer", SimpleImputer(strategy="median"))]
            )
            y_transformer = Pipeline(steps=[("orden", OrdinalEncoder())])
            categorical_transformer = Pipeline(steps=[("orden", OrdinalEncoder())])
            preprocessor = ColumnTransformer(
                transformers=[
                    ("index", index_transformer, self.index_features),
                    ("y", y_transformer, [self.target]),
                    ("num", numeric_transformer, self.ninputname),
                    ("cat", categorical_transformer, self.cinputname),
                ]
            )

            newcols = self.index_features + [self.target] + self.ninputname + self.cinputname
            print(newcols)
            newdata = data[newcols]
            print(newdata)
            print(set(newdata[self.target]))
            pdata = preprocessor.fit_transform(newdata)
            pddata = pd.DataFrame(pdata)
            col = pddata.columns.values
            print(col)
            X = pddata.drop(col[:3], axis=1)
            # Drop up to 3? or 2?
            print("drop columns", col[:2], col[:3])
            X_train = pddata[pddata[col[1]] < 2].drop(col[:3], axis=1)
            # pd.DataFrame(X).to_csv('X_vanilla.csv')
            X_test = pddata[pddata[col[1]] == 2].drop(col[:3], axis=1)
            y = pddata[col[2]]
            y_train = pddata[pddata[col[1]] < 2][col[2]]
            y_test = pddata[pddata[col[1]] == 2][col[2]]
            y_test = y_test.astype("float32")
            y_train = y_train.astype("float32")
            X_test = X_test.astype("float32")
            X_train = X_train.astype("float32")
            print(X_train.dtypes, X_train)
            print(set(y_train))
            ninputcols = len(self.ninputname)
            if self.prepb:
                feat_type = ["Numerical"] * ninputcols + ["Categorical"] * int(len(col) - 3 - ninputcols)
            else:
                feat_type = []
            #    ##########################################################
            return data, X, y, X_train, y_train, X_test, y_test, feat_type




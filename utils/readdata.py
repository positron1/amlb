############## Yongln Zhu #################
############ SAS Institute ################
########### July, 2019

'''
Transfer data from sas7bdat to csv
'''
from sas7bdat import SAS7BDAT
import pandas as pd
import os
import sys
import logging
import optparse


def sas_to_csv(dirt,dataset):
    print("\n\nReading data from",dirt+"sasdat/" + dataset+".sas7bdat")
    with SAS7BDAT(dirt+"sasdat/" + dataset+".sas7bdat") as f:
        df = f.to_data_frame()
    print("\n\nData description:\n\n",df.describe())
    cols = df.columns
    df.to_csv(dirt+dataset+'d.csv',encoding = 'utf-8',index = False,header =True)
    print("\n\nCheck column\n\n",cols)
    return df

def load_partition(dirt,dataset):
    df = sas_to_csv(dirt,dataset)
    #### last column _PartInd_ for train-1/validation-2/test-0/
    cols = df.columns
    df._PartInd_.astype(int)
    dtrain = df.loc[df[cols[-1]]==1]
    dvalidate = df.loc[df[cols[-1]]==2]
    dtest = df.loc[df[cols[-1]]==0]
    print("Train\n",dtrain.shape)
    print("Validate\n",dvalidate.shape)
    print("Test\n",dtest.shape)
    dtrain.to_csv(dirt+dataset+'dtrain.csv',encoding = 'utf-8',index= False,header =True)
    dtest.to_csv(dirt+dataset+'dtest.csv',encoding = 'utf-8',index = False,header =True)
    dvalidate.to_csv(dirt+dataset+'dvalid.csv',encoding = 'utf-8',index=False,header =True)

def main(options,args):
    dirt = options.path
    dataset = options.data
    load_partition(dirt,dataset)

if __name__ == '__main__':
    # default path and data
    dirt ="/root/data/"
    dataset = "uci_bank_marketing_p" 

    ######################################################
    parser = optparse.OptionParser()
    parser.set_usage("""%prog -p <pathtodata> -d [dataset]
    Convert sas7bdat files to csv. <pathtodata> is the path to a sas7bdat file and
    [dataset] is the optional path to the output csv file. If omitted, [dataset]
    defaults to the name of the input file with a csv extension. <pathtodata> can
    also be a glob expression in which case the [dataset] argument is ignored.
    Use --help for more details""")
    parser.add_option('-v','--verbose',action='store_True',default=True)
    parser.add_option('-p','--path',action='store',default=dirt)
    parser.add_option('-d','--data',action='store',default=dataset)
    ##############################################
    options, args = parser.parse_args()
    #if len(args) < 1:
    #    parser.print_help()
    #    sys.exit(1)
    main(options, args)

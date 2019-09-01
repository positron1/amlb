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
import arff
class data: 

    def __init__(self,dataset,metadata=True,dirt='/root/data/'):
        self.dataset = dataset
        self.metadata = metadata
        self.dirt = dirt

    def sas_to_csv(self,dirt,dataset):
        print("\n\nReading data from",dirt+"sasdat/" + dataset+".sas7bdat")
        with SAS7BDAT(dirt+"sasdat/" + dataset+".sas7bdat") as f:
            df = f.to_data_frame()
        print("\n\nData description:\n\n",df.describe())
        cols = df.columns
        df.to_csv(dirt+dataset+'d.csv',encoding = 'utf-8',index = False,header =True)
        arfffile=open(dirt+dataset+'.arff','w')
   #     arfffile.write(arff.dumps(df))
    #    arfffile.close()
        print(arff.dumps(df))   
     #print("\n\nCheck column\n\n",cols)
        return df

    def find_cat(self,df):
        #### inspired by pds from https://stackoverflow.com/questions/29803093/check-which-columns-in-dataframe-are-categorical
        df_cat=df.select_dtypes(exclude=["number","bool_","object_"])
        cat_feats = df_cat.columns.values
        if '_dmIndex_' in cat_feats:
            df_cat.drop(columns=['_dmIndex_','_PartInd_'])
        return df_cat,cat_feats
    
    def find_num(self,df):
        df_num =  df.select_dtypes(include=["number"])
        num_feats = df_num.columns.values
        if '_dmIndex_' in num_feats:
            df_num.drop(columns=['_dmIndex_','_PartInd_'])
        return df_num,num_feats

    def get_type(self,metafile):
        if self.metadata:
          metaf = pd.read_csv(dirt+"/meta/"+metafile)
          catlist = metaf[metaf['type']=='C']['NAME'].tolist()
          numlist = metaf[metaf['type']=='N']['NAME'].tolist()
        else:
          df = self.sas_to_csv(self.dirt,self.dataset)
          _, catlist = self.find_cat(df)
          _, numlist = self.find_num(df)
        if 'y' in catlist:
             catlist.remove('y')
        if '_dmIndex_' in numlist:
             numlist.remove('_dmIndex_')
             numlist.remove('_PartInd_')
        return catlist,numlist
    
    def load_partition(self,dirt,dataset):
        df = self.sas_to_csv(dirt,dataset)
        #### last column _PartInd_ for train-1/validation-2/test-0/
        cols = df.columns
        df._PartInd_.astype(int)
        dtrain = df.loc[df[cols[-1]]==1]
        dvalidate = df.loc[df[cols[-1]]==0]
        dtest = df.loc[df[cols[-1]]==2]
        print("Train\n",dtrain.shape)
        print("Validate\n",dvalidate.shape)
        print("Test\n",dtest.shape)
        return dtrain,dvalidate,dtest
    
    def partition_to_csv(self,dirt,dataset,dtrain,dvalidate,dtest):
        dtrain,dvalidate,dtest = self.load_partition(dirt,dataset)
        dtrain.to_csv(dirt+dataset+'dtrain.csv',encoding = 'utf-8',index= False,header =True)
        dtest.to_csv(dirt+dataset+'dtest.csv',encoding = 'utf-8',index = False,header =True)
        dvalidate.to_csv(dirt+dataset+'dvalid.csv',encoding = 'utf-8',index=False,header =True)
    
    def main(self,options,args):
        dirt = options.path
        dataset = options.data
        return self.load_partition(dirt,dataset)
    
    
    
    def read_df(self,dirt,dataset):
        dtrain,dvalidate,dtest = self.load_partition(dirt,dataset)
        find_cat()
    
        return df
    

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
    parser.add_option('-v','--verbose',action='store_true',default=True)
    parser.add_option('-p','--path',action='store',default=dirt)
    parser.add_option('-d','--data',action='store',default=dataset)
    ##############################################
    options, args = parser.parse_args()
    #if len(args) < 1:
    #    parser.print_help()
    #    sys.exit(1)
    data=data(dataset,metadata=True) 
    _,_,_ = data.main(options,args)
    catlist,numlist = data.get_type('uci_bank_marketing_meta.csv')
    print(catlist)
    print(numlist)

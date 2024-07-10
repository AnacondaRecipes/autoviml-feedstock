# Adapted from https://github.com/AutoViML/Auto_ViML/blob/master/Example%20Notebook/Auto_ViML_Demo.ipynb
import pandas as pd
datapath = 'https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/'
from autoviml import Auto_ViML
df = pd.read_csv(datapath+'titanic.csv')
target = 'Survived'
num = int(0.9*df.shape[0])
train = df[:num]
test = df[num:]
sample_submission=''
scoring_parameter = 'balanced-accuracy'
#test = train[-15:]
#test = pd.read_csv(datapath+'test.csv')
print(train.shape)
#print(test.shape)
print(train.head())
m, feats, trainm, testm = Auto_ViML(train, target, test, sample_submission,
                                    scoring_parameter=scoring_parameter,
                                    hyper_param='GS',feature_reduction=True,
                                    Boosting_Flag=True,Binning_Flag=False,
                                    Add_Poly=0, Stacking_Flag=False,
                                    Imbalanced_Flag=False,
                                    verbose=1)

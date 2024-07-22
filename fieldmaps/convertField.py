import pandas as pd

infile = "mediumfieldgrid.txt"

df = pd.read_csv(infile,sep = "\s+",header=None,names=['x','r','z','bz','br'])
df['bz'] = df['bz'] * 10 ** (-4)
df['br'] = df['br'] * 10 ** (-4)

df.to_csv('klm3Tfield.txt',sep = '\t',index = False,columns=['r','z','br','bz'],float_format='%.4f')

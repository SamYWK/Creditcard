# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 17:07:04 2017

@author: SamKao
"""

import numpy as np
import pandas as pd

df = pd.read_csv('creditcard.csv')
y = df['Class']
#df = df.drop("Class", axis = 1)
#print((df.Class == 1).describe())
print(df[df.Class==0])
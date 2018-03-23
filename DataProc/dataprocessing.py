# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 11:27:38 2018

@author: Mikkel
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv

def read_csv(fname):
   prefix = ""
   with open(prefix+fname, 'rt') as f:
       reader = csv.reader(f, delimiter = ",")
       data = []
       for row in reader:
           row2 = []
           for a in row:
               if(a.isdigit()):
                   row2.append(float(a))
               else:
                   row2.append(a)
           data.append(row2)
       return np.array(data)

data = read_csv("exper1.csv")



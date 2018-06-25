# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 14:47:43 2018

@author: jacki
"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import mglearn

X, y = mglearn.datasets.make_forge()

print (X)
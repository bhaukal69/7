import numpy as np
from urllib.request import urlopen
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data'
names = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n']

heartDisease = pd.read_csv(urlopen(URL), names = names) 
print(heartDisease.head())

del heartDisease['k']
del heartDisease['l']
del heartDisease['m']
del heartDisease['j']


heartDisease = heartDisease.replace('?', np.nan)
model = BayesianModel([('a','b'),('c','d'),('e','f'),('g','h'),('i','n')])

model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)
model.get_independencies()
HeartDisease_infer = VariableElimination(model)

q = HeartDisease_infer.query(variables=['n'], evidence={'a': 28})
print(q)
q = HeartDisease_infer.query(variables=['n'], evidence={'a': 50})
print(q)

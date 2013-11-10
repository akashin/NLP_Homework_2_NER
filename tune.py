#!/usr/bin/env python

from subprocess import Popen, PIPE

def runEval(params):
  command = ['./eval.sh'] + params
  #print(command)
  process = Popen(command, stdout=PIPE, stderr=PIPE)

  output, error = process.communicate()
  #print(error)
  #print(output)

  outputString = output.decode(encoding='UTF-8')
  outputString = outputString.split('\n')
  line = outputString[-6].split(' ')
  f1 = float(line[-1]) / 100
  return f1

import numpy as np

params_grid = {
    '--cutoff': np.arange(1, 10, 2),
    '--iterations': np.arange(50, 201, 50),
    '--gaussian': np.arange(0.0, 1.1, 0.2),
    '--technique': ['gis', 'lbfgs']
}

paramsToExplore = []

def genParam(currentKeyNumber, currentParams, params_grid, keys):
  if (currentKeyNumber == len(keys)):
    global paramsToExplore
    paramsToExplore += [dict(currentParams)]
    return

  currentKey = list(keys)[currentKeyNumber]
  for param in params_grid[currentKey]:
    currentParams[currentKey] = param
    genParam(currentKeyNumber + 1, currentParams, params_grid, keys)

def genParams(params_grid):
  keys = params_grid.keys()

  params = genParam(0, {}, params_grid, keys)
  return params

genParams(params_grid)

bestF1 = 0.0
bestParams = None

print(len(paramsToExplore))
for n, params in enumerate(paramsToExplore):
  print(float(n) / len(paramsToExplore) * 100, '%')
  paramsList = []
  for key in params.keys():
    paramsList += [key + '=' + str(params[key])]

  f1 = runEval(paramsList)
  print("Run results: F1: {0}, params: {1}".format(f1, paramsList))
  if (f1 > bestF1):
    print("Found better: F1: {0}, params: {1}".format(f1, paramsList))
    bestF1 = f1
    bestParams = params

print(bestF1)
print(bestParams)
#print(runEval(["-c 3", "-i 200", "-g 1.0"]))

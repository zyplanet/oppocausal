#!/usr/local/bin/python

from pycausal import search as s
from pycausal.pycausal import pycausal as pc
import numpy as np
from sklearn import preprocessing
import os
import pandas as pd
import pydot
import sys
os.environ["PATH"] += os.pathsep + 'D:/lib/Graphviz/bin/'


df = pd.read_excel(r"data\10w-directedRules-10%Noise-Binary.xlsx", encoding='gb18030')
keys = df.keys()
print(keys)
key_list = list(keys)
# continuous_keys = ['lagged_limit_cnt', 'lagged_little_limit_cnt', 'lagged_big_limit_cnt',
#                    'lagged_supper_limit_cnt', 'lagging_limit_cnt',
#                    'lagging_little_limit_cnt', 'lagging_big_limit_cnt',
#                    'lagging_supper_limit_cnt']
continuous_keys = []
unique_value = []
for key in keys:
    if key not in continuous_keys:
        result = df[key].unique()
        unique_value.append(len(result))
# invalid_col = []
# binary_col = []
# for idx, key in enumerate(keys):
#     if unique_value[idx] == 1:
#         invalid_col.append((idx, key))
#     elif unique_value[idx] == 2:
#         binary_col.append((idx, key))
# print(unique_value)
array = df.values
# print(array.shape)
# print(array[0])
new_array = -1*np.ones(array.shape)
for idx, key in enumerate(keys):
    if key not in continuous_keys:
        value_list = []
        for row, value in enumerate(array[:, idx]):
            if value not in value_list:
                if type(value) != type("str"):
                    if np.isnan(value):
                        new_array[row, idx] = -1
                    else:
                        value_list.append(value)
                        new_array[row, idx] = str(len(value_list))
                else:
                    value_list.append(value)
                    new_array[row, idx] = str(len(value_list))
            else:
                new_array[row, idx] = str(value_list.index(value))
for key in continuous_keys:
    idx = key_list.index(key)
    for row, value in enumerate(array[:, idx]):
        if np.isnan(value):
            new_array[row, idx] = -1
        else:
            new_array[row, idx] = float(value)
# for idx, key in binary_col:
#     new_array[:, idx] = new_array[:, idx]+1
# col_idx = [x[0] for x in invalid_col]
# col_name = [x[1] for x in invalid_col]
# valid_array = np.delete(new_array, col_idx, axis=1)
# valid_key = [x for x in keys if x not in col_name]

df = pd.DataFrame(new_array, columns=keys)

pc = pc()
pc.start_vm(java_max_heap_size='4000M')


# from pycausal import prior as p
# forbid = [['history_noise','class'],['history_fluctuating','class']]
# tempForbid = p.ForbiddenWithin(
#     ['class','history_fluctuating','history_noise'])
# temporal = [tempForbid]
# prior = p.knowledge(forbiddirect = forbid, addtemporal = temporal)

tetrad = s.tetradrunner()
# tetrad.getAlgorithmParameters(algoId = 'rfci', testId = 'chi-square-test')
# tetrad.getAlgorithmParameters(
#     algoId='gfci', testId='disc-bic-test', scoreId='bdeu-score')
# tetrad.getAlgorithmParameters(algoId='rfci', testId='cg-lr-test')
tetrad.getAlgorithmParameters(algoId='fges', scoreId='cg-bic-score')
# tetrad.getAlgorithmParameters(
#     algoId='gfci', testId='cg-lr-test', scoreId='cg-bic-score')
# tetrad.run(algoId='gfci', dfs=df, testId='cg-lr-test', scoreId='cg-bic-score',
#            dataType='mixed', numCategoriesToDiscretize=4,
#            maxDegree=3, maxPathLength=-1,
#            completeRuleSetUsed=False, faithfulnessAssumed=True, verbose=True,
#            numberResampling=5, resamplingEnsemble=1, addOriginalDataset=True)
# tetrad.run(algoId='rfci', dfs=df, testId='cg-lr-test',
#            dataType='mixed', numCategoriesToDiscretize=4,
#            depth=-1, maxPathLength=-1,
#            discretize=False, completeRuleSetUsed=False, verbose=True,
#            numberResampling=5, resamplingEnsemble=1, addOriginalDataset=True)
# tetrad.run(algoId='gfci', dfs=df, testId='disc-bic-test', scoreId='bdeu-score', dataType='discrete',
#            maxDegree=3, maxPathLength=-1,
#            completeRuleSetUsed=False, faithfulnessAssumed=True, verbose=True,
#            numberResampling=5, resamplingEnsemble=1, addOriginalDataset=True)
# tetrad.run(algoId = 'rfci', dfs = df, testId = 'chi-square-test', dataType = 'discrete',
#            depth = 3, maxPathLength = -1,
#            completeRuleSetUsed = True, verbose = True)
tetrad.run(algoId='fges', dfs=df, scoreId='cg-bic-score',
           dataType='mixed', numCategoriesToDiscretize=4,
           maxDegree=3, faithfulnessAssumed=True, verbose=True,
           numberResampling=5, resamplingEnsemble=1, addOriginalDataset=True)
print(tetrad.getNodes())
print(tetrad.getEdges())

graph = tetrad.getTetradGraph()
print('Graph BIC: {}'.format(graph.getAttribute('BIC')))
nodes = graph.getNodes()
for i in range(nodes.size()):
    node = nodes.get(i)
    print('Node {} BIC: {}'.format(node.getName(), node.getAttribute('BIC')))

dot_str = pc.tetradGraphToDot(graph)
graphs = pydot.graph_from_dot_data(dot_str)
graphs[0].write_svg('oppo-fges-real.svg')

pc.stop_vm()

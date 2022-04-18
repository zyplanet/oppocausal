#!/usr/local/bin/python

import numpy as np
from sklearn import preprocessing
import os
import pandas as pd
import pydot
import sys
os.environ["PATH"] += os.pathsep + 'D:/lib/Graphviz/bin/'


df = pd.read_excel(r"data\20w-directedRules-10%Noise.xlsx")
keys = df.keys()
unique_value = []
for key in keys:
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
    value_list = []
    for row, value in enumerate(array[:, idx]):
        if value not in value_list:
            if type(value) != type("str"):
                if np.isnan(value):
                    new_array[row, idx] = -1
                else:
                    value_list.append(value)
                    new_array[row, idx] = len(value_list)
            else:
                value_list.append(value)
                new_array[row, idx] = len(value_list)
        else:
            new_array[row, idx] = value_list.index(value)

# for idx, key in binary_col:
#     new_array[:, idx] = new_array[:, idx]+1
# col_idx = [x[0] for x in invalid_col]
# col_name = [x[1] for x in invalid_col]
# valid_array = np.delete(new_array, col_idx, axis=1)
# valid_key = [x for x in keys if x not in col_name]

df = pd.DataFrame(new_array, columns=keys)

from pycausal.pycausal import pycausal as pc
pc = pc()
pc.start_vm(java_max_heap_size = '4000M')

from pycausal import search as s
tetrad = s.tetradrunner()
tetrad.run(algoId = 'fges', dfs = df, scoreId = 'cg-bic-score', dataType = 'discrete',
           maxDegree = 3, faithfulnessAssumed = True,
           symmetricFirstStep = True, verbose = True)

print(tetrad.getNodes())
print(tetrad.getEdges())

graph = tetrad.getTetradGraph()
print('Graph BIC: {}'.format(graph.getAttribute('BIC')))
nodes = graph.getNodes()
for i in range(nodes.size()):
    node = nodes.get(i)
    print('Node {} BIC: {}'.format(node.getName(),node.getAttribute('BIC')))

dot_str = pc.tetradGraphToDot(graph)
graphs = pydot.graph_from_dot_data(dot_str)
graphs[0].write_svg('oppo-discrete2.svg')

pc.stop_vm()

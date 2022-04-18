#!/usr/local/bin/python

import os
import pandas as pd
import pydot
os.environ["PATH"] += os.pathsep + 'D:/lib/Graphviz/bin/'
data_dir = os.path.join(os.getcwd(), 'data', 'audiology.txt')
df = pd.read_table(data_dir, sep="\t")

from pycausal.pycausal import pycausal as pc
pc = pc()
pc.start_vm(java_max_heap_size = '100M')

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
graphs[0].write_svg('fges-discrete.svg')

pc.stop_vm()
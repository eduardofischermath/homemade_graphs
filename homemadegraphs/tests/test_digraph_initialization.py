########################################################################
# DOCUMENTATION / README
########################################################################

# File belonging to package "homemade_graphs"
# Implements classes and algorithms related to graphs and digraphs.

# For more information on functionality, see README.md
# For more information on bugs and planned features, see ISSUES.md
# For more information on the versioning, see RELEASES.md

# Copyright (C) 2021 Eduardo Fischer

# This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU Affero General Public License version 3
#as published by the Free Software Foundation. See LICENSE.
# Alternatively, see https://www.gnu.org/licenses/.

# This program is distributed in the hope that it will be useful,
#but without any warranty; without even the implied warranty of
#merchantability or fitness for a particular purpose.

########################################################################
# External imports
########################################################################

from unittest import TestCase as unittest_TestCase
from unittest import main as unittest_main

########################################################################
# Internal imports
########################################################################

from homemadegraphs.graphs_and_digraphs import Digraph

########################################################################
# Tests
########################################################################

class TestDigraphInitialization(unittest_TestCase):
  '''
  Tests Digraph.__init__ by trying it on many examples, as well as using
  different data inputs (controlled by data_type argument on __init__)
  
  All examples are the same Digraph corresponding to the digraph
  
  A ---> B <--- C.
  
  It will be initialized with every possible Digrah__init__ data_type option
  (except the ones involving edges as these require a Graph).
  No information on weights will be given.
  '''
  
  # Dict to be used in many methods within this class
  @staticmethod
  def recipes_for_data_and_data_types():
    A, B, C = 'A', 'B', 'C'
    AB, CB = ('A', 'B'), ('C', 'B')
    return {
        'all_arrows': [AB, CB],
        'some_vertices_and_all_arrows': ([A], [AB, CB]),
        'all_vertices_and_all_arrows': ([A, B, C], [AB, CB]),
        'arrows_out_as_dict': {A: [AB], B: [], C:[CB]},
        'arrows_out_as_list': [[A, AB], [B], [C, CB]],
        'neighbors_out_as_dict': {A:[B], B:[], C:[B]},
        'neighbors_out_as_list': [[A, B], [B], [C, B]]}
  
  @classmethod
  def setUpClass(cls):
    '''
    Initializes one digraph by multiple methods.
    '''
    data_and_data_types = cls.recipes_for_data_and_data_types()
    # We make another dict, indexed by data_type
    cls.dict_of_digraphs = {}
    for key in data_and_data_types:
      data_type = key
      data = data_and_data_types[key]
      cls.dict_of_digraphs[data_type] = Digraph(data = data, data_type = data_type)
      print(f'Graph successfully formed with {data_type=}')
    
  def test_pairwise_equality(self):
    count = 0
    for digraph_1 in self.__class__.dict_of_digraphs.values():
      for digraph_2 in self.__class__.dict_of_digraphs.values():
        count += 1
        assertEqual(digraph_1, digraph_2)
    # We also verify this testing tests all (total_digraphs)**2 pairs
    total_digraphs = len(self.__class__dict_of_digraphs)
    assertEqual(count, total_digraphs**2)

########################################################################
# Commands to be run on execution
########################################################################

if __name__ == '__main__':
  unittest_main()

########################################################################

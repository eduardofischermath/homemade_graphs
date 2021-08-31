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

from unittest import main as unittest_main
from unittest import skip as unittest_skip
from unittest import TestCase as unittest_TestCase

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

  def test_initialization(self, deactivate_assertions = False):
    '''
    Initializes one digraph by all multiple methods.
    '''
    dict_of_digraphs = {}
    data_and_data_types = self.recipes_for_data_and_data_types()
    # We use subTest to discriminate what we are doing
    # It accepts any keyword parameters for parametrization
    for key in data_and_data_types:
      with self.subTest(data_type = key):
        data_type = key
        data = data_and_data_types[key]
        dict_of_digraphs[data_type] = Digraph(data = data, data_type = data_type)
        if not deactivate_assertions:
          # We want to test this only when called directly by unittest
          self.assertIsInstance(dict_of_digraphs[data_type], Digraph)
    return dict_of_digraphs
  
  @unittest_skip
  def test_pairwise_equality(self):
    # Creating all instances, using the other method for better separation
    dict_of_digraphs = self.test_initialization(deactivate_assertions = True)
    count = 0
    for key_1 in dict_of_digraphs:
      for key_2 in dict_of_digraphs:
        with self.subTest(data_types = (key_1, key_2)):
          digraph_1 = dict_of_digraphs[key_1]
          digraph_2 = dict_of_digraphs[key_2]
          count += 1
          self.assertEqual(digraph_1, digraph_2)
    # We also verify this testing tests all (total_digraphs)**2 pairs
    total_digraphs = len(self.dict_of_digraphs)
    self.assertEqual(count, total_digraphs**2)

########################################################################
# Commands to be run on execution
########################################################################

if __name__ == '__main__':
  unittest_main()

########################################################################

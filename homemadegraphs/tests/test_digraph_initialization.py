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
from homemadegraphs.tests.generic_testing_classes import GenericInitializationTestCase

########################################################################
# Tests
########################################################################

class TestDigraphInitialization(GenericInitializationTestCase):
  '''
  Tests Digraph.__init__ by trying it on many examples, as well as using
  different data inputs (controlled by data_type argument on __init__)
  
  All examples are the same Digraph corresponding to the digraph
  
  A ---> B <--- C.
  
  It will be initialized with every possible Digrah__init__ data_type option
  (except the ones involving edges as these require a Graph).
  No information on weights will be given.
  '''
  
  class_being_tested = Digraph
  
  # Dict to be used in many methods within this class
  @staticmethod
  def recipes_for_data_and_data_types():
    A, B, C = 'A', 'B', 'C'
    AB, CB = ('A', 'B'), ('C', 'B')
    return {
        'all_arrows': [AB, CB],
        'some_vertices_and_all_arrows': ([A], [AB, CB]),
        'all_vertices_and_all_arrows': ([A, B, C], [AB, CB]),
        'full_arrows_out_as_dict': {A: [AB], B: [], C:[CB]},
        'arrows_out_as_dict': {A: [AB], C:[CB]},
        'full_arrows_out_as_list': [[A, AB], [B], [C, CB]],
        'arrows_out_as_list': [[A, AB], [C, CB]],
        'full_neighbors_out_as_dict': {A:[B], B:[], C:[B]},
        'neighbors_out_as_dict': {A:[B], C:[B]},
        'full_neighbors_out_as_list': [[A, B], [B], [C, B]],
        'neighbors_out_as_list': [[A, B], [C, B]]}

########################################################################
# Commands to be run on execution
########################################################################

if __name__ == '__main__':
  unittest_main()

########################################################################

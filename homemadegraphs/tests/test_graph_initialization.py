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

from homemadegraphs.graphs_and_digraphs import Graph
from homemadegraphs.tests.generic_testing_classes import GenericInitializationTestCase

########################################################################
# Tests
########################################################################

class TestGraphInitialization(GenericInitializationTestCase):
  '''
  Tests Graph.__init__ by trying it on many examples, as well as using
  different data inputs (controlled by data_type argument on __init__)
  
  All examples are the same Graph corresponding to the 3-vertex one-edge graph:
  
  A      B ---- C.
  
  It will be initialized with every possible Digraph.__init__ (called from Graph
  due to subclassing) data_type option.
  No information on weights will be given.
  '''
  
  class_being_tested = Graph
  
  @classmethod
  def property_specifications(cls):
    return [
        cls.PropertySpecification('get_number_of_vertices',
        3,
        True,
        tuple()),
        cls.PropertySpecification('get_number_of_arrows',
        2,
        True,
        tuple()),
        cls.PropertySpecification('get_number_of_edges',
        1,
        True,
        tuple())]
  
  # Dict to be used in many methods within this class
  @staticmethod
  def recipes_for_data_and_data_types():
    A, B, C = 'A', 'B', 'C'
    BC, CB = ('B', 'C'), ('C', 'B') # Can be turned into Arrows or Edges
    return {
        'some_vertices_and_all_arrows': ([A], [BC, CB]),
        'all_vertices_and_all_arrows': ([A, B, C], [BC, CB]),
        'some_vertices_and_all_edges': ([A], [BC]),
        'all_vertices_and_all_edges': ([A, B, C], [BC]),
        'full_arrows_out_as_dict': {A: [], B: [BC], C: [CB]},
        'arrows_out_as_dict': {A: [], B: [BC], C: [CB]},
        'full_arrows_out_as_list': [[A], [B, BC], [C, CB]],
        'arrows_out_as_list': [[A], [B, BC], [C, CB]],
        'full_edges_out_as_dict': {A:[], B:[], C:[CB]},
        'edges_out_as_dict': {A:[], C:[CB]},
        'full_edges_out_as_list': [[A], [B], [C, CB]],
        'edges_out_as_list': [[A], [C, CB]],
        'full_neighbors_out_as_dict': {A:[], B:[C], C:[B]},
        'neighbors_out_as_dict': {A:[], B:[C], C:[B]},
        'full_neighbors_out_as_list': [[A], [B, C], [C, B]],
        'neighbors_out_as_list': [[A], [B, C], [C, B]],
        'full_neighbors_as_dict': {A:[], B:[], C:[B]},
        'neighbors_as_dict': {A:[], C:[B]},
        'full_neighbors_as_list': [[A], [B], [C, B]],
        'neighbors_as_list': [[A], [C, B]]}

########################################################################
# Commands to be run on execution
########################################################################

if __name__ == '__main__':
  unittest_main()

########################################################################

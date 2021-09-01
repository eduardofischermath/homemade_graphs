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

class TestEmptyDigraph(unittest_TestCase):
  '''
  Tests all (di)graph methods on the empty (di)graph, ensuring the output is
  correct (or, if non-canonically defined, that is follows the specified convention).
  '''

  # Dict to be used in many methods within this class
  @staticmethod
  def recipes_for_data_and_data_types():
    return {
        'all_arrows': [],
        'some_vertices_and_all_arrows': ([], []),
        'all_vertices_and_all_arrows': ([], []),
        'full_arrows_out_as_dict': {},
        'arrows_out_as_dict': {},
        'full_arrows_out_as_list': [],
        'arrows_out_as_list': [],
        'full_neighbors_out_as_dict': {},
        'neighbors_out_as_dict': {},
        'full_neighbors_out_as_list': [],
        'neighbors_out_as_list': []}
  
  @classmethod
  def setUpClass(cls):
    '''
    Initializes the empty digraph with no vertices.
    '''
    # Note this dies with the class, so there is no need for tearDownClass
    # Also note this is a variable class, not an instance attribute
    # To access it from an instance, use self.__class__ 
    cls.empty_digraph = Digraph(
        data = ([], []), data_type = 'all_vertices_and_all_arrows')
        
  def test_number_of_vertices(self):
    self.assertEqual(self.__class__.empty_digraph.get_number_of_vertices(), 0)
    
  def test_number_of_arrows(self):
    self.assertEqual(self.__class__.empty_digraph.get_number_of_arrows(), 0)

########################################################################
# Commands to be run on execution
########################################################################

if __name__ == '__main__':
  unittest_main()

########################################################################

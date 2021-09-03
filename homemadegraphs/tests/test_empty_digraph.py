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
from homemadegraphs.tests.generic_testing_classes import GenericInitializationTestCase

########################################################################
# Tests
########################################################################

class TestEmptyDigraph(GenericInitializationTestCase):
  '''
  Tests all (di)graph methods on the empty (di)graph, ensuring the output is
  correct (or, if non-canonically defined, that is follows the specified convention).
  '''
  
  class_being_tested = Digraph
  
  intended_instance_properties = {
      'number_of_vertices': 0,
      'number_of_arrows': 0}

  # Dict to be used in many methods within this class
  @staticmethod
  def recipes_for_data_and_data_types():
    return {
        'all_arrows': [],
        'some_vertices_and_all_arrows': ([], []),
        'all_vertices_and_all_arrows': ([], []),
        'all_edges': [],
        'some_vertices_and_all_edges': ([], []),
        'all_vertices_and_all_edges': ([], []),
        'full_arrows_out_as_dict': {},
        'arrows_out_as_dict': {},
        'full_arrows_out_as_list': [],
        'arrows_out_as_list': [],
        'full_edges_out_as_dict': {},
        'edges_out_as_dict': {},
        'full_edges_out_as_list': [],
        'edges_out_as_list': [],
        'full_neighbors_out_as_dict': {},
        'neighbors_out_as_dict': {},
        'full_neighbors_out_as_list': [],
        'neighbors_out_as_list': [],
        'full_neighbors_as_dict': {},
        'neighbors_as_dict': {},
        'full_neighbors_as_list': [],
        'neighbors_as_list': []}

########################################################################
# Commands to be run on execution
########################################################################

if __name__ == '__main__':
  unittest_main()

########################################################################

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
#from unittest import skip as unittest_skip
#from unittest import TestCase as unittest_TestCase

########################################################################
# Internal imports
########################################################################

from homemadegraphs.graphs_and_digraphs import WeightedDigraph
from homemadegraphs.tests.generic_testing_classes import GenericPropertyTestCase

########################################################################
# Tests
########################################################################

class TestWeightedDigraphMethods(GenericPropertyTestCase):
  '''
  Tests all methods for WeightedDigraph on a specific instance.
  '''
  
  def get_object_for_testing(self):
    '''
    Produces the WeightedDigraph
    
    A <---> B <---> C
    |       |       ^
    |       |       |
    V       V       |
    D ----> E ----> F
     \             7 
      \___________/
       
    where each arrow has a specified weight.
    '''
    A, B, C, D, E, F = 'A', 'B', 'C', 'D', 'E', 'F'
    AB = (A, B, 10)
    BA = (B, A, 10)
    BC = (B, C, 8)
    CB = (C, B, 8)
    DE = (D, E, 15)
    EF = (E, F, 5)
    AD = (A, D, 16)
    BE = (B, E, 9)
    FC = (F, C, 3)
    DF = (D, F, 17)
    return WeightedDigraph(
        data = (
            [A, B, C, D, E, F],
            [AB, BA, BC, CB, DE, EF, AD, BE, FC, DF]),
        data_type = 'all_vertices_and_all_arrows')

  @classmethod
  def property_specifications(cls):
    return [
        cls.PropertySpecification('get_number_of_vertices',
        6,
        True,
        tuple(),
        {}),
        cls.PropertySpecification('get_number_of_arrows',
        10,
        True,
        tuple(),
        {}),
        cls.PropertySpecification('solve_traveling_salesman_problem',
        57,
        True,
        tuple(),
        {'compute_path_instead_of_cycle': False,
            'initial_vertex': 'A',
            'final_vertex': 'A',
            'use_memoization_instead_of_tabulation': True,
            'output_as': 'length',
            'skip_checks': False}),
        cls.PropertySpecification('solve_traveling_salesman_problem',
        47,
        True,
        tuple(),
        {'compute_path_instead_of_cycle': True,
            'initial_vertex': 'A',
            'final_vertex': 'B',
            'use_memoization_instead_of_tabulation': False,
            'output_as': 'length',
            'skip_checks': False})]

########################################################################
# Commands to be run on execution
########################################################################

if __name__ == '__main__':
  unittest_main()

########################################################################

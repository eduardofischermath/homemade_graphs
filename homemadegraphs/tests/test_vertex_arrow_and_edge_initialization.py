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

from homemadegraphs.vertices_arrows_and_edges import Vertex, Arrow, Edge, OperationsVAE

########################################################################
# Tests
########################################################################

class TestVertexArrowEdgeInitialization(unittest_TestCase):
  '''
  Tests the namedtuples Vertex, Arrow and Edge.
  '''
  # Note this is not derived from any Generic test case
  # This test case is closest to GenericPropertyTestCase, but since it
  #has multiple objects (at least in this formulation) this testcase
  #will not conform to those Generic classes
  
  @staticmethod
  def recipes_for_initialization():
    '''
    Provides recipes for formation of the namedtuples.
    '''
    data = {
        'vertex': (
            Vertex,
            ('String for vertex',),
            ('name',),
            1),
        'arrow': (
            Arrow,
            (400, 'String for arrow', 0.35),
            ('source', 'target', 'weight'),
            3),
        'edge': (
            Edge,
            (400, 'String for edge', 0.35),
            ('first', 'second', 'weight'),
            3)}
    return data

  def test_namedtuple_initialization(self):
    '''
    Tests the correct initialization of namedtuples, as well as their
    named attributes.
    '''
    data = self.recipes_for_initialization()
    for namedtuple_name in data:
      with self.subTest(namedtuple_name = namedtuple_name):
        recipe = data[namedtuple_name]
        init_class, init_arguments, expected_attributes, expected_length = recipe
        obj = init_class(*init_arguments)
        for attribute in expected_attributes:
          self.assertTrue(hasattr(obj, attribute))
        self.assertEqual(len(obj), expected_length)
    
########################################################################
# Commands to be run on execution
########################################################################

if __name__ == '__main__':
  unittest_main()

########################################################################


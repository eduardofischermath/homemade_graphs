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
  
  """
  def test_initialization(self):
    '''
    Tests initialization of Vertex, Arrow and Edge.
    '''
    data = self.recipes_for_initialization()
    for key in data:
      with self.subTest(namedtuple_name = key):
        class_name, init_values, fields_names, length = data[key]
        instance = class_name(*init_values)
        self.assertIsInstance(instance, class_name)
        for field_name in fields_names:
          self.assertTrue(hasattr(instance, field_name))
        self.assertTrue(hasattr(instance, '__len__'))
        self.assertEqual(len(instance), length)

  def test_weight_default(self):
    '''
    Tests whether default weight of None (to unweighted Arrows and Edges) works.
    '''
    explicitly_unweighted_arrow = Arrow('A', 1, None)
    implicitly_unweighted_arrow = Arrow('A', 1)
    self.assertEqual(explicitly_unweighted_arrow, implicitly_unweighted_arrow)
    explicitly_unweighted_edge = Edge('A', 1, None)
    implicitly_unweighted_edge = Edge('A', 1)
    self.assertEqual(explicitly_unweighted_edge, implicitly_unweighted_edge)"""

########################################################################
# Commands to be run on execution
########################################################################

if __name__ == '__main__':
  unittest_main()

########################################################################


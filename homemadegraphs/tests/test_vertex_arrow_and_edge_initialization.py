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

class TestNamedtupleInitialization(unittest_TestCase):
  '''
  Tests the namedtuples Vertex, Arrow and Edge.
  '''

  def test_vertex_initialization(self):
    vertex = Vertex('Demonstration string')
    self.assertIsInstance(vertex, Vertex)
    self.assertTrue(hasattr(vertex, 'name'))
    self.assertTrue(hasattr(vertex, '__len__'))
    self.assertEqual(len(vertex), 1)
    
  def test_arrow_initialization(self):
    arrow = Arrow('A string', 200, 0.236)
    self.assertIsInstance(arrow, Arrow)
    self.assertTrue(hasattr(arrow, 'source'))
    self.assertTrue(hasattr(arrow, 'target'))
    self.assertTrue(hasattr(arrow, 'weight'))
    self.assertTrue(hasattr(arrow, '__len__'))
    self.assertEqual(len(arrow), 3)
  
  def test_edge_initialization(self):
    edge = Edge('A string', 200, 0.236)
    self.assertIsInstance(edge, Edge)
    self.assertTrue(hasattr(edge, 'first'))
    self.assertTrue(hasattr(edge, 'second'))
    self.assertTrue(hasattr(edge, 'weight'))
    self.assertTrue(hasattr(edge, '__len__'))
    self.assertEqual(len(edge), 3)

  def test_weight_default(self):
    explicitly_unweighted_arrow = Arrow('A', 1, None)
    implicitly_unweighted_arrow = Arrow('A', 1)
    self.assertEqual(explicitly_unweighted_arrow, implicitly_unweighted_arrow)
    explicitly_unweighted_edge = Edge('A', 1, None)
    implicitly_unweighted_edge = Edge('A', 1)
    self.assertEqual(explicitly_unweighted_edge, implicitly_unweighted_edge)

########################################################################
# Commands to be run on execution
########################################################################

if __name__ == '__main__':
  unittest_main()

########################################################################


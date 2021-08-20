
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

# This code more or less assumes homemade_graphs package is imported
#to the relevant context, and doesn't try to import it here

########################################################################
# Tests
########################################################################

class TestDigraphInitialization(unittest_TestCase):
  '''
  Tests Digraph.__init__ by trying it on many examples, as well as using
  different data inputs (controlled by data_type argument on __init__)
  '''
  
  def test_initialize_empty_digraph(self):
    '''
    Tries to initializes the digraph with no vertices.
    '''
    empty_digraph = homemade_graphs.Digraph(
        data = ([], []), data_type = 'all_vertices_and_all_arrows')
    self.assertEqual(empty_digraph.get_number_of_vertices, 0)
    self.assertEqual(empty_digraph.get_number_of_arrows, 0)

########################################################################
# Commands to be run on execution
########################################################################

if __name__ == '__main__':
  unittest_main()

########################################################################

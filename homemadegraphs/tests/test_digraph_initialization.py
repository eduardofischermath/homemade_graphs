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

from ..graphs_and_digraphs import Digraph

########################################################################
# Tests
########################################################################

class TestDigraphInitialization(unittest_TestCase):
  '''
  Tests Digraph.__init__ by trying it on many examples, as well as using
  different data inputs (controlled by data_type argument on __init__)
  '''
  
  pass
  # Initiaze many digraphs/graphs with many data-type inputs

########################################################################
# Commands to be run on execution
########################################################################

if __name__ == '__main__':
  unittest_main()

########################################################################

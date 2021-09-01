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

########################################################################
# Internal imports
########################################################################



########################################################################
# Generic test procedures
########################################################################

class GenericInitializationTestCase(unittest_TestCase):
  '''
  Provides some generic and enhanced version of unittest.TestCase by
  providing a few generic class methods related to initialization.
  '''
  
  @classmethod
  def test_initialization(cls):
    pass
  
  @classmethod
  def test_pairwise_equality(cls):
    pass
    
  @classmethod
  def test_equality_against_specific(cls):
    pass

########################################################################

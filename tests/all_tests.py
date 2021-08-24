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

from unittest import TestLoader as unittest_TestLoader
from unittest import TestSuite as unittest_TestSuite

########################################################################
# 
########################################################################

# File to run all possible tests at once
# Possibility: use "python -m unittest discover" on the folder/subpackage
# Possibilities:
#i) trigger it during import,
#ii) trigger only on execution as script,
#iii) define run_all_tests to be run only at the user's discretion

def run_all_tests():
  '''
  Run all tests.
  '''
  # Creates a TestSuite using unittest_TestLoader.discover
  test_suite = unittest_TestLoader.discover(start_dir = '/../')

########################################################################
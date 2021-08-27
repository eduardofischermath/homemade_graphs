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
from unittest import TextTestRunner as unittest_TextTestRunner

########################################################################
# 
########################################################################

# File to run all possible tests at once
# Possibility: use "python -m unittest discover" on the folder/subpackage
# Possibilities:
# i) trigger it during import,
# ii) trigger only on execution as script,
# iii) define run_all_tests to be run only at the user's discretion
# iv) write a load_tests function on __init__ of subpackage tests
#(it would take priority within unittest.TestLoader.discover())

def discover_and_run_all_tests(start_dir = './tests', pattern = 'test*.py', top_level_dir = '.', verbosity = 2):
  '''
  Discover and runs all tests.
  
  Ideally, must be run at the main/top folder of the project. To run from somewhere else,
  or to change verbosity, change the arguments.
  '''
  # Creates a TestSuite using unittest_TestLoader.discover
  # The tests don't need to be imported: unittest_TestLoader.discover does it
  # The needed code from homemadegraphs is imported on the tests themselves
  loader = unittest_TestLoader()
  test_suite = loader.discover(start_dir = start_dir, pattern = pattern)
  # Use the "standard" runner:
  # Verbosity 0 is summary, verbosity 1 is one dot for each pass and one F for each fail,
  #and verbosity 2 is a line with a brief summary of the test
  runner = unittest_TextTestRunner(verbosity = verbosity)
  result = runner.run(test_suite)
  return result
  
########################################################################
# Executable
########################################################################

if __name__ == '__main__':
  # Should be run at the main/top folder of the project, very likely called
  #homemadegraphs (the exact name of the package)
  result = discover_and_run_all_tests()
  print(result)

########################################################################

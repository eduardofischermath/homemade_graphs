######################################################################
# DOCUMENTATION / README
######################################################################

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
# Internal imports of subpackages/submodules
########################################################################

# Brings all the source code in the package into scope
from .src import *

# Bring the test suites into scope only if a special global variable exists
#and is set to True
if 'IS_TESTING_ENVIRONMENT' in globals():
  if IS_TESTING_ENVIRONMENT:
    from .tests import *

########################################################################



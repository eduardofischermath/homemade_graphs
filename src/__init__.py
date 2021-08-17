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







# Since cache from functools was introduced in Python version >= 3.9,
#we check for it. If not new enough, we go with lru_cache(maxsize = None)
#and bind the decorator to the name functools_cache
# Alternative is try/except, but comparing versions is also ok
from sys import version_info as sys_version_info
if sys_version_info >= (3, 9):
  from functools import cache as functools_cache
else:
  from functools import lru_cache as functools_lru_cache
  functools_cache = functools_lru_cache(maxsize=None)
  # In code always call it through functools_cache

########################################################################
# Internal imports of subpackages/submodules
########################################################################

# Bring all files to current scope (that is, folder becomes a subpackage)

from .algorithm_oriented_classes import *
from .paths_and_cycles import *
from .graphs_and_digraphs import *
from .vertices_arrows_and_edges import *

########################################################################

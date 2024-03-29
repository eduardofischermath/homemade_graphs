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



########################################################################
# Internal imports
########################################################################

# Bring all functions and classes into homemadegraphs.tests scope
#############
# This might need more testing!
#############

from . import all_tests
from . import generic_testing_classes
from . import test_digraph_initialization
from . import test_empty_digraph
from . import test_vertex_arrow_and_edge_initialization
from . import test_weighted_digraph_methods

########################################################################

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

# Brings all the source code [technically a subpackage] into package scope
# The order is fundamental due to dependencies
# The user needs to respect the hierarchy at every call
from . import vertices_arrows_and_edges
from . import paths_and_cycles
from . import graphs_and_digraphs
from . import algorithm_oriented_classes

# The tests do not belong to the package homemadegraphs natively
# They rather form a "separate/optional" subpackage
# We could import them simply by including "from . import tests" in this file
#or writing "import homemadegraphs.tests" in a Python session or in a script/module
#(after package "homemadegraphs" is installed or the current directory is right)

########################################################################

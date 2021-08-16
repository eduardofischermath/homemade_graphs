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

print('Testing')
try:
  print(Vertex(name = 'lalala'))
  print('first')
except:
  try:
    print(src.Vertex(name = 'lalala'))
    print('second')
  except:
    try:
      print(vertices_arrows_and_edges.Vertex(name = 'lalala'))
      print('third')
    except:
      try:
        print(src.vertices_arrows_and_edges.Vertex(name = 'lalala'))
        print('fourth')
      except:
        try:
          print(src.vertices_arrows_and_edges.Vertex(name = 'lalala'))
          print('fifth')
        except:
          print('Nothing works')
        
print(dir())


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
# External imports
########################################################################

from collections import namedtuple as collections_namedtuple
from itertools import zip_longest as itertools_zip_longest
from itertools import chain as itertools_chain
from itertools import product as itertools_product
from copy import copy as copy_copy
from random import choices as random_choices
from math import log2 as math_log2
from math import inf as math_inf
from heapq import heapify as heapq_heapify
from heapq import heappush as heapq_heappush
from heapq import heappop as heapq_heappop
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

# __init__ of package homemade_graphs
# Brings all the code in the package into scope

from .src import *
#from tests import *

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


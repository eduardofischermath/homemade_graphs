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

class UnionFind(object):
  '''
  Implements the Union-Find data structure.
  
  The union-find data structure manages partitions of a collection of objects.
  
  Two operations are expected: find, which means finding the set to which
  an object/element belongs, and union, which merges two sets.
  
  The sets of the partition (also called clusters in some contexts) are
  controlled by a "leader", which is one of the elements in the same cluster.
  Every objects has a leader. Two elements are in the same cluster if and only if
  they have the same leader.
  
  This information is controlled by a dictionary (whose keys are all objects)
  which points to the parents of the objects. A parent of an objects is another object.
  A leader will be its own parent. The leader of any object can be obtained
  by reading the parents (parent, then parent of parent, then parent of parent
  of parent, and so on) until arriving at a leader.
  
  [This parent operation logically cannot contain any cycles.]
  
  Attributes:
  
  parents
  '''
  
  def __init__(self, data, data_type = 'objects'):
    '''
    Magic method. Initializes the instance.
    
    If data_type is 'objects', data should be a list or iterable of objects.
    It creates the partition in which each object is its own cluster.
    
    If data_type is 'parents', then it starts with the partition in sets
    by data which is assumed to be the dictionary parents.
    '''
    # In future implement initialization via given clusters (pick random leaders)
    data_type = data_type.lower()
    if data_type == 'objects':
      # Each object is its own leader
      self.parents = {obj: obj for obj in data}
    elif data_type == 'parents':
      self.parents = data
    else:  
      raise ValueError('Invalid data type for initialization.')

  def __repr__(self):
    '''
    Magic method. Returns faithful representation of instance.
    '''
    raise NotImplementedError('In the works')
    
  def __str__(self):
    '''
    Magic method. Returns user-friendly representation of instance.
    '''
    raise NotImplementedError('In the works')
    
  def __len__(self):
    '''
    Magic method. Returns length of the instance.
    
    In this case, it is the total number of elements.
    '''
    return len(self.parents)
  
  def find_leader(self, obj):
    '''
    
    '''
    pass
    
  def union_from_objects(self, obj_1, obj_2):
    '''
    
    '''
    pass
  
  def union_from_sets(self, set_1, set_2):
    '''
    
    '''
    # Implementation optional
    raise NotImplementedError('In the works')

  def present_partition(self, exhibit_leaders = False):
    '''
    Returns the partition by listing its sets.
    '''
    pass

########################################################################

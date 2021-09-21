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
  
  There is a second dictionary which, for each object, provides the objects
  having it as leader [empty if object is not a leader].
  '''
  
  def __init__(self, data, data_type):
    '''
    Magic method. Initializes the instance.
    
    If data_type is 'objects', data should be a list or iterable of objects.
    It creates the partition in which each object is its own cluster.
    
    If data_type is 'parents', then it starts with the partition in sets
    by data which is assumed to be the dictionary parents.
    '''
    data_type = data_type.lower()
    if data_type == 'objects':
      # Each object is its own leader
      self.parents = {obj: obj for obj in data}
      self.led_by = {obj: obj for obj in data}
    elif data_type == 'parents':
      self.parents = data
      self.led_by = self._create_dictionary_of_led_by(self.parents)
    else:  
      raise ValueError('Invalid data type for initialization.')

  @abstractmethod
  def _create_dictionary_of_led_by(parents):
    '''
    Given a dictionary representing the parenting relationship, returns
    a dict whose keys are the elements and the values are all objects led
    by the key.
    
    To be used in initialization.
    '''
    # To simplify: first create keys, then on a second later populate with values
    led_by = {obj: [] for obj in parents}
    for obj in parents:
      led_by[parents[obj]].append(obj)
    return led_by

  def __repr__(self):
    '''
    Magic method. Returns faithful representation of instance.
    '''
    return self.present_partition(
        output_as = 'dict',
        output_clusters_as = 'list')

  def __str__(self):
    '''
    Magic method. Returns user-friendly representation of instance.
    '''
    return self.present_partition(
        output_as = 'list',
        output_clusters_as = 'list')
    
  def __len__(self):
    '''
    Magic method. Returns length of the instance.
    
    In this case, it is the total number of elements.
    '''
    return len(self.parents)
    
  def __contains__(self, obj):
    '''
    Magic methods. Determines membership in instance.
    '''
    # Checks if it is a key of self.parents
    return obj in self.parents

  def get_number_of_clusters(self):
    '''
    Returns current number of clusters of partition.
    '''
    # Count the number of leaders, which have self.led_by non-empty
    return sum(1 for obj in self.led_by if self.led_by[obj])

  def is_leader(self, obj, skip_checks = False):
    '''
    Returns whether an object is a leader of its own cluster.
    '''
    # An object is a leader if and only if it is its own parent
    if not skip_checks:
      assert obj in self, 'Object must be in union-find data structure'
    return self.parents[obj] == obj
  
  def find_leader(self, obj, also_do_path_compression = False, skip_checks = False):
    '''
    Finds the leader of any given object/element.
    
    Has option to also do "path compression" in the process; in the case,
    the object and all its consecutive parents will have their parent
    updated to be their leader.
    '''
    if not skip_checks:
      assert obj in self, 'Object must be in union-find data structure'
    # Create a path to leader by taking parents in sequence
    path_to_leader = []
    path_to_leader.append(obj)
    # We loop while the leader is not found
    while not self.is_leader(path_to_leader[-1]):
      # We don't have a leader yet, so we append the parent to the path
      path_to_leader.append(parents[path_to_leader[-1]])
      if not skip_checks:
        # To avoid a case of an infinite loop:
        if len(path_to_leader) > len(self):
          raise ValueError('Cycle found in parent relations in data structure.')
    # Ok, now we have a full path to the leader in path_to_leader
    #(which is the last element)
    leader_of_obj = path_to_leader[-1]
    # Optional path compression (otherwise parents is unaltered)
    # The advantage is that in the future the leader of those objects will
    #be found earlier, without a long path_to_leader
    if also_do_path_compression:
      for item in path_to_leader:
        # Note it is not needed to update self.led_by as the leaders don't change
        parents[item] = leader_of_obj   
    return leader_of_obj
    
  def union_from_objects(self, obj_1, obj_2, require_different_clusters = False,
      also_do_path_compression = False, skip_checks = False):
    '''
    Unites the clusters corresponding to two vertices.
    
    Returns nothing. The leader of the cluster of the first object becomes
    also the leader of the cluster of the second object.
    
    This is done by updating the parent of the leader (soon to be an ex-leader)
    of the second object to be the leader of first object. The information
    of the objects led by them is also updated.
    
    Has option to do path compression within the "find" operations.
    (Note it won't be a definitive path compression as the leaders are updated
    later in the process.)
    '''
    if not skip_checks:
      assert obj in self, 'Object must be in union-find data structure'
      assert obj in self, 'Object must be in union-find data structure'
    leader_obj_1 = self.find_leader(
        obj = obj_1,
        also_do_path_compression = also_do_path_compression,
        skip_checks = skip_checks)
    leader_obj_2 = self.find_leader(
        obj = obj_2,
        also_do_path_compression = also_do_path_compression,
        skip_checks = skip_checks)
    if leader_obj_1 != leader_obj_2:
      # Update self.parents and self.led_by
      self.parents[leader_obj_2] = leader_obj_1
      self.led_by[leader_obj_1] += self.led_by[leader_obj_2]
      self.led_by[leader_obj_2] = []
    else:
      if require_different_clusters:
        raise ValueError('Objects must belong to different clusters to execute union')
      else:
        # Nothing needs to be done
        pass

  def present_partition(self, output_as = 'dict', output_clusters_as = 'list'):
    '''
    Returns the partition by listing its sets/clusters.
    
    If output_as is 'list', returns a list of clusters.
    If output_as is 'dict', returns the dectionary whose keys are the leaders
    and whose values are the clusters they lead.
    
    If output_clusters_as is 'list', each cluster will be a list of its elements.
    If output_clusters_as is 'set', a set is formed from those elements.
    '''
    output_clusters_as = output_clusters_as.lower()
    output_as = output_as.lower()
    # All info available on led_by, it's only a matter of proper formatting
    if output_clusters_as == 'set':
      relevant_dict = {set(self.led_by[obj]) for obj in self.led_by}
    elif output_clusters_as == 'list':
      relevant_dict = self.led_by
    else:
      raise ValueError('Could not recognize option for output of each cluster')
    if output_as == 'dict':
      return relevant_dict
    elif output_as == 'list':
      return list(relevant_dict.values())
    else:
      raise ValueError('Could not recognize option for output of partitions')
    
########################################################################

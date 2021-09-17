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

from collections import Counter as collections_Counter
from copy import copy as copy_copy
from heapq import heapify as heapq_heapify
from heapq import heappop as heapq_heappop
from heapq import heappush as heapq_heappush
from math import inf as math_inf
from math import log2 as math_log2
from math import sqrt as math_sqrt
from random import choices as random_choices

########################################################################
# Internal imports
########################################################################

from homemadegraphs.algorithm_oriented_classes import StateGraphGetCC, StateDigraphGetSCC, StateDigraphSolveTSP
from homemadegraphs.paths_and_cycles import VertexPath, VertexCycle
from homemadegraphs.vertices_arrows_and_edges import Vertex, Arrow, Edge, OperationsVAE

########################################################################
# Declaration of Digraph class and initialization
########################################################################

class Digraph(object):
  '''
  Class which implements digraphs, or directed graphs. A digraph is a set
  of points (called vertices) and a set of arrows (ordered pairs of vertices)
  which may or may not be weighted.
  
  An undirected graph (also called simply graph) is also implemented as
  a diagraph by interpreting its edges (an unordered pair of vertices)
  as two arrows, back and forth within the pair.
  
  Attributes:
  _arrows
  _arrows_out
  _arrows_in
  
  (A Graph instance would also have _edges and _inciding_edges)
  '''

  def __init__(self, data, data_type, cast_as_class = None):
    '''
    Magic method. Initializes the instance.
    
    Data is input and parsed according to data_type, building a digraph object.
    
    The digraph might be weighted, or not, and directed, or not.
    [An directed graph is called Digraph. An undirected graph is simply called Graph.
    Similarly, there are subclasses depending on whether the digraph is weighted.]
    This method also allows for class recasting.
    
    Data specifications for each data_type (22 possibilities):
    
    'all_arrows': self explanatory, a list [or other iterable] of arrows.
    Depending on their characteristics (i. e. the length of important information)
    we will deduce weighted or unweighted. With this option it is impossible
    to create an instance with an isolated vertex
    
    'some_vertices_and_all_arrows': similar to before but a tuple starting with
    a list [or iterable] of vertices, and then the iterable for all arrows
    
    'all_vertices_and_all_arrows': similar but we insist all vertices
    are given; arrows are forced to incide on the known vertices
    
    'all_edges': similar to 'all_arrows' with each individual edge
    generating two arrows, back and forth. [Under this option, a pair of edges
    back and forth between two vertices would produce four arrows, two in each
    direction.]
    
    'some_vertices_and_all_edges': similar to 'some_vertices_and_all_arrows'
    
    'all_vertices_and_all_edges': similar to 'all_vertices_and_all_arrows'
    
    'full_arrows_out_as_dict': gives the arrows out of vertices, each vertex
    being the key of a dict whose value has the arrows out of it
    
    'arrows_out_as_dict': similar to 'full_arrows_out_as_dict' except
    that vertices which are not the sources of any arrows (i. e. corresponding
    values are empty lists) don't need to be in the keys of the dictionary
    
    'full_arrows_out_as_list': a list [or iterable] of lists [or iterables],
    and for each of them, the item at position 0 is the vertex and all
    other items correspond to the arrows out of it
    
    'arrows_out_as_list': similar to 'full_arrows_out_as_list' but vertices
    without nontrivial values might be omitted
    
    'full_edges_out_as_dict': similar to 'arrows_out_as_dict' but each tuple
    should be interpreted as an edge inciding on the pertinent vertex,
    giving also rise to two arrows, back and forth
    
    'edges_out_as_dict': similar to 'full_edges_out_as_dict' but vertices
    without nontrivial values might be omitted
    
    'full_edges_out_as_list': similar to 'arrows_out_as_list'
    
    'edges_out_as_list': similar to 'full_edges_out_as_list' but vertices
    without nontrivial values might be omitted
    
    'full_neighbors_out_as_dict': similar to 'arrows_as_as_dict' but each value
    is not the arrow but only the target of the arrow (and possibly the weight)
    
    'neighbors_out_as_dict': similar to 'full_neighbors_out_as_dict' but vertices
    with trivial values might be omitted
    
    'full_neighbors_out_as_list': similar to 'arrows_out_as_list'
    
    'neighbors_out_as_list': similar to 'full_neighbors_out_as_list' but vertices
    with trivial values might be omitted
    
    'full_neighbors_as_dict': similar to 'arrows_as_as_dict', providing neighbors
    and possibly weight, but creating edges instead of arrows
    
    'neighbors_as_dict': similar to 'full_neighbors_as_dict' but vertices
    with trivial values might be omitted
    
    'full_neighbors_as_list': similar to 'arrows_out_as_list'
    
    'neighbors_as_list': similar to 'full_neighbors_as_list' but vertices
    with trivial values might be omitted
    '''
    # We subdivide the work into methods
    # First we do the proper subclassing and reset the attributes
    # This is a procedural method which modifies self
    self._reclass_and_reset(cast_as_class = cast_as_class)
    # Extract and organize data according to data_type
    # This is a static method which returns values, does not alter self
    init_data = self._extract_and_organize_data(data = data, data_type = data_type)
    # Add vertices and arrows (also edges if Graph) to instance, data depending on
    #use_edges_instead_of_arrows, init_vertices, init_tuplees
    # This is a procedural method which modifies self
    self._construct_instance(init_data)
    
  def _reclass_and_reset(self, cast_as_class = None):
    '''
    Creates or resets the main attributes of the instance (_arrows, _arrows_in and _arrows_out,
    and _edges and _inciding_edges for Graphs only) to have empty values,
    and also might reclass it on request.
    
    Used as the initial step of __init__.
    '''
    # We have an optional argument to override the instance class
    # None is the default value (to mean: don't interfere with subclassing)
    if cast_as_class is not None:
      # cast_as_class accepts the classes themselves as well as their string representations
      cast_as_class = Digraph.read_subclass_from_string(cast_as_class, require_string = False)
      try:
        self.__class__ = cast_as_class
      except:
        raise ValueError('Cannot cast as requested class.')
    else:
      # If cast_as_class == None we don't interfere with the initialization
      pass
    # We create the three base attributes: _arrows, _arrows_out, _arrows_in
    # (Or reset them, in case this __init__ is used for a reset/recast)
    self._arrows = []
    self._arrows_out = {}
    self._arrows_in = {}
    # These are all internal attributes for a non-Graph Digraph.
    # A Graph (and only a Graph) has self._edges and self._inciding_edges
    if isinstance(self, Graph):
      # Create or reset the needed attributes
      self._edges = []
      self._inciding_edges = {}
    else:
      # In this case we don't want to have self._edges nor self._inciding edges
      # Since this might be a cast from another class (using cast_as_another_class
      #or a similar mechanism) we prefer, for purposes of standartization,
      #to delete those attributes (if existent)
      if hasattr(self, '_edges'):
        del self._edges
      if hasattr(self, '_inciding_edges'):
        del self._inciding_edges

  @staticmethod
  def _extract_and_organize_data(data, data_type):
    '''
    Extracts and organizes the data provided according to its data_type.
    
    Used as a step in __init__.
    '''
    # We do some unpacking
    # Note that this is a static method so we proceed only based on the data
    #given and not on whether the Digraph is a Graph, for example
    if 'all_arrows' in data_type.lower():
      if data_type.lower() == 'all_arrows':
        init_vertices = []
        init_arrows = data
        require_vertices_in = False
      elif data_type.lower() == 'some_vertices_and_all_arrows':
        init_vertices, init_arrows = data
        require_vertices_in = False
      elif data_type.lower() == 'all_vertices_and_all_arrows':
        init_vertices, init_arrows = data
        require_vertices_in = True
      else:
        raise ValueError('Option not recognized')
      init_tuplees = init_arrows
      use_edges_instead_of_arrows = False
    elif 'all_edges' in data_type.lower():
      if data_type.lower() == 'all_edges':
        init_vertices = []
        init_edges = data
        require_vertices_in = False
      elif data_type.lower() == 'some_vertices_and_all_edges':
        init_vertices, init_edges = data
        require_vertices_in = False
      elif data_type.lower() == 'all_vertices_and_all_edges':
        init_vertices, init_edges = data
        require_vertices_in = True
      else:
        raise ValueError('Option not recognized')
      init_tuplees = init_edges
      use_edges_instead_of_arrows = True
    elif 'as_dict' in data_type.lower() or 'as_list' in data_type.lower():
      # We need to format the information
      # First, to save time coding, we format a list (in the 'as_list' option)
      # to have the same exact info as the corresponding dict would have
      if 'as_list' in data_type.lower():
        data_as_dict = {}
        for item in data:
          if item[0] in data_as_dict:
            # Repeated information, better to abort
            raise ValueError('Duplicated keys derived from data')
          else:
            # All clear, all first item as key and rest as value
            data_as_dict[item[0]] = item[1:]
      else:
        # Necessarily 'as_dict' in data_type.lower()
        data_as_dict = data
      # Now do things common to both, acting on data_as_dict
      # All of them are dictionaries whose keys are vertices
      init_vertices = list(data_as_dict)
      # On eaxmining the dictionary, we allow for inclusion of new vertices
      #if and only if we don't have a 'full_' data_type option
      if 'full_' in data_type.lower(): 
        require_vertices_in = True
      else:
        require_vertices_in = False
      # We now produce the arrows or edges
      # (Note they will go under further formatting later when being added)
      if 'arrows_out_as_' in data_type.lower():
        # We expect the values of the dict to be lists with the arrows
        init_arrows = []
        for vertex in data_as_dict:
          init_arrows.extend(data_as_dict[vertex])
        use_edges_instead_of_arrows = False
        init_tuplees = init_arrows
      elif 'edges_out_as_' in data_type.lower():
        init_edges = []
        for vertex in data_as_dict:
          init_edges.extend(data_as_dict[vertex])
        use_edges_instead_of_arrows = True
        init_tuplees = init_edges
      elif 'neighbors_out_as_' in data_type.lower():
        # There is some complexity and so we defer to another method
        init_arrows = OperationsVAE.get_namedtuples_from_neighbors(data_as_dict,
            output_as = 'list',
            namedtuple_choice = Arrow,
            require_namedtuple = False,
            request_vertex_sanitization = True,
            require_vertex_namedtuple = False)
        use_edges_instead_of_arrows = False
        init_tuplees = init_arrows
      elif 'neighbors_as_' in data_type.lower():
        # There is some complexity and so we defer to another method
        init_edges = OperationsVAE.get_namedtuples_from_neighbors(data_as_dict,
            output_as = 'list',
            namedtuple_choice = Edge,
            require_namedtuple = False,
            request_vertex_sanitization = True,
            require_vertex_namedtuple = False)
        use_edges_instead_of_arrows = True
        init_tuplees = init_edges
      else:
        raise ValueError('Option not recognized')
    else:
      raise ValueError('Option not recognized')
    # Return as a single tuple
    # Note init_arrows and init_edges are encoded as init_tuplees for this packing
    # use_edges_instead_of_arrows is the correct key for their unpacking
    init_data = (use_edges_instead_of_arrows, init_vertices, init_tuplees, require_vertices_in)
    return init_data

  def _construct_instance(self, init_data):
    '''
    Sets the attributes _arrows, _arrows_out and _arrows_in to have their correct values.
    (For a Graph, does also _edges and _inciding_edges.)
    
    Attributes are built from init_data, which should contain:
    use_edges_instead_of_arrows
    init_vertices
    init_tuplees (transforms into init_arrows or init_edges)
    require_vertices_in
    
    Used as final step of __init__.
    '''
    # Unpack and clarify the names of the variables
    use_edges_instead_of_arrows, init_vertices, init_tuplees, require_vertices_in = init_data
    # Note: we always set require_namedtuple and require_vertex_nametduple to False;
    #we are quite fine in being flexible with the input requirements
    if use_edges_instead_of_arrows:
      init_edges = init_tuplees
    else:
      init_arrows = init_tuplees
    # We add vertices [note this might well be empty]
    # Note: repeating vertices is never allowed
    self._add_vertices(init_vertices, require_vertex_not_in = True,
        require_vertex_namedtuple = False)
    # We build arrows and edges from init_arrows or init_edges, making sure
    #to set the attributes _edges and _inciding_edges for Graphs
    if use_edges_instead_of_arrows:
      add_as_edges = isinstance(self, Graph)
      self._add_edges(init_edges,
          require_vertices_in = require_vertices_in,
          add_as_edges = add_as_edges,
          add_as_arrows = True,
          require_namedtuple = False,
          require_vertex_namedtuple = False)
    else:
      also_add_formed_edges = isinstance(self, Graph)
      self._add_arrows(init_arrows,
          require_vertices_in = require_vertices_in,
          also_add_formed_edges = also_add_formed_edges,
          require_namedtuple = False,
          require_vertex_namedtuple = False)

########################################################################
# Methods for adding vertices, edges, arrows, used in initialization
########################################################################
    
  def _add_vertex(self, vertex, require_vertex_not_in = False,
      require_vertex_namedtuple = False):
    '''
    Adds a vertex as Vertex namedtuple to the digraph.
    
    INPUTS:
    self: own instance
    vertex: name or content of the vertex [must be hashable]
    require_vertex_not_in: gives error if vertex to be added is already in
    require_vertex_namedtuple: requires Vertex namedtuple as argument
    
    OUTPUTS:
    (None: alters self)
    '''
    # We pass most of the formatting/checking to sanitive_vertex()
    # (That includes the detection of being a Vertex if require_namedtuple is True)
    vertex = OperationsVAE.sanitize_vertex(vertex,
        require_vertex_namedtuple = require_vertex_namedtuple)
    # We determine whether the vertex is already in the graph
    if self.belongs_to_as_vertex_after_sanitization(vertex, require_vertex_namedtuple = False):
      # In this case vertex is already present
      if require_vertex_not_in:
        raise ValueError('Vertex already present')
      else:
        # Vertex present, but not a problem. Leave the method
        pass
    else:
      # In this case all clear, we can add the vertex to the relevant dicts
      self._arrows_out[vertex] = []
      self._arrows_in[vertex] = []
      # If we have Graph on top of Digraph, we need to deal with self._inciding_edges
      if isinstance(self, Graph):
        self._inciding_edges[vertex] = []

  def _add_vertices(self, vertices, require_vertex_not_in = False,
      require_vertex_namedtuple = False):
    '''
    Adds an iterable of vertices to self.
    '''
    for vertex in vertices:
      self._add_vertex(vertex,
          require_vertex_not_in = require_vertex_not_in,
          require_vertex_namedtuple = require_vertex_namedtuple)

  def _add_arrow(self, arrow, require_vertices_in = False, require_namedtuple = False,
      require_vertex_namedtuple = False):
    '''
    Adds (weighted or unweighted) arrow to self.
    '''
    # We verify it is a valid arrow, putting it into the right format if it makes sense
    # We don't mind if we start with a simple tuple instead of the named tuple Arrow
    # We will put it into a namedtuple Arrow, that is, a sanitized arrow
    # Vertex is also to be sanitized before going in, with request_vertex_sanitization
    arrow = OperationsVAE.sanitize_arrow_or_edge(arrow,
        use_edges_instead_of_arrows = False,
        require_namedtuple = require_namedtuple,
        request_vertex_sanitization = True,
        require_vertex_namedtuple = require_vertex_namedtuple)
    # We check whether the vertices are already present
    # If require_vertices_in, we raise an error if the vertices are not
    #already present. Otherwise, we add the vertices too.
    if not self.belongs_to_as_vertex_after_sanitization(arrow.source, require_vertex_namedtuple = False):
      if require_vertices_in:
        raise ValueError('Source of arrow needs to be in digraph.')
      else:
        self._add_vertex(arrow.source,
            require_vertex_not_in = True,
            require_vertex_namedtuple = False)
    if not self.belongs_to_as_vertex_after_sanitization(arrow.target, require_vertex_namedtuple = False):
      if require_vertices_in:
        raise ValueError('Target of arrow needs to be in digraph.')
      else:
        self._add_vertex(arrow.target,
            require_vertex_not_in = True,
            require_vertex_namedtuple = False)
    # We now work on the arrow
    self._arrows_in[arrow.target].append(arrow)
    self._arrows_out[arrow.source].append(arrow)
    self._arrows.append(arrow)

  def _add_arrows(self, arrows, require_vertices_in = False,
      also_add_formed_edges = False, require_namedtuple = False,
      require_vertex_namedtuple = False):
    '''
    Adds an iterable of (weighted or unweighted) arrows to self.
    
    If arrows form edges, they can be requested to be added too, in a way
    which is not possible with a single arrow.
    '''
    # If we require sanitize arrows, we do this always in this function, at once
    # We do slightly different depending on the edge formation requirement
    if also_add_formed_edges:
      arrows, edges = OperationsVAE.sanitize_arrows_and_return_formed_edges(arrows,
          require_namedtuple = False,
          request_vertex_sanitization = False,
          require_vertex_namedtuple = False,
          raise_error_if_edges_not_formed = True)
    else:
      arrows = OperationsVAE.sanitize_arrows_or_edges(arrows,
          use_edges_instead_of_arrows = False,
          require_namedtuple = require_namedtuple,
          request_vertex_sanitization = True,
          require_vertex_namedtuple = False)
    # We add the arrows
    for arrow in arrows:
      self._add_arrow(arrow,
          require_vertices_in = require_vertices_in,
          require_namedtuple = True)
    # Now we add the edges, if requested
    if also_add_formed_edges:
      # Easiest way is to call _add_edges, and of course, ask to not add arrows
      # (Otherwise they would be added in double)
      # We can set require_vertices_in = True as the vertices of any edge
      #would have been already added as an arrow previously
      self._add_edges(edges,
          require_vertices_in = True,
          add_as_edges = True,
          add_as_arrows = False,
          require_namedtuple = require_namedtuple,
          require_vertex_namedtuple = require_vertex_namedtuple)
    else:
      # Nothing else. Arrows already added
      pass
    
  def _add_edge(self, edge, require_vertices_in = False, add_as_edge = False,
      add_as_arrows = False, require_namedtuple = False,
      require_vertex_namedtuple = False):
    '''
    Adds an edge to the (di)graph.
    
    Edge appears as as two arrows in self._arrows, among other attributes
    It also appears as one edge in self._edges (in Graph instance only)
    '''
    # This method is used for Graphs and for non-Graph Digraphs
    # For non-Graph Digraphs, add_as_edges should be False, as there would be
    #no attributes self._edges and self._inciding_edges
    # We first put the edge into a namedtuple, if not already [sanitize it]
    edge = OperationsVAE.sanitize_arrow_or_edge(edge,
        use_edges_instead_of_arrows = True,
        require_namedtuple = require_namedtuple,
        request_vertex_sanitization = True,
        require_vertex_namedtuple = require_vertex_namedtuple)
    # We check whether the vertices are already present
    # Note this will be conducted even if add_as_edge and add_as_arrows are False
    if not self.belongs_to_as_vertex_after_sanitization(edge.first, require_vertex_namedtuple = False):
      if require_vertices_in:
        raise ValueError('Source of edge needs to be in (di)graph.')
      else:
        self._add_vertex(edge.first,
            require_vertex_not_in = True,
            require_vertex_namedtuple = False)
    if not self.belongs_to_as_vertex_after_sanitization(edge.second, require_vertex_namedtuple = False):
      if require_vertices_in:
        raise ValueError('Target of edge needs to be in (di)graph.')
      else:
        self._add_vertex(edge.second,
            require_vertex_not_in = True,
            require_vertex_namedtuple = False)
    # We care about adding the edges (depend on instance class)
    # Note that if Digraph is Graph we need to deal with more attributes
    # We will trust our flag add_as_edge for the proper discrimination
    # (Otherwise we run into an AttributeError)
    if add_as_edge:
      # That is, we need to add the edge information as an edge:
      self._edges.append(edge)
      self._inciding_edges[edge.first].append(edge)
      self._inciding_edges[edge.second].append(edge)
    # It may or may not be the case that we will add the arrows
    # That is, it depends on the request, independently of add_as_edge
    if add_as_arrows:
      # We now work on the arrows: every edge also makes two arrows.
      # We produce a list of two Arrows using get_arrows_from_edge,
      #then use self._add_arrow
      two_arrows = OperationsVAE.get_arrows_from_edge(edge,
          require_namedtuple = True,
          request_vertex_sanitization = False,
          require_vertex_namedtuple = True)
      for arrow in two_arrows:
        # All checks done already, don't need to put any requirement
        # (Note that edges have been sanitized already.)
        self._add_arrow(arrow,
            require_vertices_in = False,
            require_namedtuple = True,
            require_vertex_namedtuple = True)      
    else:
      # Nothing else
      pass

  def _add_edges(self, edges, require_vertices_in = False, add_as_edges = False,
      add_as_arrows = False, require_namedtuple = False, require_vertex_namedtuple = False):
    '''
    Adds an iterable of edges to the (di)graph.
    '''
    add_as_edge = add_as_edges
    for edge in edges:
      self._add_edge(edge,
          require_vertices_in = require_vertices_in,
          add_as_edge = add_as_edge,
          add_as_arrows = add_as_arrows,
          require_namedtuple = require_namedtuple,
          require_vertex_namedtuple = require_vertex_namedtuple)

########################################################################
# Methods representing the digraph as a string
########################################################################

  def __repr__(self):
    '''
    Magic method. Returns faithful representation of instance.
    '''
    # We believe using arrow_out_as_dict is the best way
    # (Could alternatively use provide_unique_presentation to generate info)
    # (Note that using edges for a Graph complicated things. Better to aim
    #for initializing always as arrows_out_as_dict in all cases)
    instance_class = self.__class__.__name__
    data = repr(self.get_arrows_out_as_dict())
    data_type = 'arrows_out_as_dict'
    return '{}(data = {}, data_type = {}, cast_as_class = None)'.format(
        instance_class, data, data_type)
    
  def __str__(self):
    '''
    Magic method. Returns user-friendly representation of instance.
    '''
    # In the future, this might become a more comprehensive description,
    #with more details, or even a visual representation.
    # Right now, we get it from provide_short_summary
    return self.provide_short_summary()
    
  def provide_short_summary(self):
    '''
    Returns a string summarizing the most basic information about the instance.
    
    For a Graph, provides number of vertices and number of edges;
    otherwise, provides number of vertices and number of arrows.
    '''
    if isinstance(self, Graph):
      object_in_one_word = 'graph'
      relevant_components = '{} edges'.format(self.get_number_of_edges())
    else:
      object_in_one_word = 'digraph'
      relevant_components = '{} arrows'.format(self.get_number_of_arrows())
    return 'A {} with {} vertices and {}.'.format(
        object_in_one_word, self.get_number_of_vertices(), relevant_components)

########################################################################
# Methods to compare digraphs
########################################################################

  def provide_unique_representation(self):
    '''
    Returns all information about the digraph.
    
    Returns a tuple with the class of the instance, with a set of its vertices
    as well as a multi-set (Counter from built-in package collections)
    of arrows.
    '''
    # We return the class/type and the information on vertices and arrows
    # Idea is that two digraphs with same output are the same for all purposes.
    # There could be shortcuts to make this faster. For example, We opt for clarity
    instance_class = type(self)
    set_of_vertices = set(self.get_vertices())
    multiset_of_arrows = collections_Counter(self.get_arrows())
    return (instance_class, set_of_vertices, multiset_of_arrows)
    
  def __eq__(self, other, *, require_equal_classes = False):
    '''
    Magic method. Returns whether two instances are the same object.
    
    Has an option (default False) to require the subclassing to coincide.
    '''
    # To reduce the computing time (in particular avoiding forming sets/Counter)
    #we first look at class, then only at vertices (typically their number
    #is one order less than ), then finally at the whole "unique representation"
    if require_equal_classes:
      if type(self) != type(other):
        return False
    try:
      if set(self.get_vertices()) == set(other.get_vertices()):
        if self.provide_unique_representation() == other.provide_unique_representation():
          return True
        else:
          return False
      else:
        return False
    except AttributeError:
      # AttributeError: most likely other instance not a Digraph
      return False

########################################################################
# Methods which read simple information from the graph
########################################################################

  def __contains__(self, vertex):
    '''
    Magic method. Returns whether a vertex belongs to the graph.
    '''
    # This is to be used throughout the code
    # So we do a direct attribute lookup and then a dict key lookup
    #(instead of calling get_vertices() which would take longer)
    # Use __contains__ for conditionals: vertex in self
    # For looping over vertices self.get_vertices() is unavoidable
    return vertex in self._arrows_out
    
  def belongs_to_as_vertex_after_sanitization(self, obj, require_vertex_namedtuple = False):
    '''
    Determines if object belongs to the graph as a vertex.
    
    If require_vertex_namedtuple, this is __contains__.
    
    If not require_vertex_namedtuple, we consider whether the object belongs
    only after being Vertex-ified: sanitized into Vertex.
    '''
    # Use this to provide a more flexible __contains__
    if require_vertex_namedtuple:
      # In this case obj is unaltered by sanitize_vertex
      return obj in self
    else:
      # In this case either obj or sanitize_vertex(obj, **args) should be in
      sanitized_obj = OperationsVAE.sanitize_vertex(obj, require_vertex_namedtuple = False)
      return sanitized_obj in self

  def __bool__(self):
    '''
    Magic method. Returns boolean value of digraph, which is False for the
    empty digraph (no vertices) are True otherwise.
    '''
    return bool(self._arrows_out)

  def is_nonempty_digraph(self):
    '''
    Alias for __bool__.
    
    SEE: __bool__
    '''
    return bool(self)

  def get_vertices(self):
    '''
    Returns all vertices of self.
    '''
    # The vertices are stored as the keys of the dict self._arrows_out
    return list(self._arrows_out)

  def get_number_of_vertices(self):
    '''
    Returns the number of vertices of self.
    '''
    return len(self._arrows_out)
    
  def get_number_of_arrows(self):
    '''
    Returns the number of arrows in self.
    '''
    return len(self._arrows)

  def get_in_degree_of_vertex(self, vertex, skip_checks = False):
    '''
    Returns in-degree of vertex in digraph: the number of arrows in.
    '''
    if not ignore_checks:
      assert self.belongs_to_as_vertex_after_sanitization(vertex, require_vertex_namedtuple = False), 'Vertex must be in digraph'
    return len(self._arrows_in[vertex])
    
  def get_out_degree_of_vertex(self, vertex, skip_checks = False):
    '''
    Returns out-degree of vertex in digraph: the number of arrows out.
    '''
    if not ignore_checks:
      assert self.belongs_to_as_vertex_after_sanitization(vertex, require_vertex_namedtuple = False), 'Vertex must be in digraph'
    return len(self._arrows_out[vertex])

  def get_neighbors_in(self, vertex, forbid_repetitions = False, skip_checks = False):
    '''
    Provides the vertices to which a vertex has arrows from.
    
    Has option to repeat or not vertices in case of multidigraph.
    '''
    if not skip_checks:
      assert self.belongs_to_as_vertex_after_sanitization(vertex, require_vertex_namedtuple = False), 'Vertex must be in digraph'
    list_with_possible_repetitions = [arrow_in.source for arrow_in in self._arrows_in[vertex]]
    if forbid_repetitions:
      return list(set(all_with_possible_repetitions))
    else:
      return list_with_possible_repetitions
    
  def get_neighbors_out(self, vertex, forbid_repetitions = False, skip_checks = False):
    '''
    Provides the vertices to which a vertex has arrows to.
    
    Has option to repeat or not vertices in case of multidigraph.
    '''
    if not skip_checks:
      assert self.belongs_to_as_vertex_after_sanitization(vertex, require_vertex_namedtuple = False), 'Vertex must be in digraph'
    list_with_possible_repetitions = [arrow_out.target for arrow_out in self._arrows_out[vertex]] 
    if forbid_repetitions:
      return list(set(all_with_possible_repetitions))
    else:
      return list_with_possible_repetitions
    
  def get_arrows(self, output_as = None):
    '''
    Returns all arrows of self.
    '''
    # We make list the default output
    # Originally self._arrows is stored as tuple
    if output_as == None:
      output_as = 'list'
    if output_as.lower() == 'list':
      return list(self._arrows)
    elif output_as.lower() == 'tuple':
      return tuple(self._arrows)
    else:
      raise ValueError('Output option not recognized')

  def get_arrows_out_as_dict(self):
    '''
    Returns arrows going out of each vertex.
    '''
    return self._arrows_out

  def get_arrows_out(self, vertex, skip_checks = False):
    '''
    Returns arrows going out of a vertex.
    '''
    if not skip_checks:
      assert self.belongs_to_as_vertex_after_sanitization(vertex, require_vertex_namedtuple = False), 'Vertex must be in digraph'
    return self._arrows_out[vertex]

  def get_arrows_in_as_dict(self):
    '''
    Returns arrows going into each of the vertices.
    '''
    return self._arrows_in

  def get_arrows_in(self, vertex, skip_checks = False):
    '''
    Returns arrows coming into a vertex.
    '''
    if not skip_checks:
      assert self.belongs_to_as_vertex_after_sanitization(vertex, require_vertex_namedtuple = False), 'Vertex must be in digraph'
    return self._arrows_in[vertex]

########################################################################
# Methods for detecting best subclassing and casting
########################################################################

  @classmethod
  def suggest_properties_from_subclassing(cls):
    '''
    Returns information about the digraph based solely on its subclass.
    '''
    # First we check if it is an undirected graph (called Graph class)
    if issubclass(cls, Graph):
      is_undirected = True
    else:
      is_undirected = False
    # A Digraph with no arrows can theoretically be simultaneously weighted and unweighted
    # That is why is_weighted and is_unweighted are independent
    if issubclass(cls, WeightedDigraph):
      is_weighted = True
    else:
      is_weighted = False
    if issubclass(cls, UnweightedDigraph):
      is_unweighted = True
    else:
      is_unweighted = False
    return (is_undirected, is_weighted, is_unweighted)

  def is_undirected_weighted_unweighted(self):
    '''
    Reports about the digraph: is undirected, is weighted, is unweighted.
    
    Does not look as subclassing. A digraph with no arrows is considered
    undirected, weighted and unweighted.
    '''
    is_undirected = self.is_digraph_undirected()
    is_weighted = self.is_digraph_weighted() 
    is_unweighted = self.is_digraph_unweighted()
    return (is_undirected, is_weighted, is_unweighted)

  def is_digraph_undirected(self):
    '''
    Returns whether the digraph can be made undirected. [For this docstring,
    it is called digraph as long as not proven undirected.]
    
    For that to happen, each arrow (v, w) will correspond to an arrow (w, v).
    
    Does not look as subclassing. A digraph with no arrows is undirected.
    '''
    # Don't want to check the class name, but to verify it is possible
    #to build, with the given arrows, a Graph [an undirected graph]
    # We use the static method get_edges_from_sanitized_arrows, which returns
    #None when the arrows cannot be used to form edges
    new_edges = OperationsVAE.get_edges_from_sanitized_arrows(
        self,
        is_data_a_digraph_instead_of_arrows = True,
        is_multiarrow_free_guaranteed = False)
    # We are explicit on what we're doing
    is_undirected = (new_edges is not None)
    return is_undirected

  def is_digraph_weighted(self):
    '''
    Determines if all arrows of digraph are weighted.
    
    Does not look as subclassing. A digraph with no arrows is weighted.
    '''
    # We verify whether all arrows are weighted.
    # We use self.get_arrows() which reads self._arrows
    # (We could also verify the edges, or also observe self._arrows_out
    #and others, but we assume initialization was correct and therefore
    #self._arrows has all information)
    # Note math.inf (i. e. infinity) counts as float. Since sometimes it
    #is a convention that an infinite distance between two vertices means
    #there is no path between them, we forbid math_inf to be a valid weight
    test = lambda weight: (isinstance(weight, float) and weight != math_inf)
    return all(test(arrow.weight) for arrow in self.get_arrows())

  def is_digraph_unweighted(self):
    '''
    Determines if all arrows of digraph are weighted.
    
    Does not look as subclassing. A digraph with no arrows is unweighted.
    '''
    return all((arrow.weight is None) for arrow in self.get_arrows())

  @staticmethod
  def suggest_subclasses_from_properties(is_undirected, is_weighted, is_unweighted):
    '''
    Given some properties a digraph might have, suggest fitting subclasses.
    
    Returns a list of all possible subclasses it could be a subclass of.
    The best fitting subclass is always the first item.
    '''
    # We provide a few names to help (they disappear outside this method)
    D, WD, UD = Digraph, WeightedDigraph, UnweightedDigraph
    G, WG, UG = Graph, WeightedGraph, UnweightedGraph
    if is_undirected:
      if is_weighted and is_unweighted:
        # Possible for an empty graph
        return [G, UG, WG, D, UD, WD]
      elif is_weighted:
        return [WG, G, WD, D]
      elif is_unweighted:
        return [UG, G, UD, D]
      else:
        return [G, UG, WG, D, UD, WD]
    else:
      if is_weighted and is_unweighted:
        # Possible for an empty digraph
        return [D, UD, WD]
      elif is_weighted:
        return [WD, D]
      elif is_unweighted:
        return [UD, D]
      else:
        return [D, UD, WD]

  @staticmethod
  def read_subclass_from_string(name, require_string = False):
    '''
    Given a string, produces the Digraph subclass which best matches the name.
    
    There are 6 possible classes: Digraph, UnweightedDigraph, WeightedDigraph,
    Graph, UnweightedGraph, WeightedGraph.
    
    Has an option to require a string as name. If False, it accepts the
    class itself as name argument.
    '''
    # If we start with a type, if allowed, we convert it to a string first
    if isinstance(name, type):
      # We have the class already. If that was a valid argument, we return it
      if require_string:
        raise TypeError('Need a string to be converted into a class.')
      elif name not in [Digraph, UnweightedDigraph, WeightedDigraph,
          Graph, WeightedDigraph, UnweightedGraph]:
        raise TypeError('Need one of the 6 (di)graph classes.')
      else:
        name = name.__name__
    # We work on a string (possibly coming from being a type previously)
    if isinstance(name, str):
      if 'digraph' in new_class.lower():
        # For Digraph, UnweightedDigraph, WeightedDigraph
        if 'unweighted' in new_class.lower():
          new_class = UnweightedDigraph
        elif 'weighted' in new_class.lower():
          new_class = WeightedDigraph
        else:
          new_class = Digraph
      elif 'graph' in new_class.lower():
        # For Graph, UnweightedGraph, WeightedGraph
        if 'unweighted' in new_class.lower():
          new_class = UnweightedGraph
        elif 'weighted' in new_class.lower():
          new_class = WeightedGraph
        else:
          new_class = Graph
      else:
        raise ValueError('Can\'t recognize requested subclass.')
    else:
      # Only accept types/classes and strings
      raise TypeError('Can only read class from a string or a type.')

  def cast_as_another_class(self, new_class = None, modify_self = False, accept_data_loss = False):
    '''
    Given an instance of any subclass of Digraph, make it into an instance
    of another subclass of Digraph.
    
    Can either modify object in place (returning None), or create a new,
    equivalent object and return it [keeping original intact].
    
    Has an option to accept data loss, for example, removing weights
    from arrows when they don't all have weight 1. If data loss is not
    accepted but is requested from the other arguments, an error is raised.
    
    Some changes are impossible: for example, a strictly directed graph
    into an undirected graph (i. e. a Graph). In this case, raises error.
    '''
    # First we deal with None; if no class is suggested, we find the best fit.
    if new_class == None:
      properties = self.is_undirected_weighted_unweighted()
      new_class = Digraph.suggest_subclasses_from_properties(properties)[0]
    # We expect new_class to be given as string or as the class itself
    new_class = Digraph.read_subclass_from_string(name, require_string = False)
    # We have new_class well established. We compare with the current class
    # (Note their names are stored in old_class.__name__ and new_class.__name__)
    old_class = self.__class__
    # If they are the same, it's easy
    if old_class == new_class:
      if modify_self:
        # Nothing to change
        return None
      else:
        return copy_copy(self)
    else:
      # From now on it is assumed old and new classes are different
      # We write down properties of the current class and of the desired class
      # We abbreviate undirected, weighted, unweighted for our convenience
      new_und, new_wei, new_unw = Digraph.suggest_properties_from_subclassing(new_class)
      old_und, old_wei, old_unw = Digraph.suggest_properties_from_subclassing(old_class)
      # This is the moment where we look for contradictions
      # That is, impossible subclassing castings
      are_all_weights_one = all(arrow.weight == 1 for arrow in self.get_arrows())
      # We use a flag to tell if casting is possible or if errors should be raised
      is_casting_allowed = True
      if new_und and not old_und:
        is_casting_allowed = False
      elif old_wei and not new_wei and not accept_data_loss and not are_all_weights_one:
        # That is, we are going from weighted to unweighted, not all weights are 1,
        #and we are not accepting data loss
        is_casting_allowed = False
      if is_casting_allowed:
        # Casting is possible
        # We use __init__ to reinitialize the own instance
        # Note an instance has self._edges and self._inciding_edges
        #if and only if it is a Graph (at least that is how __init__ is designed)
        # In the case old_class is undirected, using this information
        #could potentially speed up the formation of another undirected graph
        # (And close to zero effect on digraphs, and the information on
        #edges of that graph won't be kept at all, and that's the design)
        if old_und:
          # Use vertices and edges
          data = (self.get_vertices(), self.get_edges())
          data_type = 'all_vertices_and_all_edges'
        else:
          # Use vertices and arrows
          data = (self.get_vertices(), self.get_arrows())
          data_type = 'all_vertices_and_all_arrows'
        # If we want to change from weighted to unweightd or vice-versa
        #we need to adapt the arrows/edges
        # Note if self has no arrows then old_wei and old_unw are True
        # The following lines take it into consideration (they will have
        #no effect as there will be no arrows or edges to change)
        if new_wei and old_unw:
          # Slightly differently depending on arrows or edges
          if old_und:
            # Note default weight is supposed to be 1
            weighted_edges_or_arrows = OperationsVAE.write_weights_into_arrows_or_edges(
                unweighted_tuples = data[1],
                use_edges_instead_of_arrows = True,
                new_weights = [1 for item in range(len(data[1]))],
                require_namedtuple = True)
          else:
            weighted_edges_or_arrows = OperationsVAE.write_weights_into_arrows_or_edges(
                unweighted_tuples = data[1],
                use_edges_instead_of_arrows = False,
                new_weights = [1 for item in range(len(data[1]))],
                require_namedtuple = True)
          data = (data[0], weighted_edges_or_arrows)
        elif new_unw and old_wei:
          # Possible if accept_data_loss is True, or if all have weight 1
          if old_und:
            unweighted_edges_or_arrows = OperationsVAE.remove_weights_from_arrows_or_edges(
                weighted_tuples = data[1],
                use_edges_instead_of_arrows = True,
                require_namedtuple = True)
          else:
            unweighted_edges_or_arrows = OperationsVAE.remove_weights_from_arrows_or_edges(
                weighted_tuples = data[1],
                use_edges_instead_of_arrows = False,
                require_namedtuple = True)
          data = (data[0], unweighted_edges_or_arrows)
        else:
          # Nothing to adapt on edges/arrows regarding weights
          pass
        # Finally, data and data_type are correct, and we can either
        #modify self or return a new instance with the requested features
        if modify_self:
          self.__init__(data = data, data_type = data_type, cast_as_class = new_class)
          return None
        else:
          # Create a new instance via __init__ and return it
          # (This can me accomplished in multiple ways)
          return new_class(data = data, data_type = data_type, cast_as_class = None)
      else:
        # Casting is not possible
        raise ValueError('Cannot make the requested subclass casting.')

########################################################################
# Uncategorized methods
########################################################################

  def produce_path_or_cycle_with_at_most_one_vertex(self, compute_path_instead_of_cycle,
      vertex = None, skip_checks = False):
    '''
    Produces a path or cycle from the digraph with one or zero vertices.
    
    If argument vertex is given, that is the sole vertex of the path or cycle.
    If given as None, produces a no-vertex path or cycle.
    '''
    if vertex is None:
      vertex_data = []
    else:
      if not skip_checks:
        assert vertex in self, 'Non-None vertex should be in digraph'
      vertex_data = [vertex]
    kwargs = {'underlying_digraph': self,
        'data': vertex_data,
        'data_type': 'vertices',
        'verify_validity_on_initialization': not skip_checks}
    if compute_path_instead_of_cycle:
      instance_class = VertexPath
    else:
      instance_class = VertexCycle
    return instance_class(**kwargs)

  def make_multidigraph_simple(self, modify_self = False):
    '''
    From a multidigraph obtains a simple digraph [at most one arrow for each
    pair of vertices].
    
    When there are multiple arrows from the same pair, and the digraph is
    weighted, the shortest [lowest weight] arrow is picked.
    
    Can modify self, or return a new instance with the simple graph.
    '''
    # If we have a Graph, we can more or less cut the work in half
    #by looking at the edges instead of at the vertices
    # Since it is a bit simpler with arrows we do it first
    if not isinstance(self, Graph):
      # Procedure: we copy arrows from self.get_arrows() into shortest_by_pair
      #in corresponding pair of source and target, only one per pair
      # We keep the arrow with smaller weight if two arrows correspond to
      #same pair, or any if arrows are unweighted
      # To keep the work linear on the number of arrows (and not quadratic)
      #and avoid comparisons we use [implicitly, using dict] hashing
      vertices = self.get_vertices()
      pairs_of_vertices = [(u, v) for u in vertices for v in vertices]
      shortest_by_pair = {pair:None for pair in pairs_of_vertices}
      for arrow in self.get_arrows():
        pair_of_arrow = (arrow.source, arrow.target)
        current_shortest = shortest_by_pair[pair_of_arrow]
        if current_shortest is None:
          # First arrow with same source and target, set a flag to add it
          should_write_into_shortest = True
        elif current_shortest.weight is None:
          # This is the case where the digraph is unweighted. Keep first
          should_write_into_shortest = False
        elif arrow.weight < current_shortest.weight:
          # Only possible for weighted
          should_write_into_shortest = True
        else:
          should_write_into_shortest = False
        # If marked as shortest, put it as the value for corresponding pair
        if should_write_into_shortest:
          shortest_by_pair[pair_of_arrow] = arrow
        else:
          pass
      # We now prepare the data for using __init__
      data = (vertices, list(shortest_by_pair.values()))
      data_type = 'all_vertices_and_all_arrows'
    else:
      # We operate very similarly, except that we observe the edges
      # We need to be careful with target and source (or first and second)
      # This is easily done using set
      vertices = self.get_vertices()
      pairs_of_vertices = [set([u, v]) for u in vertices for v in vertices]
      shortest_by_pair = {pair:None for pair in pairs_of_vertices}
      for edge in self.get_edges():
        pair_of_edge = set([edge.first, edge.second])
        current_shortest = shortest_by_pair[pair_of_edge]
        if current_shortest is None:
          should_write_into_shortest = True
        elif current_shortest.weight is None:
          should_write_into_shortest = False
        elif edge.weight < current_shortest.weight:
          should_write_into_shortest = True
        else:
          should_write_into_shortest = False
        if is_this_shortest:
          shortest_by_pair[pair_of_edge] = edge
        else:
          pass
      # We prepare the data for __init__. We use vertices and edges
      data = (vertices, list(shortest_by_pair.values()))
      data_type = 'all_vertices_and_all_edges'
    # Now we modify self or create new instance
    # At this moment, data and data_type are correctly defined
    # Note that we don't change the subclassing (that's __init__ third argument)
    if modify_self:
      self.__init__(data = data, data_type = data_type, cast_as_class = None)
      return None
    else:
      return self.__class__(data = data, data_type = data_type, cast_as_class = None)

  def get_shortest_arrow_between_vertices(self, source, target, skip_checks = False):
    '''
    Gets the shortest arrow between two vertices.
    
    Returns None if there is no direct arrow between the vertices.
    
    If the digraph is unweighted (and there are arrows between them),
    any of those arrows might be returned.
    '''
    if not skip_checks:
      assert self.belongs_to_as_vertex_after_sanitization(source, require_vertex_namedtuple = False), 'Need source to be a vertex'
      assert self.belongs_to_as_vertex_after_sanitization(target, require_vertex_namedtuple = False), 'Need target to be a vertex'
    # We try to make this the most efficient possible
    # Easiest way is by looking at self._arrows_out
    shortest_arrow = None
    for arrow in self.get_arrows_out(source):
      # We only consider the arrows with right source and target
      if arrow.target == target:
        if shortest_arrow is None:
          # Make it the shortest
          should_write_into_shortest = True
        elif shortest_arrow.weight is None:
          # This should not be possible
          # After all, we already tested the current arrow for having
          #None as weight when it was made shortest_arrow
          raise RuntimeError('Internal logic error.')
        elif arrow.weight < shortest_arrow.weight:
          # In this case replace the current arrow
          should_write_into_shortest = True
        else:
          should_write_into_shortest = False
        if should_write_into_shortest:
          # Write into shortest
          shortest_arrow = arrow
          if shortest_arrow.weight is None:
            # If the weight is None, we have an unweighted digraph
            # In this case, we cannot improve the weight, so we break the loop
            break
          else:
            pass
        else:
          # Continue the loop, looking for good arrows
          pass
    # End of loop. Return value in shortest_arrow (either Arrow or None)
    return shortest_arrow

  def is_digraph_simple(self):
    '''
    Returns whether the digraph is simple.
    
    A digraph is simple if for any source and target vertices there are
    0 or 1 arrows between them [in the given direction].
    
    (A digraph which is not simple is called multidigraph.)
    '''
    # Easiest way: observe repetitions on arrows leaving each possible source vertex
    # Use set() and length on self.get_neighbors_out()
    # Let's call this event multiarrows (same source and target, multiple arrows)
    have_multiarrows_been_found = False
    for source in self.get_vertices():
      neighbors_out = self.get_neighbors_out(source, skip_checks = True)
      if len(neighbors_out) != len(set(neighbors_out)):
        # In this case there are multiarrows
        have_multiarrows_been_found = True
        # One is enough
        break
    return (not have_multiarrows_been_found)   

  def is_digraph_multidigraph(self):
    '''
    Returns whether the digraph is a multidigraph, that is, non simple.
    '''
    return (not self.is_digraph_simple())
    
  def get_reversed_graph(self, modify_self = False):
    '''
    Reverses (In the sense of switching direction) all arrows of the digraph.
    
    Can modify self [and return None] or return a new instance.
    '''
    if isinstance(self, Graph):
      # If undirected (i. e. Graph) it doesn't alter itself
      # We could try to reverse the order of the vertices on the edges,
      #just to do work. But in practice, it would be useless.
      if modify_self:
        # Do nothing
        return None
      else:
        return copy_copy(self)
    else:
      # Easiest and cleanest way is through __init__
      all_reversed_arrows = OperationsVAE.get_reversed_arrows_or_equivalent_edges(
          self.get_arrows(),
          use_edges_instead_of_arrows = False,
          require_namedtuple = False,
          request_vertex_sanitization = False,
          require_vertex_namedtuple = False,
          output_as_generator = False)
      data = (self.get_vertices(), all_reversed_arrows)
      data_type = 'all_vertices_and_all_arrows'
      if modify_self:
        # Use __init__ to reset and re-initialize
        self.__init__(
            data = data,
            data_type = data_type,
            cast_as_class = None)
        return None
      else:
        # Return new instance
        return self.__class__(
            data = data,
            data_type = data_type,
            cast_as_class = None)

########################################################################
# Methods implementing different algorithms in Graph Theory
########################################################################

  def get_sccs(self):
    '''
    Returns the strongly connected components (SCCs) of the digraph.
    
    INPUT:
    self
    
    OUTPUT:
    sccs: a list with lists, each representing a unique connected component
    sccs_lengths: the amount of vertices in each of the output SCCs
    '''
    # Note the arrow weights are ignored. Nonetheless, we keep them
    # We control everything using a StateDigraphGetSCC instance
    state = StateDigraphGetSCC(self)
    # We also need the inverse/reversed graph
    inverted_graph = self.get_reversed_graph()
    # We do DFS-Loop using the inverted graph to get a new rank
    state.manually_change_graph(inverted_graph)
    new_rank, middle_leaders = state.dfs_outer_loop()
    # We do DFS-Loop using the original graph, self, and the new_rank
    # Note that the leaders don't matter in the first pass
    # (Neither does the final_rank on the second pass.)
    state.manually_change_vertices_ranked(new_rank)
    state.manually_change_graph(self)
    final_rank, final_leaders = state.dfs_outer_loop()
    # Now the leaders give the SCCs. We will store the results in a dict
    # The leader is the key, the value a list of those having it as a leader
    # (That includes itself)
    # Note many will be keys with empty lists as values, which is okay
    state.manually_change_vertices_ranked(final_rank) # Optional
    dict_sccs = {vertex: [] for vertex in state._vertices_ranked}
    for vertex in state._vertices_ranked:
      dict_sccs[final_leaders[vertex]].append(vertex)
    # We don't return the leaders, only the SCCs and the lengths.
    # Note that logically we should only output nonempty lists following leaders
    sccs = [scc for scc in dict_sccs.values() if scc]
    sccs_lengths = [len(scc) for scc in sccs]
    return (sccs, sccs_lengths)

  def solve_two_sat_problem(self, antipode_function, skip_checks = False):
    '''
    
    '''
    if not skip_checks:
      # ValueError will be raise here if not a valid digraph
      self.validate_digraph_might_come_from_two_sat_problem(antipode_function)
    # We envelop antipode function to work on vertices
    antipode_vertex_function = lambda vertex: Vertex(name=antipode_function(vertex.name))


  def validate_digraph_might_come_from_two_sat_problem(self, antipode_function):
    '''
    Detects whether digraph might come from 2-SAT problem. If it might, does nothing.
    If not, raises a ValueError with a clarifying message.
    '''
    # We envelop antipode function to work on vertices
    antipode_vertex_function = lambda vertex: Vertex(name=antipode_function(vertex.name))
    try:
      # Easier test: need even number of vertices and edges, as explained below
      assert self.get_number_of_vertices() % 2 == 0, 'Need even number of vertices'
      assert self.get_number_of_arrows() % 2 == 0, 'Need even number of arrows'
      # Antipode function must be reflexive (to represent variables and negations)
      # We cannot test it on all values in the universe, so we at least
      #test on the names of the vertices
      list_of_vertices = list(self.get_vertices())
      while list_of_vertices:
        vertex = list_of_vertices.pop():
        try:
          antipode_vertex = antipode_vertex_function(vertex)
        except:
          raise AssertionError('Antipode function fails to generate another vertex')
        assert antipode_vertex in list_of_vertices, 'Negation of variable need to be in digraph'
        assert vertex != antipode_vertex, 'Negation of variable has to differ from variable'
        assert vertex == antipode_vertex_function(antipode_vertex), 'Antipode has to be reflexive'
        # Remove it, and continue loop
        list_of_vertices.remove(antipode_vertex)
      # By its nature, to any arrow from x to y there is another from ~y to ~x
      # This is called the contrapositive
      list_of_arrows = list(self.get_arrows())
      while list_of_arrows:
        arrow = list_of_arrows.pop():
        # Arrows must be unweighted, both original and contrapositive
        assert arrow.weight is None, 'Arrows should be unweighted'
        # Operations below certainly work; tested while testing on vertices
        antipode_source = antipode_function_vertex(arrow.source)
        antipode_target = antipode_function_vertex(arrow.target)
        contrapositive_arrow = (antipode_target, antipode_source, None)
        assert contrapositive_arrow in list_of_arrows, 'Contrapositive arrow should be present in digraph'
        # Remove it and continue loop
        list_of_arrows.remove(contrapositive_arrow)
    except AssertionError as e:
      # Raises ValueError with same message
      raise ValueError(str(e))
    # This is a validation process and should return nothing if tests are passed
    return None

  # Should be run (n**2)*log(n) times with different seeds
  def find_almost_certainly_minimal_cut(self, tries = None):
    '''
    Returns the best cut among many tries, hopefully a minimal cut.
    '''
    # Problem only makes sense with at least two vertices
    n = self.get_number_of_vertices()
    if n <= 1:
      raise ValueError('graph needs at least 2 vertices')
    if tries is None:
      # If no parameter is given, we default to (n**2)*log(n)
      tries = round((n**2)*math_log2(n)) # round() produces an int
    minimal_cut = None
    for try_idx in range(tries):
      # Note find_cut(array) is (crossing_edges, one_side, other_side)
      cut_from_try = self.find_cut()
      if minimal_cut is None or cut_from_try[0] < minimal_cut:
        minimal_cut = cut_from_try[0]
        print('Current try: {}. New minimum reached: {}'.format(try_idx, minimal_cut))
    return minimal_cut

  def k_clustering(self, k):
    '''
    Finds the optimal k-clustering (k >= 1) of the graph.
    
    Requires a complete, weighted graph.
    
    Note: for k = 1 it produces a minimum spanning tree vias Kruskal's algorithm.
    '''
    # We need a weighted, undirected, simple [i.e. non-multigraph] graph
    # Being undirected, we have access to self._edges
    assert self.is_digraph_undirected(), 'Need undirected graph'
    assert self.is_digraph_weighted(), 'Need weighted graph'
    assert self.is_digraph_simple(), 'Need simple graph (cannot be multigraph)'
    # (We don't really require complete. When the graph is not complete,
    #this reduces to the Kruskal algorithm, essentially)
    n = self.get_number_of_vertices()
    assert n >= k, 'Need at least k starting vertices to form k clusters'
    # First we start up the clusters using a union-find structure
    # By cluster we mean: each vertex will have a leader, and vertices
    #of same leader belong to the same cluster
    # But we do lazy union, so we have a parent relation, and we need to
    #transverse it up to finder the leader
    # Also, we do path compression, so we update one's parents to be
    #one's leaders when given the opportunity
    parents = {vertex:vertex for vertex in self.get_vertices()}
    ranks = {vertex:0 for vertex in self.get_vertices()}
    # We put all edges in a heap. We order them by weight, reordering it
    edges_heap = [(edge.weight, edge.first, edge.second) for edge in self._edges]
    heapq_heapify(edges_heap)
    # We need to do n-k union-operations
    # But we later do one special operation, which is part of the main loop
    # So we really do n-k+1, interrupting one
    for idx in range(n-k+1):
      # Locate the smallest edge which is a bridge between two clusters
      while True:
        # If graph is not complete there might be an error in the following
        # Nonetheless, we don't want the try/except overhead for exceptions
        new_edge = heapq_heappop(edges_heap)
        # Call the vertices u and v. Recall the order of the information
        weight, u, v = new_edge
        # Get the leaders of u and v. This is a find-operation
        # (Do path-compression while at it)
        local_leaders = {item:None for item in [u, v]}
        for vertex in [u, v]:
          # We save the path to do path compression
          # Idea is to keep appending the parents until leader is found
          accumulated_path = [vertex]
          # We loop whiel the leader of the root is not found
          while accumulated_path[-1] != parents[accumulated_path[-1]]:
            # We don't have a leader yet, so we append the parent to the path
            accumulated_path.append(parents[accumulated_path[-1]])
          # Ok, now we have a full path to the leader in accumulated_path
          # First we do path-compression
          for item in accumulated_path:
            parents[item] = accumulated_path[-1]       
          # We save the result as local_leaders dict, and break
          local_leaders[vertex] = parents[vertex]
        # Ok, now we have local_leaders[u] and local_leaders[v]
        # If they are in the same cluster, we discard the edge and try again
        # Otherwise we continue with the process
        if local_leaders[u] != local_leaders[v]:
          break
      # We found a good sparating edge.
      # Now we do the union-operation, except in the last operation
      #in which we compute a minimal separation between the clusters
      # And don't proceed, otherwise we would over-cluster the vertices
      if idx == n-k:
        minimal_distance_clusters = weight
      else:
        # ok, we are still in the process of clustering. So we do an union
        # We compare the ranks of the leaders.
        if ranks[local_leaders[u]] == ranks[local_leaders[v]]:
          # If equal, we add one tree to the other in O(1) operations
          # Without loss of generality, let's say local_leaders[u] will lead
          parents[local_leaders[v]] == local_leaders[u]
          # We also adjust the rank of local_leaders[u]
          ranks[local_leaders[u]] += 1
        elif ranks[local_leaders[u]] < ranks[local_leaders[v]]:
          # If different, the smaller/shallower tree is appended to the larger
          parents[local_leaders[u]] = local_leaders[v]
        else:
          parents[local_leaders[v]] = local_leaders[u]
    # Ok. Not the loop has finalized and we have k clusters (given by parents)
    #as well as a last execution which givs the minimal distance
    # We want to output parents (which indirectly give the clusters)
    # But we also output the objective distance, the minimal possible distance
    #between two points in different clusters
    return (parents, minimal_distance_clusters)

########################################################################
# Class WeightedDigraph
########################################################################

class WeightedDigraph(Digraph):
  '''
  A digraph whose arrows are all weighted.
  '''

  def are_weights_nonnegative(self, require_weights_positive = False):
    '''
    For a weighted digraph, returns whether all weights are negative.
    
    Has an option to require weights to be positive and not only nonnegative.
    '''
    # Read straight from arrows
    if require_weights_positive:
      answer = all(arrow.weight > 0 for arrow in self.get_arrows())
    else:
      answer = all(arrow.weight >= 0 for arrow in self.get_arrows())
    return answer

  def make_digraph_unweighted(self, modify_self = False):
    '''
    Creates an unweighted digraph (by removing weights from all arrows).
    
    Can either modify self or return a new instance.
    '''
    # We first determine if we are in an instance of Graph
    # If we are, not only we want our future class to be also a Graph,
    #but we can also greatly shorten __init__ by using edges
    if isinstance(self, Graph):
      selected_class = UnweightedGraph
      data_type = 'all_vertices_and_all_edges'
      use_edges_instead_of_arrows = True
      original_working_data = self.get_edges()
    else:
      selected_class = UnweightedDigraph
      data_type = 'all_vertices_and_all_arrows'
      use_edges_instead_of_arrows = False
      original_working_data = self.get_arrows()
    # We then create the unweighted arrows/edges
    list_new_tuplees = OperationsVAE.remove_weight_from_arrows_or_edges(
        tuplees = original_working_data,
        use_edges_instead_of_arrows = use_edges_instead_of_arrows,
        require_namedtuple = True, output_as_generator = False)
    # We need to keep track of the vertices too, lest they are isolated
    list_new_vertices = self.get_vertices()
    # We prepare for the return. This works the same way whether we had
    #arrows or we had edges
    data = (list_new_vertices, list_new_tuplees)
    # We create a new instance or modify the current
    if modify_self:
      self.__init__(data = data, data_type = data_type,
          cast_as_class = selected_class)
      return None
    else:
      return selected_class(data = data, data_type = data_type,
          cast_as_class = selected_class)

  def get_single_source_shortest_path_via_Dijkstras(self, base_vertex):
    '''
    In a weighted digraph, produces the distances from a fixed source vertex
    to all others transversing the available arrows.
    
    Uses Djikstras's Algorithm to compute the distances.
    
    Weights must be nonnegative for the algorithm to work.
    '''
    # We need all weights to be nonnegative.
    assert self.are_weights_nonnegative(), 'Need weights to be nonnegative'
    # We also need base_vertex to be one of the vertices from the self
    assert self.belongs_to_as_vertex_after_sanitization(base_vertex, require_vertex_namedtuple = False), 'Base vertex must be in the graph'
    # We use the algorithm from class.
    # We implement set_X and set_Y (set_X with vertices whose shortest paths have been computed)
    # set_Y starts as all vertives (no shortest paths computed), and set_X as empty
    # Each time one shortest path is computed, we remove from set_Y and add to set_X
    set_Y = self.get_vertices()
    set_X = []
    # Shortest paths are available as a dictionary (None if not yet computed)
    # Note that these are shortest paths to base_vertex, and thus will not be attributes
    distances = {vertex: None for vertex in self.get_vertices()}
    # Okay, we know shortest path to is base_vertex is 0
    distances[base_vertex] = 0
    set_X.append(base_vertex)
    set_Y.remove(base_vertex)
    # Now it's time to implement algorithm
    # Each "loop of the while" correspond to one vertex being added
    # If they don't connect (there is no path from base vertex to the vertex in question),
    #the distance of that vertex [to the base vertex] will be set to math_inf
    # We could use of the following upper bound for distance: sum of all weights, plus one
    # But we prefer to use inf from built-in math library because it makes it
    #simpler to tell if a vertex cannot be reached from the base_vertex
    upper_bound_distance = math_inf
    # set_Y should contain, right now all vertices except base_vertex
    while set_Y:
      # current_min_distance controls the minimum distance from X to Y this round
      #(This is updated every loop because X and Y keep changing)
      current_min_distance = upper_bound_distance
      # We also control the point in set X and in set Y to realize current_min_distance
      current_min_set_X = None
      current_min_set_Y = None
      # We scan all arrows from X to Y to see if there is a smaller distance
      for vertex in set_Y:
        # We check the arrows. Since we are indexing by set_Y use get_arrows_in
        # (That is, the vertex in set_Y is the target of the relevant arrows)
        for arrow in self.get_arrows_in(vertex):
          # The arrow only matters if the source is in set_X
          if arrow.source in set_X:
            if distances[arrow.source] + arrow.weight < current_min_distance:
              current_min_distance = distances[arrow.source] + arrow.weight
              current_min_set_X = arrow.source
              current_min_set_Y = vertex
      # Now we check if we found anything less than upper_bound_distance
      # In other words, this tests connection between set_X and set_Y
      if current_min_distance < upper_bound_distance:
        # All we need to do is update the information on the current_min_set_Y
        distances[current_min_set_Y] = current_min_distance
        set_Y.remove(current_min_set_Y)
        set_X.append(current_min_set_Y)        
      else:
        # In this case set_Y and set_X are disjoint
        # So we put distance of set_Y to be math_inf, and put them all in set_X
        # (After all, their distances to the base vertex have been calculated!)
        # This will end the while loop
        for vertex in list(set_Y):
          distances[vertex] = upper_bound_distance
          set_Y.remove(vertex)
          set_X.append(vertex)
    # Here the while loop has ended. All vertices have associated distances
    # So we can output the result, the dict distances
    return distances

  #################
  # WORK HERE
  # Use Vertex.Path to streamline this
  #################
  def get_single_source_shortest_paths_via_Bellman_Fords(self, source_vertex,
      output_as = 'lengths_and_vertices_and_arrows'):
    '''
    Computes the length of shortest paths from one single source vertex
    to all other vertices of a directed weighted graph.
    
    Accomodates negative arrows. [By negative arrows, we mean an arrow
    with negative length/weight.] If there are negative cycles [cycles
    whose total length is negative], there are no shortest paths;
    in this case, the algorithm detects and reports a negative cycle.
    
    If n is the number of vertices and m the number of edges, its time
    complexity is O(m*n).
    
    It is slower than Dijkstra's algorithm (which is O(n*log(n))), but it
    has the advantage of handling arrows of negative weight.
    
    INPUT:
    self: the graph
    source_vertex: compute distances starting from this vertex
    output_as: one of the output possibilities for VertexPath.reformat_path()
    
    OUTPUT:
    is_negative_cycle_free: whether the graph is free of negative-weight cycles
    data: if there are no negative cycles, this is a dictionary which gives,
    for each vertex in the path, the shortest path from source to it, given
    in the requested format. If there are negative cyles, this is a negative
    cycle given by its length, its vertices or arrows.
    '''
    # Check if the vertex is indeed in the graph
    assert self.belongs_to_as_vertex_after_sanitization(source_vertex, require_vertex_namedtuple = False), 'Source vertex must be in the graph'
    # At each moment, each vertex will have a number, A, and a pointer, B
    # The dict A carries information on the shortest length from the source
    # The dict B carries info on the last arrow for this shortest path
    n = self.get_number_of_vertices()
    A = {vertex:math_inf for vertex in self.get_vertices()}
    B = {vertex:None for vertex in self.get_vertices()}
    # We set the information on the source vertex
    A[source_vertex] = 0
    # Both will be updated during the algorithm, multiple times
    # If there is an answer, it is certainly reached after n-1 loops
    # But a n-th loop is needed to check for negative cycles
    might_there_be_negative_cycles = True
    for unused_index in range(n):
      # In each iteration, we reset the loop variable
      vertices_updated_this_loop = []
      # We process all arrows
      for arrow in self.get_arrows():
        # Arrows are given as tuples (source, target, weight)
        # We see if the navigating through the arrow [as last arrow]
        #provides a better result than the current one for the target vertex
        if A[arrow.source] + arrow.weight < A[arrow.target]:
          A[arrow.target] = A[arrow.source] + arrow.weight
          B[arrow.target] = arrow
          vertices_updated_this_loop.append(arrow.target)
      # We have processed all arrows. Stop algorithm if A doesn't change
      # In this case we know there are no negative cycles
      if not vertices_updated_this_loop: # I. e. empty
        might_there_be_negative_cycles = False
        break
    # After the n loops we have the answer to the existence of negative cycles
    is_negative_cycle_free = not might_there_be_negative_cycles
    if is_negative_cycle_free:
      # For each vertex, we produce a pathys from source_vertex to vertex
      # A path will be a list [source, ..., vertex]. We record it using information
      #on length (available from A), vertices and arrows (these derived from B)
      shortest_paths_lengths = dict(A) # A copy to be safe
      if ('vertices' in request.lower()) or ('arrows' in request.lower()):
        # We will need this information as long as the request includes either
        #'arrows' or 'vertices' (and both cost about the same to calculate so
        #we do them together)
        shortest_paths_as_vertices = {}
        shortest_paths_as_arrows = {}
        for vertex in self.get_vertices():
          if A[vertex] is math_inf:
            # Note: if graph is not connected some vertices will have math_inf as distance
            # That is not a problem at all, just that there are no paths
            # We put simply shortest_paths[vertex] = None
            shortest_paths_as_vertices[vertex] = None
            shortest_paths_as_arrows[vertex] = None
          else:
            # In this case we give the sequence of edges
            # (For the length of the minimum path, add up their weights)
            # We only record each arrow in each vertex...which is enough
            shortest_paths_as_vertices[vertex] = [vertex] # Start backward
            shortest_paths_as_arrows[vertex] = []
            # The following will follow the shortest path backward
            local_moving_vertex = vertex
            while local_moving_vertex != source_vertex:
              new_arrow_to_add = B[local_moving_vertex]
              new_vertex_to_add = new_arrow_to_add.source # We go backward
              shortest_paths_as_vertices[vertex] = [new_vertex_to_add] + shortest_paths_as_vertices[vertex]
              shortest_paths_as_arrows[vertex] = [new_arrow_to_add] + shortest_paths_as_arrows[vertex]
              local_moving_vertex = new_vertex_to_add # Closer to source
      # Return with variable name requested_data, as specified in request
      # We are concluding case is_negative_cycle_free == True
      # Form tuples when more than one, and return None if request == 'nothing'
      if request.lower() == 'lengths_and_vertices_and_arrows':
        requested_data = (shortest_paths_lengths, shortest_paths_as_vertices, shortest_paths_as_arrows)
      elif request.lower() == 'lengths_and_vertices':
        requested_data = (shortest_paths_lengths, shortest_paths_as_vertices)
      elif request.lower() == 'lengths_and_arrows':
        requested_data = (shortest_paths_lengths, shortest_paths_as_arrows)
      elif request.lower() == 'vertices_and_arrows':
        requested_data = (shortest_paths_as_vertices, shortest_paths_as_arrows)
      elif request.lower() == 'lengths':
        requested_data = shortest_paths_lengths
      elif request.lower() == 'vertices':
        requested_data = shortest_paths_as_vertices
      elif request.lower() == 'arrows':
        requested_data = shortest_paths_as_arrows
      elif request.lower() == 'nothing':
        requested_data = None
      else:
        raise ValueError('request option not recognized')
    else:
      # In this case we return a negative cycle (as a list of its edges)
      # We know that if any vertex updated in the last loop is part of a
      #negative cycle. So we can take vertices_updated_this_loop[0], p. ex.,
      #and then write a full cycle with it
      # (Note: we need to do this iteration outside the loop for the occasion
      #of an arrow of a vertex into itself)
      # (This is only needed once so we do even if not requested)
      negative_cycle_starting_vertex = vertices_updated_this_loop[0]
      negative_cycle_example_length = A[negative_cycle_starting_vertex]
      negative_cycle_example_as_vertices = [negative_cycle_starting_vertex] # Vertices here
      negative_cycle_example_as_arrows = [] # Put arrows here
      first_arrow_to_append = B[negative_cycle_starting_vertex]
      first_vertex_to_append = first_arrow_to_append.source # Go backwards
      negative_cycle_example_as_vertices = [first_vertex_to_append] + negative_cycle_example_as_vertices
      negative_cycle_example_as_arrows = [first_arrow_to_append] + negative_cycle_example_as_arrows
      # We use next_vertice_in_cycle to keep track of the closure of the cycle
      # Note we say next when we really mean previous... just a matter of names
      next_vertex_in_cycle = first_vertex_to_append # Go backward
      #while next_vertice_in_cycle != negative_cycle_starting_vertex:
      while next_vertex_in_cycle not in negative_cycle_example_as_vertices:
        arrow_to_append = B[next_vertice_in_cycle]
        vertex_to_append = arrow_to_append.source # Going backward
        negative_cycle_example_as_vertices = [vertex_to_append] + negative_cycle_example_as_vertices
        negative_cycle_example_as_arrows = [arrow_to_append] + negative_cycle_example_as_arrows # Go backwards
        next_vertex_in_cycle = vertex_to_append # Go backwards
        ########################
        # WORK HERE
        # Infinite recursion here... maybe it fell into a sub-cycle?
        # In this case we should verify the next vertex isn't already in the list
        ########################
      # When done, prepare data. We are in case is_negative_cycle_free == False
      if request.lower() == 'lengths_and_vertices_and_arrows':
        requested_data = (negative_cycle_example_length, negative_cycle_example_as_vertices,
            negative_cycle_example_as_arrows)
      elif request.lower() == 'lengths_and_vertices':
        requested_data = (negative_cycle_example_length, negative_cycle_example_as_vertices)
      elif request.lower() == 'lengths_and_arrows':
        requested_data = (negative_cycle_example_length, negative_cycle_example_as_arrows)
      elif request.lower() == 'vertices_and_arrows':
        requested_data = (shortest_paths_as_vertices, negative_cycle_example_as_arrows)
      elif request.lower() == 'lengths':
        requested_data = negative_cycle_example_length
      elif request.lower() == 'vertices':
        requested_data = negative_cycle_example_as_vertices
      elif request.lower() == 'arrows':
        requested_data = negative_cycle_example_as_arrows
      elif request.lower() == 'nothing':
        requested_data = None
      else:
        raise ValueError('request option not recognized')
    # Now we have formed requested_data for any value of is_negative_cycle_free
    return (is_negative_cycle_free, requested_data)

  def get_all_paths_shortest_paths_via_Floyd_Warshals(self, request = 'lengths'):
    '''
    Computes the shortest paths for every pair of vertices in a directed,
    weighted graph. (Arrow weights might be negative.)
    
    When there is no path, infinity is returned as minimal distance.
    
    If there are negative cycles, there are pairs with no shortest paths,
    but the algorithm detects the situation and outputs a negative cycle.
    
    If n is the number of vertices, the time complexity of the algorithm is O(n^3).
    
    INPUT:
    self: the graph
    request: 'lengths', 'lengths_and_vertices' or 'lengths_and_arrows'
    
    OUTPUT:
    is_negative_cycle_free: whether the graph has a negative cycle
    requested_data:
    '''
    # We define a function to build the optimal path between two vertices
    # This sub-function will be useful towards the end of this method
    # (By default we don't provide the edge as it's costly to search for them)
    def get_shortest_path_using_A_and_B(first, second, request = 'arrows'):
      '''
      Returns the shortest path from a vertex to another, either as a list
      of vertices or as a list of arrows (considerably slower).
      
      Requires A and B (of the method where function is inserted) computed.
      
      INPUT:
      first: the source vertex of the shortest path we want to find
      second: the target vertex of the shortest path we want to find
      request: 'vertices', 'arrows' or 'vertices_and_arrows'
      
      OUTPUT:
      data: depending on request, the shortest path between the vertices
      given given by vertices, given by arrows, or a tuple with it given by
      vertices and arrows
      '''
      assert 'A' in vars(), 'Need variable A available'
      assert 'B' in vars(), 'Need variable B available'
      # First we check if the vertices are not connected at all
      if A[((first, second))] == math_inf:
        # In this case no path, which we return as None (in all request options)
        if request.lower() == 'vertices':
          return None
        elif request.lower() == 'arrows':
          return None
        elif request.lower() == 'vertices_and_arrows':
          return (None, None)
        else:
          raise ValueError('Invalid choice for request')
      elif first == second and B[(first, second)] == None:
        # Now we check if a path is the trivial path (no arrows)
        # Not really a path, but we return a list with the vertex
        #for request == 'vertices', and otherwise an empty list of arrows
        if request.lower() == 'vertices':
          return [first]
        elif request.lower() == 'arrows':
          return []
        elif request.lower() == 'vertices_and_arrows':
          return ([first], [])
        else:
          raise ValueError('Invalid choice for request')
      else:
        # In this case the shortest path is a single or multiple arrows
        # (We can handle all of it in this case)
        current_path = [first, second]
        # Now, at any time in the current_path, if there is no
        #direct edge realizing the shortest distance between two consecutive
        #vertices [possibility detected by B]
        # Perhaps the simples way is using a "while" loop
        # (Note: this case generalizes the previous, but this is not a problem)
        # We use the following variable to know when to exit:
        # When all gaps are 0, there are arrows between any consecutive vertices
        #which can then be found by another method
        inserted_new_vertices_for_examination = True
        # This is also used to control whether there would be repetitions
        # (This can happen for negative cycles... because repeating them
        #always yields paths of shorter and shorted length)
        # We would prefer to have the simples negative cycle in this case,
        #i.e, without repetition of internal vertices
        # Thus, we use inserted_new_vertices_for_examination
        while inserted_new_vertices_for_examination:
          # Unless new vertices are inserted, we turn the variable to False
          inserted_new_vertices_for_examination = False
          # We compute B for every consecutive pair in the list
          current_gaps = []
          for idx in range(len(current_path) - 1):
            current_gaps.append(B[current_path[idx], current_path[idx+1]])
          # If all gaps are 0 then we have all direct edges
          # This will be evaluated during the following loop
          for idx, vertex in enumerate(current_path[:-1]):
            if current_gaps[idx] != 0:
              # Take the lowest occurrence, and replace by the appropriate vertex
              # Add the intermediate vertex to current_path
              # Recall we subtract 1 because with loop_index we consider
              #the vertex whose number is loop_index - 1
              number_to_add = B[current_path[idx], current_path[idx+1]] - 1
              vertex_to_add = number_to_vertex[number_to_add]
              # Due to the fact that we might get an infinite loop for a negative cycle
              #we will forbid vertex_to_add to being already in the path
              # (The source and target might still be the same point.)
              if not vertex_to_add in current_path:
                current_path = current_path[:idx+1] + [vertex_to_add] + current_path[idx+1:]
                inserted_new_vertices_for_examination = True
                # We break of only this for-loop (since we need to redo the list and everything)
                # (Otherwise it would be changing the list while it is being scanned, which is bad)
                # We will, of course, start again
                break
        # Okay. So we have current_path correct; it cannot be improved
        # Let's output as requested: list of vertices or of arrows
        if request.lower() == 'vertices':
          return current_path
        elif 'arrows' in request.lower():
          # For this case we must produce the shortest arrow between
          #eadch consecutive pair of vertices in current_path
          list_of_shortest_arrows = []
          for idx, vertex in enumerate(current_path[:-1]):
            new_arrow = self.get_shortest_arrow_between_vertices(current_path[idx], current_path[idx+1])
            list_of_shortest_arrows.append(new_arrow)
          if request.lower() == 'arrows':
            return list_of_shortest_arrows
          elif request.lower() == 'vertices_and_arrows':
            return (current_path, list_of_shortest_arrows)
          else:
            raise ValueError('Invalid choice for request')
        else:
          raise ValueError('Invalid choice for request')
    # End of inner function
    # Now we start the method code proper
    # We confirm the graph is weighted
    # (An undirected graph is a specific case of a directed graph)
    assert self.is_weighted_graph(), 'Graph must be weighted'
    # To do less searching on the edges, we prefer a simple graph
    # If there are multiple arrows between vertices, we take the shortest ones
    # (In the future, we might implement this removal of redundant edges)
    # For this we need to have a specific numbering of the vertices
    # The best way is a duple correspondence with two dicts
    n = self.get_number_of_vertices()
    vertex_to_number = {}
    number_to_vertex = {}
    # We don't really need the full apparatus but it's a good-to-have
    for number, vertex in enumerate(self.get_vertices):
      number_to_vertex[number] = vertex
      vertex_to_number[vertex] = number
    # We will loop n+1 times. In each time (loop_index from 0 through n)
    #we consider only paths whose interior vertices are among the loop_index first ones
    # We continuously keep track, for each pair of vertices, of dicts A and B
    # Dict A keeps track of the shortest path between from first to second
    # Dict B keeps track of one of the vertices (given by its number)
    all_keys = [(first, second) for first in self.get_vertices() for second in self.get_vertices()]
    A = {key:math_inf for key in all_keys}
    B = {key:None for key in all_keys}
    # We want to stop the iterations the moment we find the first negative cycle
    # (There could be more... we only care about providing an example, if there is one)
    have_found_negative_cycle = False
    for loop_index in range(n+1):
      if have_found_negative_cycle:
        # Stop, no need to keep iterating
        break 
      # Possible interior vertices for this iteration:
      # If loop_index is 0, an empty list. Otherwise, the list has loop_index elements
      possible_interior_numbers = list(range(0, loop_index))
      possible_interior_vertices = [number_to_vertex[number] for number in range(0,loop_index)]
      # We will do loop_index being 0 separately, since it is a bit different
      if loop_index == 0:
        # To control the existence of negative self-arrows
        negative_self_arrow_example = None
        # In this case the possible paths are the existing arrows
        # We also include the "null path", from each vertex to itself, weight 0
        trivial_paths = [(vertex, vertex, 0) for vertex in self.get_vertices()]
        for source, target, weight in list(self._arrows)+trivial_paths:
          # Note that self-arrows are one source of negative cycles
          # (That is, if the self-arrow has negative weight, it forms one)
          if source == target and weight < 0:
            have_found_negative_cycle = True
            is_negative_cycle_self_arrow = True
            negative_self_arrow_example = (source, target, weight)
            # We only need to produce one negative cycle, so we break off this inner loop
            break
          if weight < A[source, target]:
            A[(source, target)] = weight
            # We record the loop_index
            B[(source, target)] = loop_index # In this case 0
      else:
        # This is for loop_index being 1 through n
        # This step is characterized by the fact that a new vertex can
        #now be used as interior vertex.
        for first, second in all_keys:
          # We check if this new vertex provides a shortcut
          # In this case (and only this case) we update A and B for the pair
          newest_vertex = number_to_vertex[loop_index - 1]
          if A[(first, newest_vertex)] + A[(newest_vertex, second)] < A[(first, second)]:
            A[(first, second)] = A[(first, newest_vertex)] + A[(newest_vertex, second)]
            B[(first, second)] = loop_index # Corresponding vertex is loop_index - 1
        # We now want to trigger have_found_negative_cycle the earlier possible
        # To do so we look at the diagonal
        # (This testing does cost CPU time but overall saves time for breaking earlier)
        for vertex in self.get_vertices():
          if A[(vertex, vertex)] < 0:
            # We found a negative cycle, but it is not a self-arrow
            have_found_negative_cycle = True
            is_negative_cycle_self_arrow = False
            # Note that vertex is part of a negative cycle (with more vertices
            #than only itself)
            vertex_in_negative_cycle = vertex
            # One is enough; break out of this for loop
            break        
    # Okay, so now we have A and B ready and correct
    # (If we broke out earlier due to have_found_negative_cycle,
    #they are not optimal but should be enough to correctly output a negative cycle)
    # We heva also determined with 100% certainty whether there is or
    #there isn't a negative cycle. We can express with another variable too
    #(to denote our learning of the property of having or not having such cycles)
    is_negative_cycle_free = not have_found_negative_cycle
    # We first deal with the case when there are negative cycles
    if have_found_negative_cycle: # Equivalent to not is_negative_cycle_free
      if is_negative_cycle_self_arrow:
        # We have recorded a negative-weight self-arrow as example
        # Preparing this data is cheap, and so we prepare everything
        negative_cycle_example_weight = negative_self_arrow_example[2]
        negative_cycle_example_as_arrows = [negative_self_arrow_example]
        involved_vertex = negative_self_arrow_example[0] # Same as in position 1
        negative_cycle_example_as_vertices = [involved_vertex, involved_vertex]
      else:
        # That is, is_negative_cycle_self_arrow is False
        # In this case we have a cycle with more than one vertex
        #(and one of those vertices is given by vertex_in_negative_cycle)
        negative_cycle_example_weight = A[(vertex_in_negative_cycle, vertex_in_negative_cycle)]
        # Since it is a single cycle for the whole function, we don't mind
        #producing arrows and vertices even if they are not requested
        as_vertices_and_arrows = get_shortest_path_using_A_and_B(
              vertex_in_negative_cycle, vertex_in_negative_cycle, request = 'vertices_and_arrows')
        negative_cycle_example_as_vertices = as_vertices_and_arrows[0]
        negative_cycle_example_as_arrows = as_vertices_and_arrows[1]
    else:
      # Now we are in the case without negative cycles
      # We produce the shortest paths for every pair of vertices
      # Paths can be given as their weight/length, their arrows or their vertices
      # This info will be stored in dicts
      # (We only really compute the ones requested as output)
      all_shortest_paths_weights = {}
      all_shortest_paths_as_arrows = {}
      all_shortest_paths_as_vertices = {}
      for pair in all_keys:
        # (We know every path has a 0-long path to itself, any cycle will
        #have positive weight. But, to output a more complete dictionary,
        #we will put default values, exactly as specified by sub-function
        # (I. e. we don't need a separate routine for first == second)
        all_shortest_paths_weights[pair] = A[pair]
        if 'vertices' in request.lower():
          if 'arrows' in request.lower():
            as_vertices_and_arrows = get_shortest_path_using_A_and_B(
                *pair, request = 'vertices_and_arrows')
            all_shortest_paths_as_vertices[pair] = as_vertices_and_arrows[0]
            all_shortest_paths_as_arrows[pair] = as_vertices_and_arrows[1]
          else:
            all_shortest_paths_as_vertices[pair] = get_shortest_path_using_A_and_B(
                *pair, request = 'vertices')
        else:
          if 'arrows' in request.lower():
            all_shortest_paths_as_arrows[pair] = get_shortest_path_using_A_and_B(
                *pair, request = 'arrows')
    # We prepare the data before the final step!!!
    if request.lower() == 'lenghts':
      if is_negative_cycle_free:
        # The dict with all the weights
        requested_data = all_shortest_paths_weights
      else:
        # The weight of the negative cycle example
        requested_data = negative_cycle_example_weight
    elif request.lower() == 'vertices':
      if is_negative_cycle_free:
        # The dict of the vertices forming the shortest paths
        requested_data = all_shortest_paths_as_vertices
      else:
        # The vertices forming the negative cycle example
        requested_data = negative_cycle_example_as_vertices
    elif request.lower() == 'arrows':
      if is_negative_cycle_free:
        # The dict of the arrows forming the shortest paths
        requested_data = all_shortest_paths_as_arrows
      else:
        requested_data = negative_cycle_example_as_arrows
    elif request.lower() == 'lengths_and_vertices':
      if is_negative_cycle_free:
        # Tuple with dict of weights and of vertices
        requested_data = (all_shortest_paths_weights, all_shortest_paths_as_vertices)
      else:
        requested_data = (negative_cycle_example_weight, negative_cycle_example_as_vertices)
    elif request.lower() == 'lengths_and_arrows':
      # Tuple with dict of weights and of arrows
      if is_negative_cycle_free:
        requested_data = (all_shortest_paths_weights, all_shortest_paths_as_arrows)
      else:
        requested_data = (negative_cycle_example_weight, negative_cycle_example_as_arrows)
    elif request.lower() == 'vertices_and_arrows':
      # Tuple with dict of vertices and of arrows
      if is_negative_cycle_free:
        requested_data = (all_shortest_paths_as_vertices, all_shortest_paths_as_arrows)
      else:
        requested_data = (negative_cycle_example_as_vertices, negative_cycle_example_as_arrows)
    elif request.lower() == 'lengths_and_vertices_and_arrows':
      # Tuple with dict of weights, of vertices and of arrows
      if is_negative_cycle_free:
        requested_data = (all_shortest_paths_weights,
            all_shortest_paths_as_vertices, all_shortest_paths_as_arrows)
      else:
        requested_data = (negative_cycle_example_weight,
            negative_cycle_example_as_vertices, negative_cycle_example_as_arrows)
    else:
      raise ValueError('Request option not recognized.')
    # We finally return everything
    ###################
    # WORK HERE
    # Bug: Currently algorithm may fail to produce correct requested_data
    #if is_negative_cycle_free == False. Only remedy is to not output the negative cycle
    #but only announce its existence
    ###################
    if not is_negative_cycle_free:
      requested_data = 'There is a negative cycle.'
    return (is_negative_cycle_free, requested_data)

  def solve_traveling_salesman_problem(self, compute_path_instead_of_cycle,
      initial_vertex = None, final_vertex = None, use_memoization_instead_of_tabulation = False,
      output_as = None, skip_checks = False):
    '''
    Solves the Traveling Salesman Problem. That is, produces the shortest
    (by sum of weights of traveled arrows) path or cycle going through all
    vertices of the Digraph.
    
    If no such path or cycle exists, returns None.
    
    The option use_memoization_instead_of_tabulation can be used according to
    what the user believe it might be best for the specific problem; being
    True or False does not alter the result.
    '''
    # We create an instance of the specialized class StateDigraphSolveTSP
    specialized_object = StateDigraphSolveTSP(self)
    # We unload all the work to its method solve_full_problem
    return specialized_object.solve_full_problem(
        compute_path_instead_of_cycle = compute_path_instead_of_cycle,
        initial_vertex = initial_vertex,
        final_vertex = final_vertex,
        use_memoization_instead_of_tabulation = use_memoization_instead_of_tabulation,
        output_as = output_as,
        skip_checks = skip_checks)

  def produce_nearest_neighbor_heuristics_for_traveling_salesman_problem(self,
      compute_path_instead_of_cycle, initial_vertex = None, final_vertex = None,
      output_as = None, skip_checks = False):
    '''
    
    
    SEE ALSO: solve_traveling_salesman_problem
    '''
    # We mostly mirror solve_traveling_salesman_problem(), unloading the work
    #to the specialized class StateDigraphSolveTSP
    specialized_object = StateDigraphSolveTSP(self)
    return specialized_object.solve_heuristically_using_nearest_neighbors_strategy(
        compute_path_instead_of_cycle = compute_path_instead_of_cycle,
        initial_vertex = initial_vertex,
        final_vertex = final_vertex,
        output_as = output_as,
        skip_checks = skip_checks)
    

########################################################################
# Class UnweightedDigraph
########################################################################

class UnweightedDigraph(Digraph):
  '''
  A digraph whose arrows are all unweighted.
  '''
  
  def make_digraph_weighted(self, modify_self = False):
    '''
    Creates a new digraph, one where all arrows have weight 1.
    
    Can either modify self or return a new instance.
    '''
    # We detect if we are on a Graph. If we are, we use its edges instead
    if isinstance(self, Graph):
      selected_class = UnweightedGraph
      data_type = 'all_vertices_and_all_edges'
      use_edges_instead_of_arrows = True
      original_working_data = self.get_edges()
    else:
      selected_class = UnweightedDigraph
      data_type = 'all_vertices_and_all_arrows'
      use_edges_instead_of_arrows = False
      original_working_data = self.get_arrows()
    # We create the weighted arrows. Note default new weight is 1
    # Easiest way to make them all 1 is no put new_weights = None
    list_new_tuplees = OperationsVAE.write_weights_into_arrows_or_edges(
        tuplees = original_working_data,
        use_edges_instead_of_arrows = use_edges_instead_of_arrows, new_weights = None,
        require_namedtuple = True, output_as_generator = False)
    # We need to keep track of the vertices too, lest they are isolated
    list_new_vertices = self.get_vertices()
    # We prepare for the return
    data = (list_new_vertices, list_new_tuplees)
    # We create a new instance or modify the current
    if modify_self:
      # Easiest way to reset digraph information is with __init__
      self.__init__(data = data, data_type = data_type,
          cast_as_class = selected_class)
      return None
    else:
      return selected_class(data = data, data_type = data_type,
          cast_as_class = selected_class)

  def create_digraph_with_super_source(self):
    '''
    From a weighted digraph, creates a new weighted digraph which has
    all same vertices and arrows but it has one extra vertex, the "super source",
    and arrows of weight 0 from it to all other vertices.
    
    Returns another instance of the class, as well as the new vertex.
    
    This construction is needed for Johnson's algorithm.
    '''
    # First we create a name for the "super source" vertex
    name_super_source = 'super_source'
    # If already, for some reason, a vertex, we add s's to the start
    # This should not be a problem, but it won't hurt
    while name_super_source in [vertex.name for vertex in self.get_vertices()]:
      name_super_source = 's'+name_super_source
      # We consolidate it into a Vertex
      the_super_source = OperationsVAE.sanitize_vertex(name_super_source,
          require_vertex_namedtuple = False)
    # Now we create the new arrows starting from our super source
    new_arrows = []
    for vertex in self.get_vertices():
      new_arrows.append((the_super_source, vertex, 0))
    # We create a new graph with all those new arrows
    # (Note that if self is empty, this won't produce the desired effect.
    # Because of that, we will make it a separate case)
    if not self.is_nonempty_digraph():
      # Best way to avoid name confusion after importing is to use
      #self.__class__ to denote the very class of the instance
      # We create by specifying it the "neighbors way"
      selected_class = type(self)
      new_digraph = selected_class(data = [[the_super_source]], data_type = 'neighbors_out',
          cast_as_class = None)
      return (the_super_source, new_digraph)
    else:
      # In this case we join by the arrows
      all_arrows = new_arrows + self.get_arrows()
      selected_class = type(self)
      new_digraph = selected_class(data = all_arrows, data_type = 'weighted_arrows',
          cast_as_class = None)
      return (the_super_source, new_digraph)
    
  def get_all_paths_shortest_paths_via_Johnsons(self, request = 'lengths'):
    '''
    Computes the shortest paths for every pair of vertices in graph.
    
    The graph is assumed to be a directed, weighted graph. The weights
    of the arrows can be negative.
    
    For a graph with n vertices, consists of one application of Bellman-Fords
    algorithm using an extended graph with a "super source" extra vertex,
    then n applications of Dijkstra's algorithm.
    
    If m is the number of arrows, the time complexity is O(n^2*lon(n) + m*n). 
    '''
    # We first check we have a weighted graph
    assert self.is_weighted_graph(), 'Graph needs to be weighted'
    # According to the algorithm, we need to execute Bellman-Fords on
    #the graph with an extra vertex (often called the "super source")
    # the_super_source is assumed to be a Vertex namedtuple in super_source_graph
    the_super_source, super_source_graph = self.create_digraph_with_super_source()
    ###############
    # WORK HERE
    ###############
    return super_source_graph.get_single_source_shortest_paths_via_Bellman_Fords(the_super_source)

  def get_hamiltonian_cycle(self, initial_vertex = None,
      use_memoization_instead_of_tabulation = False, output_as = None,
      skip_checks = False):
    '''
    Returns a Hamiltonian cycle of the unweighted digraph. That is, any
    cycle that travels through all vertices through the available arrows.
    
    If no such cycle exists, returns None.
    
    Accomodates a request for the cycle to start at a specific vertex,
    even if it matter not in a cycle.
    
    SEE ALSO: get_hamiltonian_path
    '''
    # Default output is Cycle (meaning VertexCycle class). Can also output
    #arrows and vertices
    # A Hamiltonian path is a solution to the Traveling Salesman Problem
    #under the condition that all edges have weight/length 1 (under the TSP
    #perspective, all cycles will have the same length; in special, the shortest
    #one output by TSP algorithm will be a Hamiltonian cycle)
    # Thus, we can do Hamiltonian paths with weighing
    weighted_copy = self.make_graph_weighted(modify_self = False)
    return weighted_copy.solve_traveling_salesman_problem(
        compute_path_instead_of_cycle = False,
        initial_vertex = initial_vertex,
        final_vertex = None, # No info on final vertex for cycles
        use_memoization_instead_of_tabulation = use_memoization_instead_of_tabulation,
        output_as = output_as,
        skip_checks = skip_checks)

  def get_hamiltonian_path(self, initial_vertex = None, final_vertex = None,
      use_memoization_instead_of_tabulation = False, output_as = None,
      skip_checks = False):
    '''
    Returns a Hamiltonian path of the unweighted digraph. That is, a path
    through all the vertices traveling through the available arrows.
    
    Accomodates a request for the path to start at a specific vertex.
    
    If no such cycle exists, returns None.
    
    SEE ALSO: get_hamiltonian_cycle
    '''
    # Very similar to get_hamiltonian_cycle, except that we pass the
    #instruction to solv TSP finding a path instead of a cycle
    weighted_copy = self.make_graph_weighted(modify_self = False)
    return weighted_copy.solve_traveling_salesman_problem(
        compute_path_instead_of_cycle = False,
        initial_vertex = initial_vertex,
        final_vertex = final_vertex,
        use_memoization_instead_of_tabulation = use_memoization_instead_of_tabulation,
        output_as = output_as,
        skip_checks = skip_checks)

########################################################################
# Class Graph
########################################################################

class Graph(Digraph):
  '''
  A graph is a digraph whose arrows all have "opposites", forming edges.
  
  Edges might or not be weighted, depending on subclassing.
  '''
    
  def get_edges(self):
    '''
    Returns all edges of the graph.
    '''
    return self._edges
    
  def get_number_of_edges(self):
    '''
    Returns the number of edges of the graph.
    '''
    return len(self._edges)

  def get_inciding_edges(self, vertex, skip_checks = False):
    '''
    Returns the edges inciding on a vertex.
    '''
    # Important to note that in Digraph.__init__, inputting as 'edges'
    #does not correspond to this notion of _inciding_edges
    if not skip_checks:
      assert self.belongs_to_as_vertex_after_sanitization(vertex, require_vertex_namedtuple = False), 'Vertex must be in graph'
    return self._inciding_edges[vertex]
    
  def get_neighbors(self, vertex, forbid_repetitions = False, skip_checks = False):
    '''
    Return the neighbors of vertex.
    
    Has option to repeat or not vertices in case of multigraph.
    '''
    if not skip_checks:
      assert self.belongs_to_as_vertex_after_sanitization(vertex, require_vertex_namedtuple = False), 'Vertex must be in graph'
    # We need to be careful that in an Edge(first, second, weight)
    #the vertex itself might be the first or second
    # For this reason, we use self._arrows_out instead
    # [For self-loops: a vertex will be a neighbor of itself]
    # Also important to note that in Digraph.__init__, the meaning of
    #'neighbors' in the possible values of data_type is other than this
    list_with_possible_repetitions = [arrow.target for arrow in self._arrows_out[vertex]]
    if forbid_repetitions:
      return list(set(all_with_possible_repetitions))
    else:
      return list_with_possible_repetitions

  def get_degree_of_vertex(self, vertex, skip_checks = False):
    '''
    Returns degree of vertex in graph: number of edges adjacent to it.
    '''
    # Can do it with self._inciding_edgfes, self._arrows_in or self._arrows_out
    if not skip_checks:
      assert self.belongs_to_as_vertex_after_sanitization(vertex, require_vertex_namedtuple = False), 'Vertex must be in graph'
    return len(self._inciding_edges[vertex])

  def find_cut(self):
    '''
    Provides a randomly selected cut of the graph.
    
    Works only for undirected graphs.
    
    INPUTS:
    self
    
    OUTPUT:
    crossing_edges: number of crossing edges between the two subgraphs
    one_side: one of the subgraphs of the cut
    other_side: the other subgraph
    '''
    # Two constants for the call
    n = self.get_number_of_vertices()
    m = self.get_number_of_edges() # Number of edges. Not used in code
    if n <= 1:
      raise ValueError('graph needs at least 2 vertices')
    # We now do contractions at random edges until we are left with two points
    # That means we need to do n - 2 contractions, until there are two leaders
    # We will use [modifiable] lists to control edges and vertices
    edges = list(self._arrows) # Does not change self. Not used in code
    neighbors = dict(self._arrows_out) # Does not change self, to be very safe
    # We will use a structure similar to what is known as "Union-Find"
    # We will divide the vertices into groups. The information will be recorded for each vertex
    # It will be the same (even in the same order) for all in the same group
    # Also, each group will have a single representative, called leader.
    # This will be stored as a dict "full_groups" and as a dict "leaders"
    # We initialize with one group for each vertex, represented by the very vertex
    # Each contraction will merge two groups, with a new leader chosen
    leaders = {vertex: vertex for vertex in neighbors.keys()}
    full_groups = {vertex: [vertex] for vertex in neighbors.keys()}
    for count in range(n - 2):
      # To select edge at random, first we select a vertex at random with weights
      # The weight is the degree (the number of edges that vertex is part of)
      # Note the sum of the weights is actually the double of the sum of the degrees
      # Let's call the first vertex v
      # (Note random_choices produces a list. Thus, we need to pick the first and only element.)
      v = random_choices([vertex for vertex in neighbors], weights = [len(neighbors[vertex]) for vertex in neighbors])[0]
      # Now we select an edge from v, indirectly selecting a vertex called w
      w = random_choices(neighbors[v])[0]
      #print('\nProgress: deleting edge ({}, {}), {} to go'.format(v, w, n-3-count))
      #print('Vertex {} has {} neighbors (with repetitions)'.format(v, len(neighbors[v])))
      #print('Vertex {} has {} neighbors (with repetitions)'.format(w, len(neighbors[w])))
      # We selected (v, w) as edge. We also will merge the groups
      copy_full_group_v = list(full_groups[v])
      copy_full_group_w = list(full_groups[w])
      #print('Group of v = {} is {}'.format(v, copy_full_group_v))
      #print('Group of w = {} is {}\n'.format(w, copy_full_group_w))
      for vertex in copy_full_group_v:
        #print('Examining vertex {} in the group of v = {}'.format(vertex, v))
        full_groups[vertex] = full_groups[vertex] + full_groups[w]
        #for idx in range(len(neighbors[vertex])-1, -1, -1):
          #if leaders[neighbors[vertex][idx]] == leaders[w]:
            #neighbors[vertex].pop(idx)
      for vertex in copy_full_group_w:
        #print('Examining vertex {} in the group of w = {}'.format(vertex, w))
        full_groups[vertex] = full_groups[v] # It should be updated with the union
        #for idx in range(len(neighbors[vertex])-1, -1, -1):
          #if leaders[neighbors[vertex][idx]] == leaders[v]:
            #neighbors[vertex].pop(idx)
      # Now we should have full_groups[v] == full_groups[w], v the leader of all of them
      # The leader of the whole group of v will become the leader of the whole group of w
      for vertex in full_groups[v]:
        leaders[vertex] = v
      # We don't need to change the edges, with one exception: remove the self-edges
      # That is, remove a edge if its two vertices are in the just formed contraction
      for vertex in full_groups[v]:
        neighbors[vertex] = [u for u in neighbors[vertex] if leaders[u] != v]
      #print('')
      #print('Group of v is now {}, with leader {}'.format(full_groups[v], leaders[v]))
      #print('Group of w is now {}, with leader {}'.format(full_groups[w], leaders[w]))
    # When we finish the process, the number of cuts is the number of remaining edges
    # That is, we removed the contracted edges, exactly once, and nothing else
    double_crossing_edges = sum(len(neighbors[vertex]) for vertex in neighbors)
    # We divide by two because we double-counted
    crossing_edges = double_crossing_edges // 2
    # We now identify and return the two groups (by their leaders)
    leader_one_side = None
    leader_other_side = None
    for vertex in neighbors:
      if leader_one_side is None:
        leader_one_side = leaders[vertex]
      elif leader_other_side is None and leaders[vertex] != leader_one_side:
        leader_other_side = leaders[vertex]
    # The two groups are recorded as full_groups under their respective leaders
    one_side = full_groups[leader_one_side]
    other_side = full_groups[leader_other_side]
    return (crossing_edges, one_side, other_side)

  def is_graph_connected(self):
    '''
    Returns whether the graph is connected.
    '''
    # Graph is connected if it has 0 or 1 connected components
    if self.get_ccs()[2] <= 1:
      return True
    else:
      return False

  def get_ccs(self):
    '''
    Returns the connected components of unweighted, undirected graph.
    '''
    # We need an undirected graph. Unfortunately, testing is too costly
    #assert self.is_undirected_graph(), 'Graph needs to be undirected'
    # Graph can be weighted or not, it should work the same way
    # We use a class designed specifically for this method
    # (We need a way to store and control a state, so we use a class)
    state = StateGraphGetCC(self)
    # We use the dfs_outer_loop method to get the results
    vertices_by_label, components, number_components = state.dfs_outer_loop()
    # vertices_by_label is a dict whose keys are number from 0 up to
    #the number of connected components minus one, and whose values are
    #lists of all vertices on the respective connected components
    # components gives the label for each vertex (which is another way
    #of presenting the same information)
    # number_components is the number of connected components
    return (vertices_by_label, components, number_components)
    
  def get_vertex_cover_with_limited_size(self, k):
    '''
    Returns a vertex cover of the graph with at most k vertices, or None
    if such vertex cover doesn't exist.
    
    A vertex cover is a subset of the vertices such that, for any edge,
    one of the two vertices it incides on is on the vertex cover.
    
    If m == self.number_of_edges(), running time is O(m*(2**k)).
    If removed the variable k and tried to find the smallest vertex cover,
    then we would have an NP-complete problem.
    '''
    raise NotImplementedError('WORK HERE')

########################################################################
# Class WeightedGraph
########################################################################

class WeightedGraph(WeightedDigraph, Graph):
  '''
  A graph whose edges are all weighted.
  '''

  @classmethod
  def from_names_and_coordinates(cls, list_of_points, number_of_dimensions = None,
      maximum_distance = None, skip_checks = False):
    '''
    Creates the complete graph corresponding to a list of cities (or anything
    represented by points in any-dimensional space) and the pairwise distances.
    
    Given a list of tuples with names and any-dimensional coordinates,
    creates a weighted graph which is a complete graph whose vertices are
    Vertex namedtuples created after the names (the first item of each tuple)
    and whose edges are weighted according to the Euclidean distance
    between the names of the vertices.
    
    A maximum distance might be set as a number; if a distance is larger than that,
    the vertices won't be connected by an edge.
    '''
    # Default is two-dimensional
    if number_of_dimensions is None:
      number_of_dimensions = 2
    # We ensure data makes sense
    if not skip_checks:
      try:
        float_number_of_dimensions = float(number_of_dimensions)
      except ValueError:
        raise ValueError('Need number_of_dimensions to be a number')
      if float_number_of_dimensions.is_integer:
        number_of_dimensions = int(float_number_of_dimensions)
      else:
        raise ValueError('Need a whole number of dimensions.')
      assert number_of_dimensions >= 1, 'Needs at least one dimension'
      assert hasattr(list_of_points, '__len__'), 'Need list_of_points to be iterable'
      for item in list_of_points:
        assert hasattr(item, '__len__'), 'Items must be iterable'
        assert len(item) == number_of_dimensions + 1, 'Each item must have a name and then coordinates'
        for sub_index, sub_item in enumerate(item):
          if sub_index >= 1:
            assert float(sub_item) == sub_item, 'sub-items from second onward must be numeric'
    else:
      # Ensure number_of_dimensions is integer
      number_of_dimensions = int(number_of_dimensions)
    # Graph to be initiated with data_type "all_vertices_and_all_edges"
    # Include vertex because if list_of_points has length one it has no edges
    list_of_vertices = []
    list_of_edges = []
    for first_index, first_item in enumerate(list_of_points):
      first_vertex_name, *first_vertex_coordinates = first_item
      list_of_vertices.append(first_vertex_name)
      for second_index in range(first_index):
        second_item = list_of_points[second_index]
        second_vertex_name, *second_vertex_coordinates = second_item
        if first_vertex_name == second_vertex_name:
          raise ValueError('Cannot have two points with same name')
        # Uses are not anticipated to be extremely large
        # Thus, using built-in math package [instead of numpy] is good enough
        distance = math_sqrt(sum((first_vertex_coordinates[idx] - second_vertex_coordinates[idx])**2 for idx in range(number_of_dimensions)))
        if maximum_distance is None or distance <= maximum_distance:
          list_of_edges.append((first_vertex_name, second_vertex_name, distance))
    data = (list_of_vertices, list_of_edges)
    data_type = 'all_vertices_and_all_edges'
    # Since this is a class method this will also work for subclasses
    #of WeightedGraph, should that ever be needed
    new_instance = cls(data = data, data_type = data_type)
    return new_instance

  def get_minimal_spanning_tree_via_Prims(self):
    '''
    Returns a minimum spanning tree (a list of edges), plus its cost,
    using Prim's algorithm.
    
    Requires a WeightedGraph (undirected, weighted).
    '''
    # We need a weighted, undirected graph. Weights may be negative
    # We expect non-multigraphs, but it is not a problem (only the extra time wasted)
    # Graph needs to be connected. We can check it inside the Prim's algorithm
    assert self.is_nonempty_digraph(), 'Algorithm requires at least one vertex'
    # Principe behind Prim's algorithm is to increase, vertex a vertex, a set X
    # Let Y be the complement.
    X = []
    Y = self.get_vertices()
    # We also have a set of edges forming the tree. Call it E
    E = []
    # We start with a "random" vertex. We'll take the first from Y
    first_vertex = Y[0] # Note this is a pair (vertex, weight)
    X.append(first_vertex)
    Y.remove(first_vertex)
    # Each time, we add the crossing edge with the smallest weight to the tree
    # (Thus adding one extra point to X)
    # We do it via heaps
    # For each vertex, we control the distance by another heap and only its edges
    # First, we do the initialization process
    total_cost = 0
    individual_heaps = {vertex:[] for vertex in Y}
    for vertex in individual_heaps:
      heapq_heapify(individual_heaps[vertex]) # Optional since it's empty
    for arrow in self.get_arrows_out(first_vertex, skip_checks = True): # Equivalently, X[0]
      # We update individual_heaps. The only edges that matter connect to X
      # (We scan through first vertex to scan fewer edges)
      # Note that due to heapq ordering requirements, the edge will be
      #represented as (weight, vertex) instead
      heapq_heappush(individual_heaps[arrow.target], (arrow.weight, first_vertex))
    # We keep one heap for the vertices in Y, each indexed by the least distance to Y
    # We heapify as a tuple (shortest_distance, vertex)
    # By shortest distance we mean the minimum of the corresponding individual_heap
    # We put this cost/distance/score/weight to be math_inf if heap is empty
    vertices_heap = []
    heapq_heapify(vertices_heap)
    for vertex in individual_heaps:
      if not individual_heaps[vertex]:
        # We insert a fake node
        heapq_heappush(individual_heaps[vertex], (math_inf, None))
        # individual_heaps[vertex][0] is (math_inf, None)
        minimal_weight = individual_heaps[vertex][0][0]
      else:
        # In this case individual_heaps[vertex][0] is (minimum weight, other vertex)
        minimal_weight = individual_heaps[vertex][0][0]
      # Note that in case of equality between minimal_weight, heapq resorts to vertex
      # We cross our fingers and hope it will be able to compare them
      # Otherwise we'd need to envelop vertex in a class with __lt__ method
      heapq_heappush(vertices_heap, (minimal_weight, vertex))
    # Now we enter into the main loop
    # In each iteration, we choose the shortest relevant edge
    # And then update the relevant data
    while len(Y) > 0:
      pre_new_vertex = heapq_heappop(vertices_heap)
      # Due to duplicates, we need to check if indeed this result of heappop
      #is not already in X
      # (Otherwise we need to get the next, then the next, and so on)
      while pre_new_vertex[1] in X:
        pre_new_vertex = heapq_heappop(vertices_heap)
      cost_of_new_edge = pre_new_vertex[0]
      new_vertex = pre_new_vertex[1]
      # The vertex is stored as a pair, first the minimum weight to X, then the vertex
      cost_of_new_edge = pre_new_vertex[0]
      new_vertex = pre_new_vertex[1]
      if cost_of_new_edge == math_inf:
        # We open an exception here. If we got +Infinity, it means the graph is not connected
        raise ValueError('We cannot apply Prim\'s algorithm in an unconnected graph.')
      total_cost += cost_of_new_edge
      X.append(new_vertex)
      # Sometimes we insert duplicates of vertices (see later)
      # So we remove as many as possible, doing a complete scan of the list Y
      Y.remove(new_vertex)
      #Y = [vertex for vertex in Y if vertex != new_vertex] # This is costlier
      pre_new_edge_to_add = heapq_heappop(individual_heaps[new_vertex])
      # Recall reverse order in tuple, weight comes first in heap
      # To get the two vertices (first in X, second in Y) and the weight, we do:
      assert pre_new_edge_to_add[0] == cost_of_new_edge, 'Weight of edges does not match'
      connecting_vertex = pre_new_edge_to_add[1]
      new_edge_to_add = Edge(connecting_vertex, new_vertex, cost_of_new_edge)
      E.append(new_edge_to_add)
      # Unfortunately deletion is awful using heapq. But it's the best we have
      # We don't need to delete the vertex new_vertex from vertices_heap
      #(Since the deletion is automatic with the heappop)
      # We should delete the entry of new_vertex from the dict individual_heaps
      del individual_heaps[new_vertex]
      # Now we need to update the individual_heaps
      # All we need to do is add a new edges to a few vertices in Y
      # Namely, we need to add the edges inciding on new_vertex to the heaps
      for arrow in self.get_arrows_from(new_vertex, skip_checks = True):
        # adjacency is of the form (new_vertex, some_vertex, weight)
        if arrow.target in Y:
          # Note new_vertex is now in X. (Note weight comes first)
          heapq_heappush(individual_heaps[arrow.target], (arrow.weight, new_vertex))
          # Now we should update vertices_heap with the new weights
          # Since this won't ever increase the minimum_weights,
          #we can add this and keep the old version of adjacency[0] in Y
          # (Not worth to search for it for comparison, simply throw into heap)
          new_minimal_weight = arrow.weight
          # Since this is not a multigraph we don't repeat the adjacency[0]'s
          heapq_heappush(vertices_heap, (new_minimal_weight, arrow.target))
      # Believe it or not, I think we are done for the cycle!!
    # We return the edges as (u, v, weight) from the tree, as well as total cost
    return (E, total_cost)

########################################################################
# Class UnweightedGraph
########################################################################
  
class UnweightedGraph(UnweightedDigraph, Graph):
  '''
  A graph whose edges are all unweighted.
  '''
  pass

########################################################################

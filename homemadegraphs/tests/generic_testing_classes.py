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

from abc import ABCMeta as abc_ABCMeta
from abc import abstractmethod as abc_abstractmethod
from collections import namedtuple as collections_namedtuple
from unittest import skip as unittest_skip
from unittest import skipIf as unittest_skipIf
from unittest import SkipTest as unittest_SkipTest
from unittest import TestCase as unittest_TestCase

########################################################################
# Internal imports
########################################################################



########################################################################
# Generic test procedures
########################################################################

class GenericInitializationTestCase(unittest_TestCase):
  '''
  Provides some generic and enhanced version of unittest.TestCase by
  providing a few generic class methods related to initialization.
  
  The methods in the derived classes will be called from unittest but not
  the ones in this class due to unittest.skipIf.
  
  Idea is that a derived class will test multiple initialization methods
  (stored in a class variable) of the "same" instance (of a class which
  is being tested) and verify they all produce the "same" instance.
  '''

  # We define a namedtuple to standartize functions/methods/attributes in tests
  # See method test_property_specifications
  # If non-callable, expect arguments to be None. Otherwise, for no arguments
  #(beside self which is automatic) use the empty tuple, tuple()
  PropertySpecification = collections_namedtuple('PropertySpecification', 'attribute,output,is_callable,arguments', defaults = (False, None))

  @classmethod
  def setUpClass(cls):
    '''
    Called automatically by unittest.
    
    Only allows the tests in a class to go forward when specific information,
    in the form of a class attribute, are provided.
    
    In particular, for the generic class, no tests are performed.
    '''
    # One alternative way to implement this is with abstract base classes
    # It would have the problem of forcing the implementation of abstract methods
    #in every class (overriding the base class's), which is not the goal
    # Note: unittest doesn't call setUpClass for skipped tests, so this is
    #closer to what we want to accomplish with this generic testing class
    if not hasattr(cls, 'recipes_for_data_and_data_types'):
      raise unittest_SkipTest('Need recipes for initialization of the tested classes.')

  def test_initialization(self, deactivate_assertions = False):
    '''
    Initializes one instance by all multiple input types.
    '''
    cls = self.__class__
    dict_of_instances = {}
    data_and_data_types = self.recipes_for_data_and_data_types()
    # We use subTest to discriminate what we are doing
    # It accepts any keyword parameters for parametrization
    for key in data_and_data_types:
      with self.subTest(data_type = key):
        data_type = key
        data = data_and_data_types[key]
        dict_of_instances[data_type] = cls.class_being_tested(data = data, data_type = data_type)
        if not deactivate_assertions:
          # We want to test this only when called directly by unittest
          self.assertIsInstance(dict_of_instances[data_type], cls.class_being_tested)
    return dict_of_instances

  @unittest_skip('Probably redundant given test_equality_against_specific')
  def test_pairwise_equality(self):
    '''
    Tests equality for all pairs of instances emerging from different input types.
    '''
    # Creating all instances, using the test_initialization for better separation
    dict_of_instances = self.test_initialization(deactivate_assertions = True)
    count = 0
    for key_1 in dict_of_instances:
      for key_2 in dict_of_instances:
        with self.subTest(data_types = (key_1, key_2)):
          instance_1 = dict_of_instances[key_1]
          instance_2 = dict_of_instances[key_2]
          count += 1
          self.assertEqual(instance_1, instance_2)
    # We also verify this testing tests all (total_instances)**2 pairs
    number_of_instances = len(dict_of_instances)
    self.assertEqual(count, number_of_instances**2)

  def test_equality_against_specific(self):
    '''
    Compares equality for all instances emerging from different inputs
    against one specific among them.
    '''
    dict_of_instances = self.test_initialization(deactivate_assertions = True)
    # Pick only one of them, and compare all others to this special fixed instance
    if dict_of_instances:
      fixed_key = list(dict_of_instances)[0]
      fixed_instance = dict_of_instances[fixed_key]
      for variable_key in dict_of_instances:
        with self.subTest(fixed_key = fixed_key, variable_key = variable_key):
          variable_instance = dict_of_instances[variable_key]
          self.assertEqual(fixed_instance, variable_instance)
          
  def test_property_specifications(self):
    '''
    Checks whether inputs produce instance with the speficied qualities.
    '''
    cls = self.__class__
    if not hasattr(cls, 'property_specifications'):
      raise unittest_SkipTest('Need standard properties to compare instances against.')
    # We bring all the instances
    dict_of_instances = self.test_initialization(deactivate_assertions = True)
    # We check if each instance has each property as specified
    # (Note property is a reserved word so we might write propertyy in code)
    for property_specification in self.property_specifications():
      for instance_key in dict_of_instances:
        with self.subTest(property_specification = property_specification, instance_key = instance_key):
          instance = dict_of_instances[instance_key]
          # We compare property computed with property given
          property_given = property_specification.output
          if property_specification.is_callable:
            property_computed = getattr(instance, property_specification.attribute)(*property_specification.arguments)
          else:
            property_computed = getattr(instance, property_specification.attribute)
          self.assertEqual(property_computed, property_given)
            
########################################################################



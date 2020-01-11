from typing import Any

from utils import *


class Node:
	def __init__(self, attribute, father_value=""):
		self.attribute = attribute
		self.father_value = father_value
		self.children = []

	def add_child(self, node):
		self.children.append(node)

	def is_leaf(self):
		return not self.children

	def set_father_value(self, father_value):
		self.father_value = father_value

	def __str__(self):
		return self.attribute


def same_examples_classification(examples):
	labels = [tup[1] for tup in examples]
	return len(set(labels)) <= 1


def mode(labels):
	return most_frequent(labels)


def get_examples_by_attribute_value(examples, best, value):
	return [example for example in examples if example[best] == value]


def DTL(examples, attributes, default):
	labels = [tup[1] for tup in examples]

	if not examples:
		return default
	elif same_examples_classification(labels):
		# return the same classification.
		return examples[0][1]
	elif not attributes:
		return mode(examples)
	else:
		best = choose_attribute(attributes, examples)
		tree = Node(best)
		for value in best:
			examples_value = get_examples_by_attribute_value(examples, value)
			attributes_no_best = attributes.copy()
			attributes_no_best.remove(best)
			subtree = DTL(examples_value, attributes_no_best, mode(examples))
			# TODO: add a branch to tree with label v_i and subtree subtree mabye thise implementation:
			subtree.set_father_value(value)
			tree.add_child(subtree)
	return tree

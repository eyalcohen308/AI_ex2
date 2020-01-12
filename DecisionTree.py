from math import log2
from Algorithms import DistanceAlgorithm
from utils import *


class DecisionTree(DistanceAlgorithm):
	def __init__(self, attributes, default):
		super(DecisionTree, self).__init__()
		self.tree = DTL(examples, attributes, default)

	def get_accuracy_on_test(self, train, test):
		# TODO: need to understand how to run on the tree
		correct_answers = sum([get_knn_prediction(example, train, self.k) == example[1] for example in test])
		return correct_answers / len(test)


class Node:
	def __init__(self, attribute, father_value=""):
		self.attribute = attribute
		self.children = []

	def add_child(self, node, value):
		self.children.append((node, value))

	def is_leaf(self):
		return not self.children

	def __int__(self):
		return self.attribute

	def __str__(self):
		return self.attribute if self.is_leaf else ""


def is_same_classification(labels):
	return len(set(labels)) <= 1


def entropy(data):
	values = get_tags_from_data(data)
	values_counter = dict(Counter(values))
	values_len = len(values)
	entropy_value = 0
	for value, occurrence in values_counter.items():
		value_prob = occurrence / values_len
		entropy_value += value_prob * log2(value_prob)
	return -entropy_value


def gain(s, a, a_values):
	s_entropy = entropy(s)
	weighted_average = 0
	for value in a_values:
		s_v = get_examples_by_attribute_value(s, a, value)
		weighted_average += (len(s_v) / len(s)) * entropy(s_v)
	return s_entropy - weighted_average


def mode(labels):
	return most_frequent(labels)


def get_examples_by_attribute_value(examples, attribute, value):
	# return [example for example in examples if example[attribute] == value]
	return [example for example in examples if example[0][attribute] == value]


def choose_attribute(attributes, examples):
	values = [gain(examples, attribute[0], attribute[1]) for attribute in attributes]
	return attributes[argmax(values)]


def print_tree(root, I2F, file, layer=0):
	for child, edge in sorted(root.children, key=lambda ch: ch[1]):
		prefix = ("\t" * layer) + ("|" if layer else "")
		if child.is_leaf():
			line = prefix + "{0}={1}:{2}".format(I2F[int(root)], edge, str(child))
			print(line)
			file.write(line + "\n")
		else:
			line = prefix + "{0}={1}".format(I2F[int(root)], edge)
			print(line)
			file.write(line + "\n")
			print_tree(child, I2F, file,layer + 1)


def DTL(examples, attributes, default):
	labels = get_tags_from_data(examples)

	if not examples:
		return Node(default)
	elif is_same_classification(labels):
		# return the same classification.
		return Node(labels[0])
	elif not attributes:
		return Node(mode(labels))
	else:
		best_attribute, best_values = choose_attribute(attributes, examples)
		tree = Node(best_attribute)
		for value in best_values:
			examples_value = get_examples_by_attribute_value(examples, best_attribute, value)
			# for DTL new input (attributes - best)
			attributes_no_best = attributes.copy()

			attributes_no_best = list(filter(lambda tup: tup[0] != best_attribute, attributes_no_best))

			subtree = DTL(examples_value, attributes_no_best, mode(labels))
			# TODO: add a branch to tree with label v_i and subtree subtree maybe this implementation:
			tree.add_child(subtree, value)
	return tree


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="AI Ex3")
	parser.add_argument("--k_cross", help="choose k for k cross validation. default k is 5", type=int, default=5)
	parser.add_argument("--knn", help="choose k for k nearest neighbors. default k is 5", type=int, default=5)
	args = parser.parse_args()
	examples, feature_names = parse_data(DATASET_PATH, with_attributes_names=True)
	I2F = get_I2F(feature_names)
	dicts = Dicts(examples)
	tags = get_tags_from_data(examples)
	default = mode(tags)
	attributes_values = dicts.attributes_sets
	# attributes = list(range(len(feature_names)))
	attributes = [(i, values) for i, values in enumerate(attributes_values)]
	algorithm = DecisionTree(attributes, default)
	with open("tree.txt", 'w') as file:
		print_tree(algorithm.tree, I2F, file)
	# k_cross_validation_acu(algorithm, data, args.k_cross)

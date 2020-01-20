from math import log2
from Algorithms import ModelAlgorithm
from utils import *


def get_dtl_default(train):
	tags = get_tags_from_data(train)
	default = mode(tags)
	return default


def all_leaves_class(node):
	answers = []
	for child, _ in node.children:
		if child.is_leaf():
			return str(child)
		else:
			answers.append(all_leaves_class(child))
	if None in answers or len(set(answers)) > 1:
		return None
	else:
		return answers[0]


def prune_the_tree(node):
	answers = []
	for child, _ in node.children:

		if child.is_leaf():
			answers.append(str(child))
		else:
			answers.append(prune_the_tree(child))

	if len(set(answers)) == 1 and answers[0] is not None:
		node.attribute = answers[0]
		node.children = []
		return answers[0]
	else:
		return None


#
# if not child.is_leaf():
# 	tag = all_leaves_class(child)
# 	if tag:
# 		node.attribute = tag
# 		node.children = []
# 	else:
# 		prune_the_tree(child)


class DecisionTree(ModelAlgorithm):
	# def __init__(self, attributes, default):
	# 	super(DecisionTree, self).__init__()
	# 	self.tree = DTL(examples, attributes, default)

	def __init__(self, examples):
		super(DecisionTree, self).__init__()

		dicts = Dicts(examples)
		attributes_values = dicts.attributes_sets
		attributes = [(i, values) for i, values in enumerate(attributes_values)]
		self.attributes = attributes
		self.print_examples = examples
		self.print_default = get_dtl_default(examples)
		self.tree = ""

	def get_accuracy_on_test(self, train, test):
		default = get_dtl_default(train)
		self.tree = DTL(train, self.attributes, default)

		correct_answers = sum([DTL_predict(self.tree, example) == example[1] for example in test])
		return correct_answers / len(test)

	def print_tree(self, I2F, prune_tree=False):
		with open("tree.txt", 'w') as file:
			self.tree = DTL(self.print_examples, self.attributes, self.print_default)
			if prune_tree:
				prune_the_tree(self.tree)
			print_tree_recursive(self.tree, I2F, file)


class Node:
	def __init__(self, attribute, father_value=""):
		self.attribute = attribute
		self.children = []

	def add_child(self, node, value):
		self.children.append((node, value))

	def get_child_by_edge(self, given_edge):
		for child, edge in self.children:
			if edge == given_edge:
				return child
		raise IndexError("didn't find child in tree by feature value")

	def is_leaf(self):
		return not self.children

	def get_attribute(self):
		return self.attribute

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


def print_tree_recursive(root, I2F, file, layer=0):
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
			print_tree_recursive(child, I2F, file, layer + 1)


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
			tree.add_child(subtree, value)
	return tree


def DTL_predict(tree: Node, example):
	example_features = example[0]
	current_node = tree
	while not current_node.is_leaf():
		node_attribute = current_node.get_attribute()
		attribute_value = example_features[node_attribute]
		current_node = current_node.get_child_by_edge(attribute_value)
	return str(current_node)


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="AI Ex3")
	parser.add_argument("--k_cross", help="choose k for k cross validation. default k is 5.", type=int, default=5)
	parser.add_argument("--knn", help="choose k for k nearest neighbors. default k is 5.", type=int, default=5)
	parser.add_argument("--save_tree", help="if you want to save the tree to txt file.", action="store_true")
	args = parser.parse_args()

	data, feature_names = parse_data(DATASET_PATH, with_attributes_names=True)
	algorithm = DecisionTree(data)
	if args.save_tree:
		I2F = get_I2F(feature_names)
		algorithm.print_tree(I2F, prune_tree=False)
	k_cross_validation_acu(algorithm, data, args.k_cross)

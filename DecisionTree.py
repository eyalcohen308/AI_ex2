from math import log2
from Algorithms import ModelAlgorithm
from utils import *

EQUAL_DEFAULT_VALUE = ""


def get_dtl_default(train):
	"""
	get dtl default by mox frequent.
	:param train: train data.
	:return: deafult value.
	"""
	tags = get_tags_from_data(train)
	default = first_time_most_frequent(tags)
	global EQUAL_DEFAULT_VALUE
	EQUAL_DEFAULT_VALUE = default
	return default


def all_leaves_class(node):
	"""
	check if all leaves has same value.
	:param node: node to check.
	:return: the value or None.
	"""
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
	"""
	cut irrelevant tree.
	:param node: root node.
	:return: pruned tree.
	"""
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


class DecisionTree(ModelAlgorithm):
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
		"""
		get accuracy on test by train and algorithm.
		:param train: train data.
		:param test: test data.
		:return: accuracy
		"""
		default = get_dtl_default(train)
		self.tree = DTL(train, self.attributes, default)

		correct_answers = sum([DTL_predict(self.tree, example) == example[1] for example in test])
		return correct_answers / len(test)

	def save_tree_to_file(self, I2F, tree_file_path, prune_tree=False):
		"""
		save tree to file
		:param I2F: index to feature.
		:param tree_file_path:  tree path.
		:param prune_tree: if need to prune.
		:return: Void.
		"""
		with open(tree_file_path, 'w') as file:
			self.tree = DTL(self.print_examples, self.attributes, self.print_default)
			if prune_tree:
				prune_the_tree(self.tree)
			print_tree_recursive(self.tree, I2F, file)
		file.close()


class Node:
	def __init__(self, attribute, father_value=""):
		self.attribute = attribute
		self.children = []

	def add_child(self, node, value):
		"""
		Add child to tree
		:param node: node to add.
		:param value: value indicate that node.
		:return: void.
		"""
		self.children.append((node, value))

	def get_child_by_edge(self, given_edge):
		"""
		get child by value/edge.
		:param given_edge: given edge.
		:return: none.
		"""
		for child, edge in self.children:
			if edge == given_edge:
				return child
		raise IndexError("didn't find child in tree by feature value")

	def is_leaf(self):
		"""
		return true if leaf
		:return: true if leaf.
		"""
		return not self.children

	def get_attribute(self):
		"""
		get self attribute.
		:return: attribute value.
		"""
		return self.attribute

	def __int__(self):
		"""
		return the attribute.
		:return: the attribute.
		"""
		return self.attribute

	def __str__(self):
		"""
		return the attribute.
		:return: the attribute.
		"""
		return self.attribute if self.is_leaf else ""


def is_same_classification(labels):
	"""
	check if the list has more then one label.
	:param labels: list.
	:return: true or false.
	"""
	return len(set(labels)) <= 1


def entropy(data):
	"""
	Entropy like lecture on data.
	:param data: data to entropy.
	:return: entropy value.
	"""
	values = get_tags_from_data(data)
	values_counter = dict(Counter(values))
	values_len = len(values)
	entropy_value = 0
	for value, occurrence in values_counter.items():
		value_prob = occurrence / values_len
		entropy_value += value_prob * log2(value_prob)
	return -entropy_value


def gain(s, a, a_values):
	"""
	gain on attribute values.
	:param s: score.
	:param a: attribute.
	:param a_values: attribute values.
	:return: gain value.
	"""
	s_entropy = entropy(s)
	weighted_average = 0
	for value in a_values:
		s_v = get_examples_by_attribute_value(s, a, value)
		weighted_average += (len(s_v) / len(s)) * entropy(s_v)
	return s_entropy - weighted_average


def mode(labels, equal_default):
	"""
	most frequent value.
	:param labels: labels of data.
	:param equal_default: the default value.
	:return: most frequent.
	"""
	return most_frequent(labels, equal_default)


def get_examples_by_attribute_value(examples, attribute, value):
	"""
	get examples attributes by theres values.
	:param examples: examples data.
	:param attribute: attribute index.
	:param value: value of attribute.
	:return: all relevant examples.
	"""
	# return [example for example in examples if example[attribute] == value]
	return [example for example in examples if example[0][attribute] == value]


def choose_attribute(attributes, examples):
	"""
	choose attribute by biggest gain score.
	:param attributes: attributes to choose from.
	:param examples: examples.
	:return: the attribute index.
	"""
	values = [gain(examples, attribute[0], attribute[1]) for attribute in attributes]
	return attributes[argmax(values)]


def print_tree_recursive(root, I2F, file, layer=0, print_tree=False):
	"""
	print the tree recursively.
	:param root: root of tree.
	:param I2F: index to feature dictionary.
	:param file: file to save to.
	:param layer: which layer is it.
	:param print_tree: if want to print the tree
	:return: null.
	"""
	for child, edge in sorted(root.children, key=lambda ch: ch[1]):
		prefix = ("\t" * layer) + ("|" if layer else "")
		if child.is_leaf():
			line = prefix + "{0}={1}:{2}".format(I2F[int(root)], edge, str(child))
			if print_tree:
				print(line)
			file.write(line + "\n")
		else:
			line = prefix + "{0}={1}".format(I2F[int(root)], edge)
			if print_tree:
				print(line)
			file.write(line + "\n")
			print_tree_recursive(child, I2F, file, layer + 1)


def DTL(examples, attributes, default):
	"""
	DTL function.
	:param examples: examples.
	:param attributes: all possible attributes.
	:param default: deafult value.
	:return: the build tree.
	"""
	labels = get_tags_from_data(examples)

	if not examples:
		return Node(default)
	elif is_same_classification(labels):
		# return the same classification.
		return Node(labels[0])
	elif not attributes:
		return Node(mode(labels, EQUAL_DEFAULT_VALUE))
	else:
		best_attribute, best_values = choose_attribute(attributes, examples)
		tree = Node(best_attribute)
		for value in best_values:
			examples_value = get_examples_by_attribute_value(examples, best_attribute, value)
			# for DTL new input (attributes - best)
			attributes_no_best = attributes.copy()

			attributes_no_best = list(filter(lambda tup: tup[0] != best_attribute, attributes_no_best))

			subtree = DTL(examples_value, attributes_no_best, mode(labels, EQUAL_DEFAULT_VALUE))
			tree.add_child(subtree, value)
	return tree


def DTL_predict(tree: Node, example):
	"""
	predict example by tree.
	:param tree: tree to predict by.
	:param example: example to check on.
	:return: the prediction.
	"""
	example_features = example[0]
	current_node = tree
	while not current_node.is_leaf():
		node_attribute = current_node.get_attribute()
		attribute_value = example_features[node_attribute]
		current_node = current_node.get_child_by_edge(attribute_value)
	return str(current_node)


save_tree = False
k_cross = 5

if __name__ == "__main__":

	data, feature_names = parse_data(DATASET_PATH, with_attributes_names=True)
	algorithm = DecisionTree(data)
	if save_tree:
		I2F = get_I2F(feature_names)
		algorithm.save_tree_to_file(I2F, "tree.txt", prune_tree=False)
	k_cross_validation_acu(algorithm, data, k_cross)

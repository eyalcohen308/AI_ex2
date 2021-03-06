import unittest
import random
from string import digits
from collections import Counter

DATASET_PATH = "./dataset.txt"


class Dicts:
	def __init__(self, dataset):
		dataset_T = list(map(list, zip(*[features[0] for features in dataset])))
		yes_list, no_list = [], []
		for x in dataset:
			(no_list, yes_list)[x[1] == "yes"].append(x[0])
		yes_list_size = len(yes_list)
		no_list_size = len(no_list)

		self.prior_yes = yes_list_size / len(dataset)
		self.prior_no = no_list_size / len(dataset)
		yes_T = list(map(list, zip(*yes_list)))
		no_T = list(map(list, zip(*no_list)))
		non_normalize_yes = [Counter(col) for col in yes_T]
		non_normalize_no = [Counter(col) for col in no_T]

		self.yes_features_probs = [{k: v / yes_list_size for k, v in dic.items()} for dic in non_normalize_yes]
		self.no_features_probs = [{k: v / no_list_size for k, v in dic.items()} for dic in non_normalize_no]

		self.attributes_sets = [set(attribute_col) for attribute_col in dataset_T]

	def get_prob_feature(self, feature, col, label):
		if label == "yes":
			answer = self.yes_features_probs[col].get(feature)
		else:
			answer = self.no_features_probs[col].get(feature)

		if not answer:
			raise KeyError("Key '{0}' does not exists in '{1}' features".format(feature, label))

		return answer

	def get_prior(self, label):
		return self.prior_yes if label == "yes" else self.prior_no


def first_time_most_frequent(tags):
	data = Counter(tags)
	if len(data) == 2:
		most_common, second_common = data.most_common(2)
		if most_common[1] == second_common[1]:
			return most_common[1]
	return data.most_common(1)[0][0]


def shuffle_data(data, tags):
	zipped = list(zip(data, tags))
	random.shuffle(zipped)
	shuffled_data, shuffled_tags = zip(*zipped)
	return shuffled_data, shuffled_tags


class TestUtils(unittest.TestCase):

	def test_hamming_distance(self):
		self.assertEqual(hamming_distance(*generete_hamming_examples(5, 0)), 0)
		self.assertEqual(hamming_distance(*generete_hamming_examples(5, 1)), 1)
		self.assertEqual(hamming_distance(*generete_hamming_examples(5, 2)), 2)
		self.assertEqual(hamming_distance(*generete_hamming_examples(5, 3)), 3)
		self.assertEqual(hamming_distance(*generete_hamming_examples(5, 4)), 4)
		self.assertEqual(hamming_distance(*generete_hamming_examples(5, 5)), 5)
		self.assertEqual(hamming_distance(*generete_hamming_examples(10, 6)), 6)
		self.assertEqual(hamming_distance(*generete_hamming_examples(20, 20)), 20)


def parse_data(dataset_path, shuffle=False, with_attributes_names=False):
	"""
	Parse data to line of features. data type: [([features], tag)].
	:param dataset_path: dataset path.
	:param shuffle if want to shuffle the data.
	:param with_attributes_names index to feature dict.
	:return: parsed dataset list, tags of dataset and features set to index dicts (or not).
	"""
	with open(dataset_path) as file:
		content = file.readlines()
	# you may also want to remove whitespace characters like `\n` at the end of each line
	parsed_data = [row.strip().split('\t') for row in content]
	features_names = parsed_data[0][:-1]
	# convert the data to be list of ([features], label) without the first row(features headlines)
	parsed_data = [(row[:-1], row[-1]) for row in parsed_data[1:]]
	if shuffle:
		random.shuffle(parsed_data)

	if with_attributes_names:
		return parsed_data, features_names
	else:
		return parsed_data


def hamming_distance(point1, point2):
	"""Calculate the Hamming distance between two bit strings"""
	assert len(point1) == len(point2), "Point do not have the same number of features"
	return sum(c1 != c2 for c1, c2 in zip(point1, point2))


def most_frequent(List, equal_default):
	"""
	get most frequent value in list.
	:param List: list
	:return: most frequent value.
	"""
	data = Counter(List)
	if len(data) == 2:
		most_common, second_common = data.most_common(2)
		if most_common[1] == second_common[1]:
			return equal_default
	return data.most_common(1)[0][0]


def generete_hamming_examples(test_size, unmatch_num):
	"""
	generete hamming examples.
	:param test_size: size of the sequence.
	:param unmatch_num: unmatch number of chars.
	:return: two generated strings.
	"""
	indexes = random.sample(range(0, test_size), unmatch_num)
	str1 = ''.join(random.choice(digits) for i in range(test_size))
	str1_list = list(str1)
	str2_list = str1_list.copy()
	for i in indexes:
		str2_list[i] = '$'
	return str1, ''.join(str2_list)


def split_to_k_lists(data, k, shuffle=False):
	"""
	Created list of n lists from given list.
	:param data: data to make k list from.
	:param k: number of lists.
	:param shuffle: if want's to shuffle.
	:return: list of k lists from data.
	"""
	cross_validation_list = []
	sublist_size = len(data) // k
	for i in range(k - 1):
		cross_validation_list.append(data[i * sublist_size:(i + 1) * sublist_size])
	cross_validation_list.append(data[(k - 1) * sublist_size:])
	if shuffle:
		random.shuffle(cross_validation_list)
	return cross_validation_list


def get_I2F(feature_names):
	"""
	Create index to feature dictionary
	:param feature_names: list of feature names.
	:return: dictionary.
	"""
	return dict(enumerate(feature_names))


def get_tags_from_data(dataset):
	return [tup[1] for tup in dataset]


def argmax(l):
	return max(range(len(l)), key=l.__getitem__)


def k_cross_validation_acu(algorithm, data, k, shuffle=False):
	k_lists = split_to_k_lists(data, k, shuffle)
	avg_acu = 0
	for i in range(k):
		train = k_lists.copy()
		del train[i]
		train = [item for sublist in train for item in sublist]
		test = k_lists[i]
		iter_acu = algorithm.get_accuracy_on_test(train, test)
		avg_acu += iter_acu
		print("Iteration {0}/{1} | Algorithm: {2} | accuracy: {3}".format(i + 1, k, type(algorithm).__name__, iter_acu))
	avg_acu /= k
	avg_acu = "{0:.2f}".format(avg_acu)
	print("------- Finished Calculate accuracy -------")
	print("Algorithm: {0} Aaccuracy: {1}".format(type(algorithm).__name__, avg_acu))
	return avg_acu


if __name__ == "__main__":
	data, I2F = parse_data(DATASET_PATH, with_attributes_names=True)
	dicts = Dicts(data)
	print(dicts)
# unittest.main()
# lists = [
# 	[1, 1, 1, 1],
# 	[1, 1, 1, 1],
# 	[1, 1, 1, 1],
# 	[1, 1, 1, 1],
# 	[2, 2, 2, 2],
# 	[2, 2, 2, 2],
# 	[2, 2, 2, 2],
# 	[2, 2, 2, 2],
# 	[3, 3, 3, 3],
# 	[3, 3, 3, 3],
# 	[3, 3, 3, 3],
# 	[3, 3, 3, 3]
# ]
# liststs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
# print(split_to_k_lists(liststs, 2))

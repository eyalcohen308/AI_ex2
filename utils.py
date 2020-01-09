import unittest
import random
from string import digits

DATASET_PATH = "./dataset.txt"


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


def parse_data(dataset_path, shuffle=False):
	"""
	Parse data to line of features. data type: [([features], tag)].
	:param dataset_path: dataset path.
	:param shuffle if want to shuffle the data.
	:return: parsed data set list.
	"""
	with open(dataset_path) as file:
		content = file.readlines()
	# you may also want to remove whitespace characters like `\n` at the end of each line
	parsed_data = [row.strip().split('\t') for row in content]
	# convert the data to be list of ([features], label) without the first row(features headlines)
	parsed_data = [(row[:-1], row[-1]) for row in parsed_data[1:]]
	if shuffle:
		random.shuffle(parsed_data)
	return parsed_data


def hamming_distance(point1, point2):
	"""Calculate the Hamming distance between two bit strings"""
	assert len(point1) == len(point2), "Point do not have the same number of features"
	return sum(c1 != c2 for c1, c2 in zip(point1, point2))


def most_frequent(List):
	"""
	get most frequent value in list.
	:param List: list
	:return: most frequent value.
	"""
	return max(set(List), key=List.count)


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


def calculate_acu(train, test, algorithm):
	# sum all the right answers (prediction = tag(example[1]).
	correct_answers = sum([algorithm.get_point_prediction(train, example) == example[1] for example in test])
	return correct_answers / len(test)


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


def k_cross_validation_acu(algorithm, data, k, shuffle=False):
	k_lists = split_to_k_lists(data, k, shuffle)
	avg_acu = 0
	for i in range(k):
		train = k_lists.copy()
		del train[i]
		train = [item for sublist in train for item in sublist]
		test = k_lists[i]
		iter_acu = calculate_acu(train, test, algorithm)
		avg_acu += iter_acu
		print("Iteration {0}/{1} | Algorithm: {2} | accuracy: {3}".format(i + 1, k, type(algorithm).__name__, iter_acu))
	avg_acu /= k
	print("Finished Calculate accuracy\n")
	print("Algorithm: {0} Aaccuracy: {1}".format(type(algorithm).__name__, avg_acu))


if __name__ == "__main__":
	# unittest.main()
	lists = [
		[1, 1, 1, 1],
		[1, 1, 1, 1],
		[1, 1, 1, 1],
		[1, 1, 1, 1],
		[2, 2, 2, 2],
		[2, 2, 2, 2],
		[2, 2, 2, 2],
		[2, 2, 2, 2],
		[3, 3, 3, 3],
		[3, 3, 3, 3],
		[3, 3, 3, 3],
		[3, 3, 3, 3]
	]
	liststs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
	print(split_to_k_lists(liststs, 2))
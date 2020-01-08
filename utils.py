import unittest
import random
from random import choice
from string import digits


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


def parse_data(dataset_path):
	with open(dataset_path) as file:
		content = file.readlines()
	# you may also want to remove whitespace characters like `\n` at the end of each line
	parsed_data = [row.strip().split('\t') for row in content]
	# convert the data to be list of ([features], label) without the first row(features headlines)
	parsed_data = [(row[:-1], row[-1]) for row in parsed_data[1:]]
	return parsed_data


def hamming_distance(point1, point2):
	"""Calculate the Hamming distance between two bit strings"""
	assert len(point1) == len(point2), "Point do not have the same number of features"
	return sum(c1 != c2 for c1, c2 in zip(point1, point2))


def most_frequent(List):
	return max(set(List), key=List.count)


def generete_hamming_examples(test_size, unmatch_num):
	indexes = random.sample(range(0, test_size), unmatch_num)
	str1 = ''.join(choice(digits) for i in range(test_size))
	str1_list = list(str1)
	str2_list = str1_list.copy()
	for i in indexes:
		str2_list[i] = '$'
	return str1, ''.join(str2_list)


if __name__ == "__main__":
	unittest.main()

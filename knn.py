from Algorithms import ModelAlgorithm
from utils import *


def get_knn_prediction(point, data, k):
	"""
	get k nearest neighbors by finding the k closest neighbors by hamming distance.
	:param point: the center, the data want to be predicted.
	:param data: dataset rows, each row is tuple of ([features],label).
	:param k: number of neighbors.
	:return: point prediction.
	"""

	"""
	create a tuple list of index and hamming distance between given point to the entire data.
	example: (1, 11), 1 = index, 11 = hamming distance.
	"""
	distances = [(i, hamming_distance(point[0], train_row[0])) for i, train_row in enumerate(data)]
	# sort the distances by hamming distance, ascending.
	distances.sort(key=lambda tup: tup[1])
	# get the k closest neighbors, (tup[0] is index).
	knn_indexes = [tup[0] for tup in distances[:k]]
	# iterate over the knn indexes, get the specified index's row, and form it get the label.
	knn_labels = [data[knn_current_index][1] for knn_current_index in knn_indexes]
	point_prediction = most_frequent(knn_labels)
	return point_prediction


class KNN(ModelAlgorithm):
	def __init__(self, k):
		super(KNN, self).__init__()
		self.k = k

	def get_accuracy_on_test(self, train, test):
		# sum all the right answers (prediction = tag(example[1]).
		correct_answers = sum([get_knn_prediction(example, train, self.k) == example[1] for example in test])
		return correct_answers / len(test)


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="AI Ex3")
	parser.add_argument("--k_cross", help="choose k for k cross validation. default k is 5", type=int, default=5)
	parser.add_argument("--knn", help="choose k for k nearest neighbors. default k is 5", type=int, default=5)
	args = parser.parse_args()
	data = parse_data(DATASET_PATH)
	algorithm = KNN(args.knn)
	k_cross_validation_acu(algorithm, data, args.k_cross)

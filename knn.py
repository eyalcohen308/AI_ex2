from Algorithms import ModelAlgorithm
import heapq
from utils import *

EQUAL_DEFAULT_VALUE = ""


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
	distances = heapq.nsmallest(k, distances, key=lambda tup: tup[1])
	# get the k closest neighbors, (tup[0] is index).
	knn_indexes = [tup[0] for tup in distances]
	# iterate over the knn indexes, get the specified index's row, and form it get the label.
	knn_labels = [data[knn_current_index][1] for knn_current_index in knn_indexes]
	point_prediction = most_frequent(knn_labels, EQUAL_DEFAULT_VALUE)
	return point_prediction


class KNN(ModelAlgorithm):
	def __init__(self, k):
		super(KNN, self).__init__()
		self.k = k

	def get_accuracy_on_test(self, train, test):
		"""
		get accuracy on test by train and algorithm.
		:param train: train data.
		:param test: test data.
		:return: accuracy
		"""
		tags = get_tags_from_data(train)
		# sum all the right answers (prediction = tag(example[1]).
		global EQUAL_DEFAULT_VALUE
		EQUAL_DEFAULT_VALUE = first_time_most_frequent(tags)
		correct_answers = sum([get_knn_prediction(example, train, self.k) == example[1] for example in test])
		return correct_answers / len(test)


k_cross = 5
knn = 5

if __name__ == "__main__":
	data = parse_data(DATASET_PATH)
	algorithm = KNN(knn)
	k_cross_validation_acu(algorithm, data, k_cross)

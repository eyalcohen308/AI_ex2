from utils import *

DATASET_PATH = "./dataset.txt"


def get_knn(point, data, k):
	'''
	get k nearest neighbors by finding the k closest neighbors by hamming distance.
	:param point: the center, the data want to be predicted.
	:param data: dataset rows, each row is tuple of ([features],label).
	:param k: number of neighbors.
	:return: point prediction.
	'''
	'''
	create a tuple list of index and hamming distance between given point to the entire data.
	example: (1, 11), 1 = index, 11 = hamming distance.
	'''
	distances = [(i, hamming_distance(point, train_row[0])) for i, train_row in enumerate(data)]
	# sort the distances by hamming distance, ascending.
	distances.sort(key=lambda tup: tup[1])
	# get the k closest neighbors, (tup[0] is index).
	knn_indexes = [tup[0] for tup in distances[:k]]
	# iterate over the knn indexes, get the specified index's row, and form it get the label.
	knn_labels = [data[knn_current_index][1] for knn_current_index in knn_indexes]
	point_prediction = most_frequent(knn_labels)
	return point_prediction


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Deep ex3")
	parser.add_argument("--k_num", help="choose k for k nearest neighbors. default k is 5", type=int, default=5)
	args = parser.parse_args()
	data = parse_data(DATASET_PATH)
	point = data[0][0]
	get_knn(point, data, args.k_num)

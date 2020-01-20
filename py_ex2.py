from utils import *
from knn import KNN
from NativeBase import NaiveBase
from DecisionTree import DecisionTree

k_cross = 5
k_knn = 5
accuracy_file_path = "accuracy.txt"
if __name__ == "__main__":
	data = parse_data(DATASET_PATH)
	knn = KNN(k_knn)
	naive_base = NaiveBase()
	decision_tree = DecisionTree(data)

	algorithms = [decision_tree, knn, naive_base]
	with open(accuracy_file_path, 'w') as file:
		accuracies = []
		for algorithm in algorithms:
			algorithm_accuracy = k_cross_validation_acu(algorithm, data, k_cross)
			accuracies.append(algorithm_accuracy)

		file.write("\t".join(accuracies))

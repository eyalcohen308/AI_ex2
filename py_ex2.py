from utils import *
from knn import KNN
from NativeBase import NaiveBase
from DecisionTree import DecisionTree

k_cross = 5
k_knn = 5
accuracy_file_path = "accuracy.txt"
output_file_path = "output.txt"
train_file_path = "train.txt"
test_file_path = "test.txt"
tree_file_path = "tree.txt"

to_create_accuracy_file = True
to_create_tree_file = False
to_create_output_file = True


def create_output_file():
	"""
	create the output file.
	:return: none.
	"""
	train, feature_names = parse_data(train_file_path, with_attributes_names=True)
	test = parse_data(test_file_path)
	I2F = get_I2F(feature_names)
	knn = KNN(k_knn)
	naive_base = NaiveBase()
	decision_tree = DecisionTree(train)

	decision_tree.save_tree_to_file(I2F, output_file_path)

	algorithms = [decision_tree, knn, naive_base]
	accuracies = []
	with open(output_file_path, 'a') as output_file:
		for algorithm in algorithms:
			algorithm_accuracy = "{0:.2f}".format(algorithm.get_accuracy_on_test(train, test))
			accuracies.append(algorithm_accuracy)
		output_file.write("\n")
		output_file.write("\t".join(accuracies))


def save_accuracies_to_file(decision_tree, knn, naive_base, accuracy_file_path, append=False):
	"""
	save accuracies to file.
	:param decision_tree: tree to decide with.
	:param knn: knn algorithm.
	:param naive_base: naive base algorithm.
	:param accuracy_file_path: the file path.
	:param append: if one wants to append file.
	:return: none.
	"""
	file_option = 'a' if append else 'w'
	algorithms = [decision_tree, knn, naive_base]
	with open(accuracy_file_path, file_option) as file:
		accuracies = []
		for algorithm in algorithms:
			algorithm_accuracy = k_cross_validation_acu(algorithm, data, k_cross)
			accuracies.append(algorithm_accuracy)

		file.write("\t".join(accuracies))


if __name__ == "__main__":
	data, feature_names = parse_data(DATASET_PATH, with_attributes_names=True)
	I2F = get_I2F(feature_names)
	knn = KNN(k_knn)
	naive_base = NaiveBase()
	decision_tree = DecisionTree(data)

	if to_create_accuracy_file:
		print("-----------------  Create k fold prediction  -----------------\n")
		save_accuracies_to_file(decision_tree, knn, naive_base, accuracy_file_path, append=False)

	if to_create_tree_file:
		print("-----------------  Create tree file  -----------------\n")
		decision_tree.save_tree_to_file(I2F, tree_file_path)
	if to_create_output_file:
		print("-----------------  Create output file  -----------------\n")
	create_output_file()

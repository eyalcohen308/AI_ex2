from Algorithms import ModelAlgorithm
from utils import *


def get_naive_base_prediction(point, features_dicts):
	"""
	get prediction by naive base algorithm
	:type features_dicts: Dicts
	:param point: the center, the data want to be predicted.
	:param features_dicts: dicts that maps from feature to its

	:return: point prediction.
	"""

	point_yes_prob = 1
	point_no_prob = 1
	point_features = point[0]
	# Calculate pr(x_1,x_2, ... ,x_k | label = yes) * pr(label = yes)
	for index, feature in enumerate(point_features):
		# Calculate pr(x_1,x_2, ... ,x_k | label = yes)
		point_yes_prob *= features_dicts.get_prob_feature(feature, index, "yes")
	# point_yes_prob =  pr(x_1,x_2, ... ,x_k | label = yes) * pr(label = yes)
	point_yes_prob = point_yes_prob * features_dicts.get_prior("yes")

	# Calculate pr(x_1,x_2, ... ,x_k | label = yes) * pr(label = yes)
	for index, feature in enumerate(point_features):
		# Calculate pr(x_1,x_2, ... ,x_k | label = yes)
		point_no_prob *= features_dicts.get_prob_feature(feature, index, "no")
	# point_no_prob =  pr(x_1,x_2, ... ,x_k | label = no) * pr(label = no)
	point_no_prob = point_no_prob * features_dicts.get_prior("no")

	return "yes" if point_yes_prob >= point_no_prob else "no"


class NaiveBase(ModelAlgorithm):
	def __init__(self):
		super(NaiveBase, self).__init__()

	def get_accuracy_on_test(self, train, test):
		dicts = Dicts(train)
		correct_answers = sum([get_naive_base_prediction(example, dicts) == example[1] for example in test])

		return correct_answers / len(test)


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="AI Ex3")
	parser.add_argument("--k_cross", help="choose k for k cross validation. default k is 5", type=int, default=5)
	args = parser.parse_args()
	data = parse_data(DATASET_PATH)
	algorithm = NaiveBase()
	k_cross_validation_acu(algorithm, data, args.k_cross)

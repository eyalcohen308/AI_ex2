def hamming_distance(point1, point2):
	"""Calculate the Hamming distance between two bit strings"""
	assert len(point1) == len(point2), "Point do not have the same number of features"
	return sum(c1 != c2 for c1, c2 in zip(point1, point2))


def most_frequent(List):
	return max(set(List), key=List.count)


if __name__ == "__main__":
	string_input = " "
	while string_input:
		string_input = input("Enter two strings separate by space:\n")
		assert len(string_input.split()) == 2, "Error - did not enter two string separated with space"
		text1, text2 = string_input.split()
		print("Hamming Distance between '{0}' and '{1}' is: {2}".format(text1, text2, hamming_distance(text1, text2)))

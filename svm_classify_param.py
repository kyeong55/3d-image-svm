import numpy as np
import pandas as pd

path_data = "data/"
path_model = "model/"
path_out = "out/"
csv_file = path_data + "Parameter_data.csv"


def load_data():
	global csv_file
	df = pd.read_csv(csv_file)

	dataset_num = [149,95,113]
	trainset_num = [104,66,77]

	X = [[],[],[]]
	Y = []
	labels = df.values[:,0]
	features = df.values[:,1:]
	for i in range(len(labels)):
		label = labels[i]
		feature = features[i]
		if "B" in label:
			X[0].append(feature)
		elif "CD4" in label:
			X[1].append(feature)
		elif "CD8" in label:
			X[2].append(feature)
	for i in range(3):
		X[i] = np.array(X[i])
		Y.append()
	X = np.array(df.values[:,1:])
	Y = np.array(Y)

	return X, Y

load_data()
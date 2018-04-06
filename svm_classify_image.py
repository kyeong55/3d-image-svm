import h5py
# import os
import numpy as np
# from scipy import misc
# from scipy import ndimage
# from data.voxelgrid import VoxelGrid
# from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.externals import joblib

path_data = "data/"
path_model = "model/"
path_out = "out/"
datafile_X_b = path_data + "dataset_input_b.npy"
datafile_X_cd4 = path_data + "dataset_input_cd4.npy"
datafile_X_cd8 = path_data + "dataset_input_cd8.npy"
# datafile_Y_bt = path_data + "dataset_label_bt.npy"
# datafile_Y_cd = path_data + "dataset_label_cd.npy"
labels_all = ['B','CD4','CD8']
labels_bt = ['B','T']
labels_cd = ['CD4','CD8']

def normalize_data():
	global path_data, datafile_X_b, datafile_X_cd4, datafile_X_cd8
	# X = np.array()
	with h5py.File(path_data+'dataset.h5', 'r') as hf:
		N = len(hf)
		shape = (N, np.shape(np.array(hf['0']['data']))[0]**3)
		# if flatten:
		# 	shape = (N, np.shape(np.array(hf['0']['data']))[0]**3)
		# else:
		# 	shape = (N,) + np.shape(np.array(hf['0']['data']))
		X_b = []
		X_cd4 = []
		X_cd8 = []

		for data_num in hf:
			label = hf[data_num]['data'].attrs['label']
			# original_name = hf[data_num]['data'].attrs['original_name']
			# id = hf[data_num]['data'].attrs['id']
			# if flatten:
			# 	X[id] = np.array(hf[data_num]['data']).flatten()
			# else:
			# 	X[id] = np.array(hf[data_num]['data'])
			x = np.array(hf[data_num]['data']).flatten()
			y = labels_all.index(label)
			if y == 0:
				X_b.append(x)
			elif y == 1:
				X_cd4.append(x)
			else:
				X_cd8.append(x)

		X_b = np.array(X_b)
		X_cd4 = np.array(X_cd4)
		X_cd8 = np.array(X_cd8)

		max_X = max(np.max(X_b),np.max(X_cd4),np.max(X_cd8))
		min_X = min(np.min(X_b),np.min(X_cd4),np.min(X_cd8))
		X_b = (X_b - np.ones(np.shape(X_b))*min_X) * (1.0/(max_X-min_X))
		X_cd4 = (X_cd4 - np.ones(np.shape(X_cd4))*min_X) * (1.0/(max_X-min_X))
		X_cd8 = (X_cd8 - np.ones(np.shape(X_cd8))*min_X) * (1.0/(max_X-min_X))
		np.save(datafile_X_b,X_b)
		np.save(datafile_X_cd4,X_cd4)
		np.save(datafile_X_cd8,X_cd8)

def load_data(datafile_X,label):
	X = np.load(datafile_X)
	Y = np.ones(len(X))*label
	return X, Y

def train_SVC(train_X,train_Y,k):
	model = svm.SVC(kernel=k,decision_function_shape='ovo', verbose=False)
	model.fit(train_X, train_Y)
	return model

def train_NuSVC(train_X,train_Y,k):
	model = svm.NuSVC(kernel=k, decision_function_shape='ovo', verbose=False)
	model.fit(train_X, train_Y)
	return model

def save_model(filename, model):
	joblib.dump(model, filename)

def load_model(filename):
	return joblib.load(filename)

def evaluate(model, test_X, test_Y):
	L = np.max(test_Y) + 1
	result = np.zeros((L,L))
	predict_Y = model.predict(test_X)
	for i in range(len(test_X)):
		pred_y = int(predict_Y[i])
		y = int(test_Y[i])
		result[pred_y][y] += 1

	print (result)

	return np.sum(np.diag(result)) / np.sum(result)

def run(train_model, kernel):#, model_load, model_file):
	trainset_num = 57
	print(str(train_model)+" [ Kernel: "+kernel+" ]")
	
	X, Y = load_data(datafile_X_all,datafile_Y_all)
	N = len(X)
	D = len(X[0])
	
	# Shuffle data
	XY = np.concatenate((X,Y.reshape((N,1))), axis=1)
	np.random.shuffle(XY)
	X = np.split(XY,[D],axis=1)[0]
	Y = np.split(XY,[D],axis=1)[1].reshape((N,))
	
	offset = 0
	count = 0

	while offset < N:
		print("CV Fold: "+str(count))
		offset_next = min(offset + trainset_num, N)
		train_X = np.concatenate((X[0:offset],X[offset_next:N]), axis=0)
		train_Y = np.concatenate((Y[0:offset],Y[offset_next:N]), axis=0)
		test_X = X[offset:offset_next]
		test_Y = Y[offset:offset_next]
		model = train_model(train_X,train_Y,kernel)
		print("--- Performance: "+str(evaluate(model, test_X, test_Y)))
		offset = offset_next
		count += 1
	# if model_load:
	# 	model = load_model(model_file)
	# else:
	# 	print("##### Train Data #####")
	# 	model = train_model(X,Y,kernel)
	# 	save_model(model_file, model)
	# print("##### Evaluate #####")
	# print("Performance: evaluate(model,X,Y)")

def run2(train_model, kernel):
	global datafile_X_b, datafile_X_cd4, datafile_X_cd8
	global labels_all, labels_bt, labels_cd

	dataset_num = [149,95,113]
	trainset_num = [104,66,77]

	X_b, Y_b = load_data(datafile_X_b,0)
	X_cd4, Y_cd4 = load_data(datafile_X_cd4,1)
	X_cd8, Y_cd8 = load_data(datafile_X_cd8,2)
	_, Y_cd4_ = load_data(datafile_X_cd4,0)
	_, Y_cd8_ = load_data(datafile_X_cd8,1)

	X_train_b = X_b[0:trainset_num[0]]
	X_test_b = X_b[trainset_num[0]:]
	Y_train_b = Y_b[0:trainset_num[0]]
	Y_test_b = Y_b[trainset_num[0]:]

	X_train_cd4 = X_cd4[0:trainset_num[1]]
	X_test_cd4 = X_cd4[trainset_num[1]:]
	Y_train_cd4 = Y_cd4[0:trainset_num[1]]
	Y_test_cd4 = Y_cd4[trainset_num[1]:]
	Y_train_cd4_ = Y_cd4_[0:trainset_num[1]]
	Y_test_cd4_ = Y_cd4_[trainset_num[1]:]

	X_train_cd8 = X_cd8[0:trainset_num[2]]
	X_test_cd8 = X_cd8[trainset_num[2]:]
	Y_train_cd8 = Y_cd8[0:trainset_num[2]]
	Y_test_cd8 = Y_cd8[trainset_num[2]:]
	Y_train_cd8_ = Y_cd8_[0:trainset_num[2]]
	Y_test_cd8_ = Y_cd8_[trainset_num[2]:]

	X_train_all = [X_train_b, X_train_cd4, X_train_cd8]
	Y_train_all = [Y_train_b, Y_train_cd4, Y_train_cd8]
	X_test_all = [X_test_b, X_test_cd4, X_test_cd8]
	Y_test_all = [Y_test_b, Y_test_cd4, Y_test_cd8]

	X_train_BT = [X_train_b, X_train_cd4, X_train_cd8]
	Y_train_BT = [Y_train_b, Y_train_cd4, Y_train_cd8_]
	X_test_BT = [X_test_b, X_test_cd4, X_test_cd8]
	Y_test_BT = [Y_test_b, Y_test_cd4, Y_test_cd8_]

	X_train_CD = [X_train_cd4, X_train_cd8]
	Y_train_CD = [Y_train_cd4_, Y_train_cd8_]
	X_test_CD = [X_test_cd4, X_test_cd8]
	Y_test_CD = [Y_test_cd4_, Y_test_cd8_]

	print(str(train_model)+" [ Kernel: "+kernel+" ]")

	eval_all = ["All",X_train_all,Y_train_all,X_test_all,Y_test_all]
	eval_bt = ["B vs T",X_train_BT,Y_train_BT,X_test_BT,Y_test_BT]
	eval_cd = ["CD4 vs CD8",X_train_CD,Y_train_CD,X_test_CD,Y_test_CD]

	for eval in [eval_all,eval_bt,eval_cd]:
		print(" === "+eval[0])

		train_X = np.concatenate(eval[1],axis=0)
		train_Y = np.concatenate(eval[2],axis=0)
		test_X = np.concatenate(eval[3],axis=0)
		test_Y = np.concatenate(eval[4],axis=0)

		model = train_model(train_X,train_Y,kernel)
		print("\tPerformance: "+str(evaluate(model, test_X, test_Y)))

# normalize_data()

# run(train_model=train_SVC, kernel='rbf', model_load=False, model_file=path_model + "svm_svc_rbf.pkl")
# run(train_model=train_SVC, kernel='linear', model_load=False, model_file=path_model + "svm_svc_linear.pkl")
# run(train_model=train_SVC, kernel='poly', model_load=False, model_file=path_model + "svm_svc_poly.pkl")
# run(train_model=train_SVC, kernel='sigmoid', model_load=False, model_file=path_model + "svm_svc_sigmoid.pkl")
# run(train_model=train_SVC, kernel='precomputed', model_load=False, model_file=path_model + "svm_svc_precomputed.pkl")

# run(train_model=train_NuSVC, kernel='rbf')#, model_load=False, model_file=path_model + "svm_nusvc_rbf.pkl")
# run(train_model=train_NuSVC, kernel='linear')#, model_load=False, model_file=path_model + "svm_nusvc_linear.pkl")
# run(train_model=train_NuSVC, kernel='poly')#, model_load=False, model_file=path_model + "svm_nusvc_poly.pkl")
# run(train_model=train_NuSVC, kernel='sigmoid', model_load=False, model_file=path_model + "svm_nusvc_sigmoid.pkl")
# run(train_model=train_NuSVC, kernel='precomputed', model_load=False, model_file=path_model + "svm_nusvc_precomputed.pkl")

# run2(train_model=train_NuSVC, kernel='rbf')
# run2(train_model=train_NuSVC, kernel='linear')
# run2(train_model=train_NuSVC, kernel='poly')
run2(train_model=train_NuSVC, kernel='sigmoid')
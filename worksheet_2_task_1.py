""" For worksheet_2 """


import numpy as np
from scipy import ndimage
from scipy.spatial import distance
from scipy import signal
import natsort as ns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

import dataset as ds








def load_feat_list(path, metadata_barefoot_path, metadata_aligned_foot_path, data_aligned_foot_path, feat_len=37):
	""" Loads the feature list from disk and populate the dataset accessing structure """
	# create preprocessed raw data
	dsobj = ds.DatasetRaw(metadata_barefoot_path, metadata_aligned_foot_path, data_aligned_foot_path)
	dsobj.preprocess()
	dsobj.create_per_person_per_foot_data_structure()
	# find features and delete raw data to create space
	dsfeatobj = ds.DatasetFeat(dsobj)
	feat_list = np.load(path)
	iter = 0
	feat_list_exp = []		# expanded feature list
	assert feat_len == feat_list.shape[-1]
	for i in ns.natsorted(dsobj.person_dict.keys()):
		for j in (0,1):
			tmp_lst = []
			for k in range(len(dsobj.person_dict[i][j])):	# for each foot
				tmp_lst.append(feat_list[iter])
				feat_list_exp.append(np.hstack((i,j,feat_list[iter])))		# person, left/right, feat_vec
				iter += 1
			dsfeatobj.person_dict[i][j] = np.array(tmp_lst)
	del dsobj, feat_list
	return dsfeatobj, np.array(feat_list_exp)

def divide_train_test(dsfeatobj, side='whole', ratio=0.5):
	"""
		@side: 'whole' for both left & right foot, 'left' for left foot only, 'right' for right foot only. Default is 'whole'.
	"""
	if side == 'whole':
		foot_tuple = (0,1)
	elif side == 'left':
		foot_tuple = (0,)
	elif side == 'right':
		foot_tuple = (1,)
	else:
		assert False, 'unknown foot type'
	X_train, X_test = None, None
	y_train, y_test = None, None
	for i in ns.natsorted(dsfeatobj.person_dict.keys()):
		for j in foot_tuple:
			midx = np.math.ceil( len(dsfeatobj.person_dict[i][j])*ratio )
			tr_arr = dsfeatobj.person_dict[i][j][:midx]
			te_arr = dsfeatobj.person_dict[i][j][midx:]
			if X_train is None:
				X_train = tr_arr
				y_train = np.repeat(i*10+j,len(tr_arr))
				X_test = te_arr
				y_test = np.repeat(i*10+j,len(te_arr))
			else:
				X_train = np.vstack((X_train, tr_arr))
				y_train = np.concatenate((y_train, np.repeat(i*10+j,len(tr_arr))), axis=0)
				X_test = np.vstack((X_test, te_arr))
				y_test = np.concatenate((y_test, np.repeat(i*10+j,len(te_arr))), axis=0)
	return X_train, y_train, X_test, y_test

def normalize_data(X_train, X_test):
	""" scaling data with zero mean and unit variance """
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)
	return X_train, X_test

def apply_PCA(X_train, X_test):
	""" Applying PCA with explained_variance_ratio of 99.07 """
	pca = PCA(n_components=19, svd_solver='full')
	X_train_transformed = pca.fit_transform(X_train)
	print('variance = ', np.sum(pca.explained_variance_ratio_))
	X_test_transformed = pca.transform(X_test)
	return X_train_transformed, X_test_transformed

def apply_SVM(X_train, y_train, X_test, y_test):
	""" Do LinearSVM based classification in one vs rest setting. """
	clf = LinearSVC(random_state=0, tol=1e-5)
	clf = OneVsRestClassifier(clf, n_jobs=4)
	clf.fit(X_train, y_train)
	y_scores = clf.decision_function(X_test)
	y_hat_test = clf.predict(X_test)
	print('SVM one-vs-rest score = ', clf.score(X_test,y_test))

def apply_LDA(X_train, y_train, X_test, y_test, apply_one_vs_rest=False):
	""" Do LDA based classification in one vs rest setting. """
	clf = LinearDiscriminantAnalysis()
	if apply_one_vs_rest: clf = OneVsRestClassifier(clf, n_jobs=4)
	clf.fit(X_train, y_train)
	y_scores = clf.decision_function(X_test)
	y_hat_test = clf.predict(X_test)
	print('LDA score = ', clf.score(X_test,y_test))

def apply_MLP(X_train, y_train, X_test, y_test, apply_one_vs_rest=False):
	""" Do MLP based classification in one vs rest setting. """
	clf = MLPClassifier(hidden_layer_sizes=(100,256), random_state=1, max_iter=300, learning_rate_init=0.001)
	if apply_one_vs_rest: clf = OneVsRestClassifier(clf, n_jobs=4)
	clf.fit(X_train, y_train)
	y_scores = clf.predict_proba(X_test)
	y_hat_test = clf.predict(X_test)
	print('MLP score = ', clf.score(X_test,y_test))






if __name__ == "__main__":
	metadata_barefoot_path = './data/perFootDataBarefoot/PerFootMetaDataBarefoot.npy'
	metadata_aligned_foot_path = './data/alignedPerFootDataBarefoot/AlignedFootMetaDataBarefoot.npy'
	data_aligned_foot_path = './data/alignedPerFootDataBarefoot/AlignedFootDataBarefoot.npz'
	feat_list_path = './features_mat/feat_list_37.npy'

	dsfeatobj, feat_list_exp = load_feat_list(feat_list_path, metadata_barefoot_path, metadata_aligned_foot_path, data_aligned_foot_path, feat_len=37)		# load feat_list
	X_train, y_train, X_test, y_test = divide_train_test(dsfeatobj, side='left', ratio=0.5)
	X_train, X_test = normalize_data(X_train, X_test)
	X_train, X_test = apply_PCA(X_train, X_test)
	apply_SVM(X_train, y_train, X_test, y_test)
	apply_LDA(X_train, y_train, X_test, y_test, apply_one_vs_rest=True)
	apply_MLP(X_train, y_train, X_test, y_test, apply_one_vs_rest=True)







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
import worksheet_1_task_2_enrollment as ws1t2en
import metrics








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

def divide_train_test(dsfeatobj, side='left', ratio=0.5):
	"""
		Divide the dataset into train and test sets and return the structures for accessing them.
		@side: 'left' for left foot only, 'right' for right foot only. Default is 'left'.
	"""
	if side == 'left':
		side = 0
	elif side == 'right':
		side = 1
	else:
		assert False, 'unknown foot type'
	dsfeatobj_l1_tr, dsfeatobj_l1_te = {}, {}
	for i in ns.natsorted(dsfeatobj.person_dict.keys()):
		midx = np.math.ceil( len(dsfeatobj.person_dict[i][side])*ratio )
		dsfeatobj_l1_tr[i] = dsfeatobj.person_dict[i][side][:midx]
		dsfeatobj_l1_te[i] = dsfeatobj.person_dict[i][side][midx:]
	return dsfeatobj_l1_tr, dsfeatobj_l1_te

def find_genuine_imposter(subject_id, trial_id, dsfeatobj_l1_tr, include_self=False):
	""" Return sets of genuine and imposter subjects for specific query (subject_id and trial_id)  """
	genuine_subjects = []
	imposter_subjects = None
	for t_id in range(len(dsfeatobj_l1_tr[subject_id])):	# iterate over each foot/trial
		if t_id != trial_id or include_self:
			genuine_subjects.append(dsfeatobj_l1_tr[subject_id][t_id])
	for s_id in ns.natsorted(dsfeatobj_l1_tr.keys()):
		if s_id != subject_id:
			if imposter_subjects is None:		imposter_subjects = dsfeatobj_l1_tr[s_id]
			else:		imposter_subjects = np.vstack((imposter_subjects, dsfeatobj_l1_tr[s_id]))
	query_feature = dsfeatobj_l1_tr[subject_id][trial_id]
	return np.array(genuine_subjects), imposter_subjects, query_feature

def find_mean_distance(v1,sub,dsfeatobj_l1_tr,dist_type='euclidean'):
	"""
		Finds the mean of distances of a test subject sample from all ot the same subject training data samples.
		type: 'euclidean' or 'corr coef' for correlation coefficient
	"""
	if dist_type=='euclidean':		dist = lambda v1, v2: distance.euclidean(v1,v2)
	elif dist_type=='corr coef':		dist = lambda v1, v2: np.sum((v1-np.mean(v1))*(v2-np.mean(v2)))/(np.std(v1)*np.std(v2))
	else: assert False, 'Unknown distance type'
	dist_vec = []
	for v2 in dsfeatobj_l1_tr[sub]:	# iterate over all training trials of subject='sub'
		dist_vec.append(dist(v1,v2))
	return np.array(dist_vec).mean()






def find_per_subject_threshold(dsfeatobj_l1_tr, dist_type='euclidean'):
	"""
		Find mean threshold for each subject in the training data by finding the
		thresholds at the EER for each foot/trial and taking their average

		dist_type: 'euclidean', 'corr coef'
	"""
	dsfeatobj_l1_th = {}
	for sub in ns.natsorted(dsfeatobj_l1_tr.keys()):	# iterating over subjects
		tmp_th_lst = []
		for trial_id in range(len(dsfeatobj_l1_tr[sub])):	# iterating over each foot
			genuine_subjects, imposter_subjects, query_feature = find_genuine_imposter(sub,trial_id,dsfeatobj_l1_tr)
			scores, labels = ws1t2en.find_distance(query_feature, genuine_subjects, imposter_subjects, type=dist_type)
			if dist_type=='euclidean':	lower_is_closer = True
			elif dist_type=='corr coef':	lower_is_closer = False
			else: assert False, 'Unknown distance type'
			FRR_series, FAR_series, th_series = metrics.find_FRR_FAR_series(scores, labels, th_range=(np.min(scores)-1, np.max(scores)+1), interval_num=100, lower_is_closer=lower_is_closer)
			_, th_at_EER, _, _ = metrics.find_EER(FRR_series, FAR_series, th_series)
			tmp_th_lst.append(th_at_EER)
		dsfeatobj_l1_th[sub] = np.array(tmp_th_lst).mean()		# assign subject-wise mean threshold
	return dsfeatobj_l1_th

def multi_template_classification(dsfeatobj_l1_tr, dsfeatobj_l1_te, dsfeatobj_l1_th, dist_type='euclidean'):
	"""
		Make predictions based on individual (subjectwise-trialwise) thresholds
		Binary predictions: 1 if correctly verified, 0 incorrectly verified

		Using mean of thresholds for a subject, as implemented by 'find_per_sample_threshold()'.

		dist_type: 'euclidean', 'corr coef'
	"""
	# iterate over the testing samples
	# find classification using thresholding on the mean distance from the genuine data test_sets
	# note down the predicted labels
	y_test_predicted = []
	for sub in ns.natsorted(dsfeatobj_l1_te.keys()):	# iterating over subjects
		for v1 in dsfeatobj_l1_te[sub]:			# iterate over each foot/trial
			mean_dist = find_mean_distance(v1,sub,dsfeatobj_l1_tr,dist_type=dist_type)
			mean_th = dsfeatobj_l1_th[sub]
			if dist_type == 'euclidean':	# case: smaller is closer
				if mean_dist > mean_th: y_test_predicted.append(0)	# incorrect classification
				else: y_test_predicted.append(1)					# correct classification
			elif dist_type == 'corr coef':	# case: larger is closer
				if mean_dist < mean_th: y_test_predicted.append(0)	# incorrect classification
				else: y_test_predicted.append(1)					# correct classification
			else:
				assert False, 'Unknown dist type'
	y_test_predicted = np.array(y_test_predicted)
	y_test_original = np.repeat(1, len(y_test_predicted))
	return y_test_original, y_test_predicted

def show_performance(y_test_original, y_test_predicted):
	""" True positive rate (TPR), False negative rate (FNR) """
	true_positive = np.sum(y_test_predicted)
	false_negative = len(y_test_predicted) - true_positive
	TPR = true_positive/len(y_test_predicted)
	FNR = 1-TPR
	print('TPR = ', TPR)
	print('FNR = ', FNR)







if __name__ == "__main__":
	metadata_barefoot_path = './data/perFootDataBarefoot/PerFootMetaDataBarefoot.npy'
	metadata_aligned_foot_path = './data/alignedPerFootDataBarefoot/AlignedFootMetaDataBarefoot.npy'
	data_aligned_foot_path = './data/alignedPerFootDataBarefoot/AlignedFootDataBarefoot.npz'
	feat_list_path = './features_mat/feat_list_37.npy'

	dist_type='euclidean'
	side = 'left'
	dsfeatobj, feat_list_exp = load_feat_list(feat_list_path, metadata_barefoot_path, metadata_aligned_foot_path, data_aligned_foot_path, feat_len=37)		# load feat_list
	dsfeatobj_l1_tr, dsfeatobj_l1_te = divide_train_test(dsfeatobj, side=side, ratio=0.5)

	dsfeatobj_l1_th = find_per_subject_threshold(dsfeatobj_l1_tr, dist_type=dist_type)
	y_test_original, y_test_predicted = multi_template_classification(dsfeatobj_l1_tr, dsfeatobj_l1_te, dsfeatobj_l1_th, dist_type=dist_type)
	show_performance(y_test_original, y_test_predicted)






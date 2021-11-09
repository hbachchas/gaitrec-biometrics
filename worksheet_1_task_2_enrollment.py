""" For worksheet_1 """


import numpy as np
from numpy.core.records import array
from numpy.lib.function_base import append
from scipy import ndimage
from scipy.sparse.construct import vstack
from scipy.spatial import distance
from scipy import signal
import natsort as ns

import dataset as ds
import metrics
import worksheet_1_task_1_cop_measures as ws1t1cm







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

def find_genuine_imposter(subject_id, foot_side, trial_id, dsfeatobj, include_self=False):
	""" Find genuine/imposter feature vectors corresponding to a given feature vector with id = 'trial_id' """
	genuine_subjects = []
	imposter_subjects = None
	for t_id in range(len(dsfeatobj.person_dict[subject_id][foot_side])):
		if t_id != trial_id or include_self:
			genuine_subjects.append(dsfeatobj.person_dict[subject_id][foot_side][t_id])
		else:
			query_feature = dsfeatobj.person_dict[subject_id][foot_side][t_id]
	for s_id in ns.natsorted(dsfeatobj.person_dict.keys()):
		if s_id != subject_id:
			if imposter_subjects is None:		imposter_subjects = dsfeatobj.person_dict[s_id][foot_side]
			else:		imposter_subjects = np.vstack((imposter_subjects, dsfeatobj.person_dict[s_id][foot_side]))
	query_feature = dsfeatobj.person_dict[subject_id][foot_side][trial_id]
	return np.array(genuine_subjects), imposter_subjects, query_feature

def find_distance(v1, genuine_subjects, imposter_subjects, type='euclidean', combined_score=True):
	"""
		Find distance of a feature vector from its genuine/imposter subjects
		type: 'euclidean' or 'corr coef' for correlation coefficient
	"""
	if type=='euclidean':		dist = lambda v1, v2: distance.euclidean(v1,v2)
	elif type=='corr coef':		dist = lambda v1, v2: np.sum((v1-np.mean(v1))*(v2-np.mean(v2)))/(np.std(v1)*np.std(v2))
	else: assert False, 'Unknown distance type'
	if combined_score:
		scores = []
		labels = np.hstack((np.repeat(1,len(genuine_subjects)), np.repeat(0,len(imposter_subjects))))
		for v2 in genuine_subjects:
			scores.append(dist(v1,v2))
		for v2 in imposter_subjects:
			scores.append(dist(v1,v2))
		return np.array(scores), labels
	else:
		genuine_scores, imposter_scores = [], []
		genuine_labels, imposter_labels = np.repeat(1,len(genuine_subjects)), np.repeat(0,len(imposter_subjects))
		for v2 in genuine_subjects:
			genuine_scores.append(dist(v1,v2))
		for v2 in imposter_subjects:
			imposter_scores.append(dist(v1,v2))
		return np.array(genuine_scores), np.array(imposter_scores), genuine_labels, imposter_labels





def task_1_example_of_functionality_for_an_individual():
	"""
		This method shows:
		1. how to calculate FAR, FRR, EER, accuracy plot, ROC plot for an individual
		2. how to find genuine and imposter person distance based on (Eucledian or Correlation coefficient)
	"""
	metadata_barefoot_path = './data/perFootDataBarefoot/PerFootMetaDataBarefoot.npy'
	metadata_aligned_foot_path = './data/alignedPerFootDataBarefoot/AlignedFootMetaDataBarefoot.npy'
	data_aligned_foot_path = './data/alignedPerFootDataBarefoot/AlignedFootDataBarefoot.npz'
	feat_list_path = './features_mat/feat_list_37.npy'

	dsfeatobj, feat_list_exp = load_feat_list(feat_list_path, metadata_barefoot_path, metadata_aligned_foot_path, data_aligned_foot_path, feat_len=37)		# load feat_list
	genuine_subjects, imposter_subjects, query_feature = find_genuine_imposter(4,0,0,dsfeatobj)		# subject:4, foot:left, trial_id:0

	scores, labels = find_distance(query_feature, genuine_subjects, imposter_subjects, type='euclidean')

	th = np.mean(scores)
	FRR, FAR = metrics.find_FRR_FAR(scores, labels, th, lower_is_closer=True)
	FRR_series, FAR_series, th_series = metrics.find_FRR_FAR_series(scores, labels, th_range=(np.min(scores)-5, np.max(scores)+5), interval_num=100, lower_is_closer=True)
	EER, th_at_EER, FRR_at_EER, FAR_at_EER = metrics.find_EER(FRR_series, FAR_series, th_series)

	metrics.draw_accuracy_plot(FRR_series, FAR_series, th_series)
	metrics.draw_ROC_plot(FRR_series, FAR_series)




def worksheet_1_checkpoint():
	"""
		This method presents the checkpointing for MDIST_features
		The final genuine/imposter matrices are saved on the disk
	"""
	metadata_barefoot_path = './data/perFootDataBarefoot/PerFootMetaDataBarefoot.npy'
	metadata_aligned_foot_path = './data/alignedPerFootDataBarefoot/AlignedFootMetaDataBarefoot.npy'
	data_aligned_foot_path = './data/alignedPerFootDataBarefoot/AlignedFootDataBarefoot.npz'
	feat_list_path = './features_mat/feat_list_37.npy'

	dsfeatobj, feat_list_exp = ws1t1cm.load_feat_list_MDIST(feat_list_path, metadata_barefoot_path, metadata_aligned_foot_path, data_aligned_foot_path)		# load feat_list

	# Eucledian distance
	mat_euc_genuine, mat_euc_imposter = [], []
	for i in range(len(dsfeatobj.person_dict[4][0])):
		genuine_subjects, imposter_subjects, query_feature = find_genuine_imposter(4,0,i,dsfeatobj,include_self=True)
		genuine_scores, imposter_scores, _, _ = find_distance(query_feature, genuine_subjects, imposter_subjects, type='euclidean', combined_score=False)
		mat_euc_genuine.append(genuine_scores)
		mat_euc_imposter.append(imposter_scores)
	mat_euc_genuine, mat_euc_imposter = np.array(mat_euc_genuine), np.transpose(np.array(mat_euc_imposter))

	# Correlation coefficient
	mat_cor_coef_genuine, mat_cor_coef_imposter = [], []
	for i in range(len(dsfeatobj.person_dict[4][0])):
		genuine_subjects, imposter_subjects, query_feature = find_genuine_imposter(4,0,i,dsfeatobj,include_self=True)
		genuine_scores, imposter_scores, _, _ = find_distance(query_feature, genuine_subjects, imposter_subjects, type='corr coef', combined_score=False)
		mat_cor_coef_genuine.append(genuine_scores)
		mat_cor_coef_imposter.append(imposter_scores)
	mat_cor_coef_genuine, mat_cor_coef_imposter = np.array(mat_cor_coef_genuine), np.transpose(np.array(mat_cor_coef_imposter))
	np.save('./check_point_1_mat/mat_euc_genuine.npy', mat_euc_genuine)
	np.save('./check_point_1_mat/mat_euc_imposter.npy', mat_euc_imposter)
	np.save('./check_point_1_mat/mat_cor_coef_genuine.npy', mat_cor_coef_genuine)
	np.save('./check_point_1_mat/mat_cor_coef_imposter.npy', mat_cor_coef_imposter)





if __name__ == "__main__":
	# task_1_example_of_functionality_for_an_individual()
	worksheet_1_checkpoint()

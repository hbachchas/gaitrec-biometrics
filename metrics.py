import numpy as np
from scipy.spatial import distance
import natsort as ns
import matplotlib.pyplot as plt




def find_FRR_FAR(scores, labels, th, lower_is_closer=True):
	"""
		Find FRR and FAR for individual subject at threshold 'th'
		labels: 0 for imposter, 1 for genuine
	"""
	# FRR
	genuine = scores[labels == 1]			# find genuines
	rejections = genuine[genuine > th]		# find rejections
	FRR = len(rejections)/len(genuine)	# divide by len(genuine)
	# FAR
	imposters = scores[labels == 0]			# find imposters
	accepted = imposters[imposters < th]	# find acceptance
	FAR = len(accepted)/len(imposters)	# divide by len(imposters)
	return FRR, FAR

def find_FRR_FAR_series(scores, labels, th_range, interval_num, lower_is_closer=True):
	""" Find FRR and FAR series for variable threshold for making the accuracy curve """
	th_series = np.linspace(th_range[0], th_range[1], interval_num)
	FRR_series, FAR_series = [], []
	for th in th_series:
		FRR, FAR = find_FRR_FAR(scores, labels, th)
		FRR_series.append(FRR)
		FAR_series.append(FAR)
	return np.array(FRR_series), np.array(FAR_series), th_series

def find_EER(FRR_series, FAR_series, th_series):
	""" Find EER for FRR_series and FAR_series """
	max_val = np.max(FRR_series) + np.max(FAR_series) + 100
	for i in range(len(FRR_series)):
		if FRR_series[i]+FAR_series[i] < max_val:
			min_idx = i
			max_val = FRR_series[i]+FAR_series[i]
	return (FRR_series[min_idx]+FAR_series[min_idx])/2, th_series[min_idx], FRR_series[min_idx], FAR_series[min_idx]



def vis_mat(mat):
	""" Visualize any matrix heatmap """
	plt.matshow(mat);
	plt.colorbar()
	plt.show()

def draw_accuracy_plot(y1, y2, x):
	plt.plot(x, y1, linestyle = 'dashed', color = 'r', marker='o')
	plt.plot(x, y2, linestyle = 'dashed', color = 'b', marker='x')
	plt.legend(['FRR', 'FAR'])
	plt.xlabel('Threshold')
	plt.show()

def draw_ROC_plot(FRR_series, FAR_series):
	plt.plot(FAR_series, FRR_series, linestyle = 'dashed', color = 'r', marker='o')
	plt.ylabel('FRR')
	plt.xlabel('FAR')
	plt.show()





if __name__ == "__main__":
	""" Short example of using these functions """
	scores = np.array(ns.natsorted([.1,.15,.2,.25,.35,.26,.47,.35,.39,.67,.84,.75,.98,.58,.45,.09]))
	labels = np.hstack((np.repeat(0,6),np.repeat(1,len(scores)-6)))
	th = 0.5

	FRR, FAR = find_FRR_FAR(scores, labels, th, lower_is_closer=True)
	FRR_series, FAR_series, th_series = find_FRR_FAR_series(scores, labels, th_range=(.1,.9), interval_num=9, lower_is_closer=True)
	EER, th_at_EER, FRR_at_EER, FAR_at_EER = find_EER(FRR_series, FAR_series, th_series)

	draw_accuracy_plot(FRR_series, FAR_series, th_series)
	draw_ROC_plot(FRR_series, FAR_series)



""" For worksheet_1 """


import numpy as np
from scipy import ndimage
from scipy.spatial import distance
from scipy import signal
import natsort as ns

import dataset as ds





class COPobj():
	""" Contains all COP/COA and other parameter values and different measures/features for a single subject, single foot-type """
	def __init__(self) -> None:
		self.AP = None		# sequence of AP
		self.ML = None		# sequence of ML
		self.COP = None		# sequence of COP

		self.meanAP = None
		self.meanML = None
		self.meanCOP = None

		self.refAP = None	# referenced AP, subtracting meanAP
		self.refML = None	# referenced ML, subtracting meanML
		self.refCOP = None	# referenced COP, subtracting meanCOP

		self.RD_COP = None		# sequence of RD (resultant distance)

		self.COA = None		# sequence of COA
		self.meanCOA = None
		self.refCOA = None	# referenced COP, subtracting meanCOP
		self.RD_COA = None		# sequence of RD (resultant distance)

		##### Measures #####
		self.MDIST = None	# avg. distance from meanCOP
		self.MDIST_AP = None
		self.MDIST_ML = None

		self.RDIST = None				# rms dist RD
		self.RDIST_AP = None			# rms dist AP
		self.RDIST_ML = None			# rms dist ML

		self.TOTEX = None				# total excursion COP
		self.TOTEX_AP = None			# total excursion AP
		self.TOTEX_ML = None			# total excursion ML

		self.MVELO = None				# mean velocity COP, T=1s
		self.MVELO_AP = None			# mean velocity AP, T=1s
		self.MVELO_ML = None			# mean velocity ML, T=1s

		self.range = None				# range COP
		self.range_AP = None			# range AP
		self.range_ML = None			# range ML

		self.circle_area = None			# 95% confidence circle area
		self.elipse_area = None			# 95% confidence elipse area

		self.AREA_SW = None				# sway area

		self.MFREQ = None				# mean frequency
		self.MFREQ_AP = None
		self.MFREQ_ML = None

		self.FD_PD = None				# fractal dimension
		self.FD_CC = None
		self.FD_CE = None

		self.total_power_RD = None		# total power RD
		self.total_power_AP = None
		self.total_power_ML = None

		self.power_freq50_RD = None		# 50% power frequency RD
		self.power_freq50_AP = None		# 50% power frequency AP
		self.power_freq50_ML = None		# 50% power frequency ML

		self.power_freq95_RD = None		# 95% power frequency RD
		self.power_freq95_AP = None		# 95% power frequency AP
		self.power_freq95_ML = None		# 95% power frequency ML

		self.CFREQ_RD = None			# centroidal frequency RD
		self.CFREQ_AP = None			# centroidal frequency AP
		self.CFREQ_ML = None			# centroidal frequency ML

		self.FREQD_RD = None			# frequency dispersion RD
		self.FREQD_AP = None			# frequency dispersion AP
		self.FREQD_ML = None			# frequency dispersion ML




class COPObjMeasuresOps():
	""" Defines the functions for calculating the COP measures/features """
	def __init__(self) -> None:
		pass

	def TOTEX(self, copobj):
		val = 0.0
		ap = copobj.refAP
		ml = copobj.refML
		for i in range(len(ap)-1):
			val += np.sqrt( (ap[i+1] - ap[i])**2 + (ml[i+1] - ml[i])**2 )
		return val
	def TOTEX_AP(self, copobj):
		val = 0.0
		ap = copobj.refAP
		for i in range(len(ap)-1):
			val +=  np.absolute(ap[i+1] - ap[i])
		return val
	def TOTEX_ML(self, copobj):
		val = 0.0
		ml = copobj.refML
		for i in range(len(ml)-1):
			val +=  np.absolute(ml[i+1] - ml[i])
		return val

	def MVELO(self, copobj, T):
		return copobj.TOTEX/T
	def MVELO_AP(self, copobj, T):
		return copobj.TOTEX_AP/T
	def MVELO_ML(self, copobj, T):
		return copobj.TOTEX_ML/T

	def range(self, copobj):
		d = distance.cdist(copobj.refCOP, copobj.refCOP, metric='euclidean')
		# bestpair = np.unravel_index(d.argmax(), d.shape)
		return np.max(d)
	def range_AP(self, copobj):
		return np.absolute(np.max(copobj.refAP)-np.min(copobj.refAP))
	def range_ML(self, copobj):
		return np.absolute(np.max(copobj.refML)-np.min(copobj.refML))

	def circle_area(self, copobj):
		return np.math.pi*(copobj.MDIST + 1.645*np.sqrt(copobj.RDIST**2 - copobj.MDIST**2))**2
	def elipse_area(self, copobj):
		s_ap = np.std(copobj.refAP)
		s_ml = np.std(copobj.refML)
		s_ap_ml = np.mean( np.sum(copobj.refAP*copobj.refML) )
		D = np.sqrt((s_ap**2+s_ml**2)-4*(s_ap**2*s_ml**2 - s_ap_ml**2))
		a = np.sqrt( 3.0*(s_ap**2+s_ml**2+D) )
		b = np.sqrt( 3.0*(s_ap**2+s_ml**2-D) )
		return np.math.pi*a*b

	def MFREQ(self, copobj):
		return copobj.MVELO/(2*np.math.pi*copobj.MDIST)
	def MFREQ_AP(self, copobj):
		return copobj.MVELO_AP/(4*np.sqrt(2)*copobj.MDIST_AP)
	def MFREQ_ML(self, copobj):
		return copobj.MVELO_ML/(4*np.sqrt(2)*copobj.MDIST_ML)

	def AREA_SW(self, copobj, T):
		val = 0.0
		ap = copobj.refAP
		ml = copobj.refML
		for i in range(len(ap)-1):
			val += np.absolute( ap[i+1]*ml[i]-ap[i]*ml[i+1] )
		assert val != 0
		return 1/(2*T)*val

	def FD_PD(self, copobj):
		d = distance.cdist(copobj.refCOP, copobj.refCOP, metric='euclidean')
		d = np.max(d)
		return np.math.log(len(copobj.refCOP))/np.math.log(len(copobj.refCOP)*d/copobj.TOTEX)
	def FD_CC(self, copobj):
		d = 2*(copobj.MDIST + 1.645*np.sqrt(copobj.RDIST**2 - copobj.MDIST**2))
		return np.math.log(len(copobj.refCOP))/np.math.log(len(copobj.refCOP)*d/copobj.TOTEX)
	def FD_CE(self, copobj):
		s_ap = np.std(copobj.refAP)
		s_ml = np.std(copobj.refML)
		s_ap_ml = np.mean( np.sum(copobj.refAP*copobj.refML) )
		D = np.sqrt((s_ap**2+s_ml**2)-4*(s_ap**2*s_ml**2 - s_ap_ml**2))
		a = np.sqrt( 3.0*(s_ap**2+s_ml**2+D) )
		b = np.sqrt( 3.0*(s_ap**2+s_ml**2-D) )
		d = np.sqrt(2*a*2*b)
		return np.math.log(len(copobj.refCOP))/np.math.log(len(copobj.refCOP)*d/copobj.TOTEX)

	def spectral_moment_k(self, k, i, j, freqs, G):
		return np.sum(G[i:])			# eqn 25
	def total_power_RD(self, copobj):
		rd = copobj.RD_COP
		freqs, psd = signal.welch(rd,return_onesided=False)
		return self.spectral_moment_k(k=0,i=2,j=99,freqs=freqs,G=psd)
	def total_power_AP(self, copobj):
		ap = copobj.refAP
		freqs, psd = signal.welch(ap,return_onesided=False)
		return self.spectral_moment_k(k=0,i=2,j=99,freqs=freqs,G=psd)
	def total_power_ML(self, copobj):
		ml = copobj.refML
		freqs, psd = signal.welch(ml,return_onesided=False)
		return self.spectral_moment_k(k=0,i=2,j=99,freqs=freqs,G=psd)

	def spectral_moment_k_onesided_50(self, k, i, j, freqs, G):
		val = 0.0
		mu0 = np.sum(G[i:])
		for m in range(i,len(G)):
			val += G[m]
			if val >= 0.5*mu0: break
		delta_f = freqs[m]-freqs[i-1]
		return m*delta_f
	def power_freq50_RD(self, copobj):
		rd = copobj.RD_COP
		freqs, psd = signal.welch(rd,return_onesided=True)
		return self.spectral_moment_k_onesided_50(k=0,i=2,j=len(freqs)-1,freqs=freqs,G=psd)
	def power_freq50_AP(self, copobj):
		ap = copobj.refAP
		freqs, psd = signal.welch(ap,return_onesided=True)
		return self.spectral_moment_k_onesided_50(k=0,i=2,j=len(freqs)-1,freqs=freqs,G=psd)
	def power_freq50_ML(self, copobj):
		ml = copobj.refML
		freqs, psd = signal.welch(ml,return_onesided=True)
		return self.spectral_moment_k_onesided_50(k=0,i=2,j=len(freqs)-1,freqs=freqs,G=psd)

	def spectral_moment_k_onesided_95(self, k, i, j, freqs, G):
		val = 0.0
		mu0 = np.sum(G[i:])
		for m in range(i,len(G)):
			val += G[m]
			if val >= 0.95*mu0: break
		delta_f = freqs[m]-freqs[i-1]
		return m*delta_f
	def power_freq95_RD(self, copobj):
		rd = copobj.RD_COP
		freqs, psd = signal.welch(rd,return_onesided=True)
		return self.spectral_moment_k_onesided_95(k=0,i=2,j=len(freqs)-1,freqs=freqs,G=psd)
	def power_freq95_AP(self, copobj):
		ap = copobj.refAP
		freqs, psd = signal.welch(ap,return_onesided=True)
		return self.spectral_moment_k_onesided_95(k=0,i=2,j=len(freqs)-1,freqs=freqs,G=psd)
	def power_freq95_ML(self, copobj):
		ml = copobj.refML
		freqs, psd = signal.welch(ml,return_onesided=True)
		return self.spectral_moment_k_onesided_95(k=0,i=2,j=len(freqs)-1,freqs=freqs,G=psd)

	def spectral_moment_k_onesided(self, k, i, j, freqs, G):
		val = 0.0
		for m in range(i,len(G)):
			# delta_f = freqs[m]-freqs[m-1]
			delta_f = freqs[m]-freqs[i-1]
			val += np.math.pow(m*delta_f,k)*G[m]		# eqn 25
		return val
	def CFREQ_RD(self, copobj):
		rd = copobj.RD_COP
		freqs, psd = signal.welch(rd,return_onesided=True)
		mu0 = np.sum(psd[2:])
		mu2 = self.spectral_moment_k_onesided(k=2,i=2,j=len(freqs)-1,freqs=freqs,G=psd)
		assert mu0 != 0
		return np.sqrt(mu2/mu0)
	def CFREQ_AP(self, copobj):
		ap = copobj.refAP
		freqs, psd = signal.welch(ap,return_onesided=True)
		mu0 = np.sum(psd[2:])
		mu2 = self.spectral_moment_k_onesided(k=2,i=2,j=len(freqs)-1,freqs=freqs,G=psd)
		assert mu0 != 0
		return np.sqrt(mu2/mu0)
	def CFREQ_ML(self, copobj):
		ml = copobj.refML
		freqs, psd = signal.welch(ml,return_onesided=True)
		mu0 = np.sum(psd[2:])
		mu2 = self.spectral_moment_k_onesided(k=2,i=2,j=len(freqs)-1,freqs=freqs,G=psd)
		assert mu0 != 0
		return np.sqrt(mu2/mu0)

	def FREQD_RD(self, copobj):
		rd = copobj.RD_COP
		freqs, psd = signal.welch(rd,return_onesided=True)
		mu0 = np.sum(psd[2:])
		mu1 = self.spectral_moment_k_onesided(k=1,i=2,j=len(freqs)-1,freqs=freqs,G=psd)
		mu2 = self.spectral_moment_k_onesided(k=2,i=2,j=len(freqs)-1,freqs=freqs,G=psd)
		return np.sqrt(1-mu1**2/(mu0*mu2))
	def FREQD_AP(self, copobj):
		ap = copobj.refAP
		freqs, psd = signal.welch(ap,return_onesided=True)
		mu0 = np.sum(psd[2:])
		mu1 = self.spectral_moment_k_onesided(k=1,i=2,j=len(freqs)-1,freqs=freqs,G=psd)
		mu2 = self.spectral_moment_k_onesided(k=2,i=2,j=len(freqs)-1,freqs=freqs,G=psd)
		return np.sqrt(1-mu1**2/(mu0*mu2))
	def FREQD_ML(self, copobj):
		ml = copobj.refML
		freqs, psd = signal.welch(ml,return_onesided=True)
		mu0 = np.sum(psd[2:])
		mu1 = self.spectral_moment_k_onesided(k=1,i=2,j=len(freqs)-1,freqs=freqs,G=psd)
		mu2 = self.spectral_moment_k_onesided(k=2,i=2,j=len(freqs)-1,freqs=freqs,G=psd)
		return np.sqrt(1-mu1**2/(mu0*mu2))





class COPobjPrimitiveFeat():
	def __init__(self) -> None:
		pass

	def find_AP(self, pressure_mat):
		num, den = 0.0, 0.0
		for i in range(pressure_mat.shape[0]):	# y
			for j in range(pressure_mat.shape[1]):	# x
				num += i*pressure_mat[i,j]
				den += pressure_mat[i,j]
		return num/den

	def find_ML(self, pressure_mat):
		num, den = 0.0, 0.0
		for i in range(pressure_mat.shape[0]):	# y
			for j in range(pressure_mat.shape[1]):	# x
				num += j*pressure_mat[i,j]
				den += pressure_mat[i,j]
		return num/den

	def find_COA(self, pressure_mat):
		""" COA is same as COM """
		pressure_mat = (pressure_mat > 0).astype(np.int32)	# binarize
		return ndimage.center_of_mass(pressure_mat)	# return COM

	def find_COP_measures(self, copobj, como):
		# Distance measures
		copobj.MDIST = np.mean(copobj.RD_COP)							# mean dist RD
		copobj.MDIST_AP = np.mean(np.absolute(copobj.refAP))			# mean dist AP
		copobj.MDIST_ML = np.mean(np.absolute(copobj.refML))			# mean dist ML

		copobj.RDIST = np.sqrt(np.mean(copobj.RD_COP**2))				# rms dist RD
		copobj.RDIST_AP = np.sqrt(np.mean(copobj.refAP**2))				# rms dist AP
		copobj.RDIST_ML = np.sqrt(np.mean(copobj.refML**2))				# rms dist ML

		copobj.TOTEX = como.TOTEX(copobj)								# total excursion COP
		copobj.TOTEX_AP = como.TOTEX_AP(copobj)							# total excursion AP
		copobj.TOTEX_ML = como.TOTEX_ML(copobj)							# total excursion ML

		copobj.MVELO = como.MVELO(copobj, 1)								# mean velocity COP, T=1s
		copobj.MVELO_AP = como.MVELO_AP(copobj, 1)							# mean velocity AP, T=1s
		copobj.MVELO_ML = como.MVELO_ML(copobj, 1)							# mean velocity ML, T=1s

		copobj.range = como.range(copobj)								# range COP
		copobj.range_AP = como.range_AP(copobj)							# range AP
		copobj.range_ML = como.range_ML(copobj)							# range ML

		copobj.circle_area = como.circle_area(copobj)					# 95% confidence circle area
		# copobj.elipse_area = como.elipse_area(copobj)					# 95% confidence elipse area

		copobj.AREA_SW = como.AREA_SW(copobj, 1)				# sway area

		copobj.MFREQ = como.MFREQ(copobj)					# mean frequency
		copobj.MFREQ_AP = como.MFREQ_AP(copobj)
		copobj.MFREQ_ML = como.MFREQ_ML(copobj)

		copobj.FD_PD = como.FD_PD(copobj)					# fractal dimension
		copobj.FD_CC = como.FD_CC(copobj)
		# copobj.FD_CE = como.FD_CE(copobj)

		copobj.total_power_RD = como.total_power_RD(copobj)		# total power RD
		copobj.total_power_AP = como.total_power_AP(copobj)		# total power AP
		copobj.total_power_ML = como.total_power_ML(copobj)		# total power ML

		copobj.power_freq50_RD = como.power_freq50_RD(copobj)	# 50% power frequency RD
		copobj.power_freq50_AP = como.power_freq50_AP(copobj)	# 50% power frequency AP
		copobj.power_freq50_ML = como.power_freq50_ML(copobj)	# 50% power frequency ML

		copobj.power_freq95_RD = como.power_freq95_RD(copobj)	# 95% power frequency RD
		copobj.power_freq95_AP = como.power_freq95_AP(copobj)	# 95% power frequency AP
		copobj.power_freq95_ML = como.power_freq95_ML(copobj)	# 95% power frequency ML

		copobj.CFREQ_RD = como.CFREQ_RD(copobj)					# centroidal frequency RD
		copobj.CFREQ_AP = como.CFREQ_AP(copobj)					# centroidal frequency AP
		copobj.CFREQ_ML = como.CFREQ_ML(copobj)					# centroidal frequency ML

		copobj.FREQD_RD = como.FREQD_RD(copobj)					# frequency dispersion RD
		copobj.FREQD_AP = como.FREQD_AP(copobj)					# frequency dispersion AP
		copobj.FREQD_ML = como.FREQD_ML(copobj)					# frequency dispersion ML

		return copobj

	def get_cop_obj(self, pressure_mat_seq):
		"""
			Find all COP/COA measures/features of pressure_mat_seq \n
			pressure_mat_seq: y,x,t  (60,40,100)	(AP,ML)
		"""
		AP_list = []
		ML_list = []
		COP_list = []
		COA_list = []

		for k in range(pressure_mat_seq.shape[-1]):
			t_AP = self.find_AP(pressure_mat_seq[:,:,k])	# time axis
			t_ML = self.find_ML(pressure_mat_seq[:,:,k])	# time axis
			AP_list.append(t_AP)
			ML_list.append(t_ML)
			COP_list.append([t_AP,t_ML])
			COA_list.append( self.find_COA(pressure_mat_seq[:,:,k]) )	# time axis

		copobj = COPobj()
		como = COPObjMeasuresOps()

		# COP
		copobj.AP = np.array(AP_list)
		copobj.ML = np.array(ML_list)
		copobj.COP = np.array(COP_list)
		copobj.meanAP = np.mean(copobj.AP)
		copobj.meanML = np.mean(copobj.ML)
		copobj.meanCOP = np.array([copobj.meanAP, copobj.meanML])
		copobj.refAP = copobj.AP - copobj.meanAP
		copobj.refML = copobj.ML - copobj.meanML
		copobj.refCOP = copobj.COP - copobj.meanCOP
		copobj.RD_COP = np.linalg.norm(copobj.refCOP, axis=-1)
		# COA
		copobj.COA = np.array(COA_list)
		copobj.meanCOA = np.mean(copobj.COA, axis=0)
		copobj.refCOA = copobj.COA - copobj.meanCOA
		copobj.RD_COA = np.linalg.norm(copobj.refCOA, axis=-1)
		# Measures
		copobj = self.find_COP_measures(copobj, como)
		del como
		return copobj

	def get_cop_feat_vec(self, pressure_mat_seq):
		copobj = self.get_cop_obj(pressure_mat_seq)
		# feat_vec = np.array([copobj.MDIST, copobj.MDIST_AP, copobj.MDIST_ML, copobj.RDIST, copobj.RDIST_AP, copobj.RDIST_ML, copobj.TOTEX, copobj.TOTEX_AP, copobj.TOTEX_ML, copobj.MVELO, copobj.MVELO_AP, copobj.MVELO_ML, copobj.range, copobj.range_AP, copobj.range_ML, copobj.circle_area, copobj.elipse_area, copobj.AREA_SW, copobj.MFREQ, copobj.MFREQ_AP, copobj.MFREQ_ML, copobj.FD_PD, copobj.FD_CC, copobj.FD_CE, copobj.total_power_RD, copobj.total_power_AP, copobj.total_power_ML])
		# feat_vec = np.array([copobj.MDIST, copobj.MDIST_AP, copobj.MDIST_ML, copobj.RDIST, copobj.RDIST_AP, copobj.RDIST_ML, copobj.TOTEX, copobj.TOTEX_AP, copobj.TOTEX_ML, copobj.MVELO, copobj.MVELO_AP, copobj.MVELO_ML, copobj.range, copobj.range_AP, copobj.range_ML, copobj.circle_area, copobj.AREA_SW, copobj.MFREQ, copobj.MFREQ_AP, copobj.MFREQ_ML, copobj.FD_PD, copobj.FD_CC, copobj.total_power_RD, copobj.total_power_AP, copobj.total_power_ML])
		feat_vec = np.array([copobj.MDIST, copobj.MDIST_AP, copobj.MDIST_ML, copobj.RDIST, copobj.RDIST_AP, copobj.RDIST_ML, copobj.TOTEX, copobj.TOTEX_AP, copobj.TOTEX_ML, copobj.MVELO, copobj.MVELO_AP, copobj.MVELO_ML, copobj.range, copobj.range_AP, copobj.range_ML, copobj.circle_area, copobj.AREA_SW, copobj.MFREQ, copobj.MFREQ_AP, copobj.MFREQ_ML, copobj.FD_PD, copobj.FD_CC, copobj.total_power_RD, copobj.total_power_AP, copobj.total_power_ML, copobj.power_freq50_RD, copobj.power_freq50_AP, copobj.power_freq50_ML, copobj.power_freq95_RD, copobj.power_freq95_AP, copobj.power_freq95_ML, copobj.CFREQ_RD, copobj.CFREQ_AP, copobj.CFREQ_ML, copobj.FREQD_RD, copobj.FREQD_AP, copobj.FREQD_ML])
		assert not np.isnan( np.sum(feat_vec) )
		del copobj
		return feat_vec






def create_features(metadata_barefoot_path, metadata_aligned_foot_path, data_aligned_foot_path):
	""" Find COP features for all subjects each sample (entire dataset) """
	# create preprocessed raw data
	dsobj = ds.DatasetRaw(metadata_barefoot_path, metadata_aligned_foot_path, data_aligned_foot_path)
	dsobj.preprocess()
	dsobj.create_per_person_per_foot_data_structure()
	# find features and delete raw data to create space
	dsfeatobj = ds.DatasetFeat(dsobj)
	feat_list = []
	for i in ns.natsorted(dsobj.person_dict.keys()):
		for j in (0,1):
			obj = COPobjPrimitiveFeat()
			tmp_lst = []
			for k in range(len(dsobj.person_dict[i][j])):	# for each foot
				fv = obj.get_cop_feat_vec(dsobj.person_dict[i][j][k])
				tmp_lst.append(fv)
				feat_list.append(fv)
			dsfeatobj.person_dict[i][j] = np.array(tmp_lst)
	del obj, dsobj
	return dsfeatobj, np.array(feat_list)

def load_feat_list(path, metadata_barefoot_path, metadata_aligned_foot_path, data_aligned_foot_path, feat_len=37):
	""" Read COP features from a file and populate the dataset structure """
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

def load_feat_list_MDIST(path, metadata_barefoot_path, metadata_aligned_foot_path, data_aligned_foot_path):
	""" Load only MDIST features from file """
	# create preprocessed raw data
	dsobj = ds.DatasetRaw(metadata_barefoot_path, metadata_aligned_foot_path, data_aligned_foot_path)
	dsobj.preprocess()
	dsobj.create_per_person_per_foot_data_structure()
	# find features and delete raw data to create space
	dsfeatobj = ds.DatasetFeat(dsobj)
	feat_list = np.load(path)
	feat_list = feat_list[:,:3]		# for selecting only MDIST features
	iter = 0
	feat_list_exp = []		# expanded feature list
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




def create_or_load_feat_37():
	"""
		Demonstrate the functionality to either create+save the COP features on the disk OR
		Load them from the disk
	"""
	metadata_barefoot_path = './data/perFootDataBarefoot/PerFootMetaDataBarefoot.npy'
	metadata_aligned_foot_path = './data/alignedPerFootDataBarefoot/AlignedFootMetaDataBarefoot.npy'
	data_aligned_foot_path = './data/alignedPerFootDataBarefoot/AlignedFootDataBarefoot.npz'
	feat_list_path = './features_mat/feat_list_37.npy'
	bool_load_feat_list = True

	if bool_load_feat_list:
		dsfeatobj, feat_list_exp = load_feat_list(feat_list_path, metadata_barefoot_path, metadata_aligned_foot_path, data_aligned_foot_path, feat_len=37)		# load feat_list
	else:
		dsfeatobj, feat_list = create_features(metadata_barefoot_path, metadata_aligned_foot_path, data_aligned_foot_path)
		np.save(feat_list_path, feat_list)		# save feat_list

def create_or_load_feat_MDIST():
	""" Load only MDIST features from the disk """
	metadata_barefoot_path = './data/perFootDataBarefoot/PerFootMetaDataBarefoot.npy'
	metadata_aligned_foot_path = './data/alignedPerFootDataBarefoot/AlignedFootMetaDataBarefoot.npy'
	data_aligned_foot_path = './data/alignedPerFootDataBarefoot/AlignedFootDataBarefoot.npz'
	feat_list_path = './features_mat/feat_list_37.npy'

	dsfeatobj, feat_list_exp = load_feat_list_MDIST(feat_list_path, metadata_barefoot_path, metadata_aligned_foot_path, data_aligned_foot_path)		# load feat_list only MDIST features





if __name__ == "__main__":
	# create_or_load_feat_37()
	create_or_load_feat_MDIST()






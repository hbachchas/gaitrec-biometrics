import numpy as np
import natsort as ns


class DatasetRaw():
	def __init__(self, metadata_barefoot_path, metadata_aligned_foot_path, data_aligned_foot_path) -> None:
		# Read metadata npy file
		self.md_bfoot = np.load(metadata_barefoot_path)				# barefoot: b
		self.md_abfoot = np.load(metadata_aligned_foot_path)		# aligned barefoot: ab
		# Read data files npz
		self.d_abfoot = np.load(data_aligned_foot_path)

	def preprocess(self):
		""" Read the origianl data and preprocess it """
		list_of_valid_foot_index = self.md_bfoot[:,3] == 0
		# Removing incomplete/invalid steps
		self.md_bfoot = self.md_bfoot[list_of_valid_foot_index]
		self.md_abfoot = self.md_abfoot[list_of_valid_foot_index]
		# Removing invalid from data
		lst = np.array(self.d_abfoot.files)		# read npz data file
		lst = lst[list_of_valid_foot_index]
		d_arr = []
		for item in lst:
			d_arr.append(self.d_abfoot[item])
		self.d_abfoot = np.array(d_arr)

	def create_per_person_per_foot_data_structure(self):
		""" Organize the original data with the help of dictionaries """
		self.person_dict = {}
		# person >> foot >> data
		for p in np.unique(self.md_bfoot[:,0]):			# for all unique people
			self.person_dict[p] = {0:[], 1:[]}			# left/right foot: empty structure
		for i in range(len(self.md_bfoot)):
			self.person_dict[self.md_bfoot[i,0]][self.md_bfoot[i,1]].append(self.d_abfoot[i])
		for i in ns.natsorted(self.person_dict.keys()):
			for j in (0,1):
				self.person_dict[i][j] = np.array(self.person_dict[i][j])




class DatasetFeat():
	def __init__(self, dsobj) -> None:
		""" Create a dummy structure of the dataset to access the saved features """
		self.person_dict = {}
		# person >> foot >> feature_vec
		for p in np.unique(dsobj.md_bfoot[:,0]):		# for all unique people
			self.person_dict[p] = {0:[], 1:[]}			# left/right foot: empty structure









if __name__ == "__main__":
	metadata_barefoot_path = './data/perFootDataBarefoot/PerFootMetaDataBarefoot.npy'
	metadata_aligned_foot_path = './data/alignedPerFootDataBarefoot/AlignedFootMetaDataBarefoot.npy'
	data_aligned_foot_path = './data/alignedPerFootDataBarefoot/AlignedFootDataBarefoot.npz'

	dsobj = DatasetRaw(metadata_barefoot_path, metadata_aligned_foot_path, data_aligned_foot_path)
	dsobj.preprocess()
	dsobj.create_per_person_per_foot_data_structure()



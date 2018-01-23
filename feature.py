#!/usr/bin/python
# coding: utf-8


###################################
#####       Author            #####
###################################
#####    Veronique DEFONTE    #####
#####    Valentin GABEUR      #####
#####    Edouard VILLAIN      #####
#####    Thomas ROLLAND       #####
#####    Arnaud JOUANNEAU     #####
###################################


##################################
#####       Licenses        ######
##################################
##### Non commercial usage  ######
### Research and Teaching only ###
##################################

import numpy as np

class Feature(object):
	'''
		class Feature :
			attribute :
				- label : int
				- nRow : int
				- nCol : int
				- Ratio_band1_0_band2_1_scale_1 : float
				- Ratio_band1_0_band2_1_scale_2 : float
				- Ratio_band1_0_band2_1_scale_4 : float
				- Ratio_band1_0_band2_2_scale_2 : float
				- Ratio_band1_0_band2_2_scale_4 : float
				- Ratio_band1_0_band2_3_scale_1 : float
				- Ratio_band1_0_band2_3_scale_2 : float
				- Ratio_band1_0_band2_3_scale_4 : float
				- Ratio_band1_1_band2_2_scale_1 : float
				- Ratio_band1_1_band2_2_scale_2 : float
				- Ratio_band1_1_band2_2_scale_4 : float
				- Ratio_band1_1_band2_3_scale_1 : float
				- Ratio_band1_1_band2_3_scale_2 : float
				- Ratio_band1_1_band2_3_scale_4 : float
				- Ratio_band1_2_band2_3_scale_1 : float
				- Ratio_band1_2_band2_3_scale_2 : float
				- Ratio_band1_2_band2_3_scale_4 : float
				- patch : int list
	'''

	def __init__(self, label, nRow, nCol):
		self.label = label
		self.nRow = nRow
		self.nCol = nCol

	####################################
	########## Setter methods ##########
	####################################

	def set_ratio_band(self, Ratio_band1_0_band2_1_scale_1,
				Ratio_band1_0_band2_1_scale_2,
				Ratio_band1_0_band2_1_scale_4,
				Ratio_band1_0_band2_2_scale_2,
				Ratio_band1_0_band2_2_scale_4,
				Ratio_band1_0_band2_3_scale_1,
				Ratio_band1_0_band2_3_scale_2,
				Ratio_band1_0_band2_3_scale_4,
				Ratio_band1_1_band2_2_scale_1,
				Ratio_band1_1_band2_2_scale_2,
				Ratio_band1_1_band2_2_scale_4,
				Ratio_band1_1_band2_3_scale_1,
				Ratio_band1_1_band2_3_scale_2,
				Ratio_band1_1_band2_3_scale_4,
				Ratio_band1_2_band2_3_scale_1,
				Ratio_band1_2_band2_3_scale_2,
				Ratio_band1_2_band2_3_scale_4):
		self.Ratio_band1_0_band2_1_scale_1 = Ratio_band1_0_band2_1_scale_1
		self.Ratio_band1_0_band2_1_scale_2 = Ratio_band1_0_band2_1_scale_2
		self.Ratio_band1_0_band2_1_scale_4 = Ratio_band1_0_band2_1_scale_4
		self.Ratio_band1_0_band2_2_scale_2 = Ratio_band1_0_band2_2_scale_2
		self.Ratio_band1_0_band2_2_scale_4 = Ratio_band1_0_band2_2_scale_4
		self.Ratio_band1_0_band2_3_scale_1 = Ratio_band1_0_band2_3_scale_1
		self.Ratio_band1_0_band2_3_scale_2 = Ratio_band1_0_band2_3_scale_2
		self.Ratio_band1_0_band2_3_scale_4 = Ratio_band1_0_band2_3_scale_4
		self.Ratio_band1_1_band2_2_scale_1 = Ratio_band1_1_band2_2_scale_1
		self.Ratio_band1_1_band2_2_scale_2 = Ratio_band1_1_band2_2_scale_2
		self.Ratio_band1_1_band2_2_scale_4 = Ratio_band1_1_band2_2_scale_4
		self.Ratio_band1_1_band2_3_scale_1 = Ratio_band1_1_band2_3_scale_1
		self.Ratio_band1_1_band2_3_scale_2 = Ratio_band1_1_band2_3_scale_2
		self.Ratio_band1_1_band2_3_scale_4 = Ratio_band1_1_band2_3_scale_4
		self.Ratio_band1_2_band2_3_scale_1 = Ratio_band1_2_band2_3_scale_1
		self.Ratio_band1_2_band2_3_scale_2 = Ratio_band1_2_band2_3_scale_2
		self.Ratio_band1_2_band2_3_scale_4 = Ratio_band1_2_band2_3_scale_4

	def set_patch(self, patch):
		self.patch = patch



	####################################
	########## Getter methods ##########
	####################################

	def get_patch_in_array(self):
		return np.asarray(self.patch).reshape((self.nRow,self.nCol,4))


	def get_feature_in_array(self):
		ret = []
		ret.append(self.Ratio_band1_0_band2_1_scale_1)
		ret.append(self.Ratio_band1_0_band2_1_scale_2)
		ret.append(self.Ratio_band1_0_band2_1_scale_4)
		ret.append(self.Ratio_band1_0_band2_2_scale_2)
		ret.append(self.Ratio_band1_0_band2_2_scale_4)
		ret.append(self.Ratio_band1_0_band2_3_scale_1)
		ret.append(self.Ratio_band1_0_band2_3_scale_2)
		ret.append(self.Ratio_band1_0_band2_3_scale_4)
		ret.append(self.Ratio_band1_1_band2_2_scale_1)
		ret.append(self.Ratio_band1_1_band2_2_scale_2)
		ret.append(self.Ratio_band1_1_band2_2_scale_4)
		ret.append(self.Ratio_band1_1_band2_3_scale_1)
		ret.append(self.Ratio_band1_1_band2_3_scale_2)
		ret.append(self.Ratio_band1_1_band2_3_scale_4)
		ret.append(self.Ratio_band1_2_band2_3_scale_1)
		ret.append(self.Ratio_band1_2_band2_3_scale_2)
		ret.append(self.Ratio_band1_2_band2_3_scale_4)
		for elem in self.patch:
			ret.append(elem)

		return np.array(ret)


	###################################
	############# Methods #############
	###################################






class Image_feature(object):
	'''
		class Image_feature :
			attribute :
				- nPatchRow : int
				- nPatchCol : int
				- feature_list : Feature list
	'''

	def __init__(self, nPatchRow, nPatchCol, feature_list):
		self.nPatchRow = nPatchRow
		self.nPatchCol = nPatchCol
		self.feature_list = feature_list




	####################################
	########## Getter methods ##########
	####################################

	def get_image_in_matrix(self):
		ret = []
		for row in range(self.nPatchRow):
			line = []
			for col in range(self.nPatchCol):
				line.append(self.feature_list[row * self.nPatchRow + col].get_feature_in_array())
			ret.append(line)
		return np.array(ret)

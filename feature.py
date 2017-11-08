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


	###################################
	############# Methods #############
	###################################






class Feature_Set(object):
	'''
		class Feature_Set :
			attribute : 
				- set : Feature list
	'''

	def __init__(self, feature_list):
		self.feature_list = feature_list
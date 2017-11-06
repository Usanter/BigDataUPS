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


try:
	import json 
	import ijson
	import numpy as np
	import matplotlib.pyplot as plt
	import scipy.misc
	from PIL import Image
except:
	print("Import Error : Please install required module")



def extract_patch_from_json(json_filename, path_save):
	'''
		extract_patch_from_json(json_filename, path_save)
			extract and save patch from the json file 
					- json_filename : string : path to the json file 
					- path_save : string : path to the save's folder
	'''
	# read json file
	json_file = open(json_filename)
	# parse json file 
	parser = ijson.parse(json_file)


	feature = {}
	num_feature = 0
	for prefix, event, value in parser:
		# begin of a new features 
#		if prefix == "features":
#			# save the features 
#			if not feature == {}:
#				with open(path_save + "feature_" + str(num_feature) + ".json", 'w') as outfile:  
#					json.dump(feature, outfile)
    		# begin a new feature 
#			feature = {}
#			patch = []
#			num_feature += 1

		if prefix == "features.item.Label":
			feature['Label'] = value
		if prefix == "features.item.nRow":
			feature['nRow'] = value
		if prefix == "features.item.nCol":
			feature['nCol'] = value
		if prefix == "features.item.Ratio_band1_0_band2_1_scale_1":
			feature['Ratio_band1_0_band2_1_scale_1'] = str(value)
		if prefix == "features.item.Ratio_band1_0_band2_1_scale_2":
			feature['Ratio_band1_0_band2_1_scale_2'] = str(value)
		if prefix == "features.item.Ratio_band1_0_band2_1_scale_4":
			feature['Ratio_band1_0_band2_1_scale_4'] = str(value)
		if prefix == "features.item.Ratio_band1_0_band2_2_scale_1":
			feature['Ratio_band1_0_band2_2_scale_1'] = str(value)
		if prefix == "features.item.Ratio_band1_0_band2_2_scale_2":
			feature['Ratio_band1_0_band2_2_scale_2'] = str(value)
		if prefix == "features.item.Ratio_band1_0_band2_2_scale_4":
			feature['Ratio_band1_0_band2_2_scale_4'] = str(value)
		if prefix == "features.item.Ratio_band1_0_band2_3_scale_1":
			feature['Ratio_band1_0_band2_3_scale_1'] = str(value)
		if prefix == "features.item.Ratio_band1_0_band2_3_scale_2":
			feature['Ratio_band1_0_band2_3_scale_2'] = str(value)
		if prefix == "features.item.Ratio_band1_0_band2_3_scale_4":
			feature['Ratio_band1_0_band2_3_scale_4'] = str(value)
		if prefix == "features.item.Ratio_band1_1_band2_2_scale_1":
			feature['Ratio_band1_1_band2_2_scale_1'] = str(value)
		if prefix == "features.item.Ratio_band1_1_band2_2_scale_2":
			feature['Ratio_band1_1_band2_2_scale_2'] = str(value)
		if prefix == "features.item.Ratio_band1_1_band2_2_scale_4":
			feature['Ratio_band1_1_band2_2_scale_4'] = str(value)
		if prefix == "features.item.Ratio_band1_1_band2_3_scale_1":
			feature['Ratio_band1_1_band2_3_scale_1'] = str(value)
		if prefix == "features.item.Ratio_band1_1_band2_3_scale_2":
			feature['Ratio_band1_1_band2_3_scale_2'] = str(value)
		if prefix == "features.item.Ratio_band1_1_band2_3_scale_4":
			feature['Ratio_band1_1_band2_3_scale_4'] = str(value)
		if prefix == "features.item.Ratio_band1_2_band2_3_scale_1":
			feature['Ratio_band1_2_band2_3_scale_1'] = str(value)
		if prefix == "features.item.Ratio_band1_2_band2_3_scale_2":
			feature['Ratio_band1_2_band2_3_scale_2'] = str(value)
		if prefix == "features.item.Ratio_band1_2_band2_3_scale_4":
			feature['Ratio_band1_2_band2_3_scale_4'] = str(value)

		if prefix == "features.item.Patch" and event == "start_array":
			patch = []
		if prefix == "features.item.Patch.item":
			patch.append(value) 
		if prefix == "features.item.Patch" and event == "end_array":
			feature['Patch'] = patch
			with open(path_save + "feature_" + str(num_feature) + ".json", 'w') as outfile:  
					json.dump(feature, outfile)
    		# begin a new feature 
			feature = {}
			num_feature += 1




if __name__ == '__main__':        
	json_filename = '../../json/part_features_set.json'
	# adapt you json path to file 
	path_save = '../../json/features/'
	extract_patch_from_json(json_filename, path_save)
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
	import re 
	import os 
	import numpy as np
	import matplotlib.pyplot as plt
	import scipy.misc
	from PIL import Image
except:
	print("Import Error : Please install required module")

#############################################################


def example_regex_function(features_folder, label):
	#expr = r"feature_*_label_" + label + r".json"
	expr = re.compile('feature_[0-9]*_label_%s.json'%label)
	l = os.listdir(features_folder)
	nb_feature_label = 0
	nb_file = 0
	for filename in l:
		nb_file += 1
		if expr.match(filename):
			nb_feature_label +=1 

	print ("Il y a " + str(nb_feature_label) + " feature labelisé " + str(label))
	print ("sur " + str(nb_file) + " feature")



def feature_full(feature):
	'''
		feature_full(feature) 
				return True if the feature is full
				else return False 
	'''
	try:
		feature['Label']
	except:
		return False
	try:
		feature['nRow']
	except:
		return False
	try:
		feature['nCol']
	except:
		return False
	try:
		feature['Ratio_band1_0_band2_1_scale_1']
	except:
		return False
	try:
		feature['Ratio_band1_0_band2_1_scale_2']
	except:
		return False
	try:
		feature['Ratio_band1_0_band2_1_scale_4']
	except:
		return False
	try:
		feature['Ratio_band1_0_band2_2_scale_2']
	except:
		return False
	try:
		feature['Ratio_band1_0_band2_2_scale_4']
	except:
		return False
	try:
		feature['Ratio_band1_0_band2_3_scale_1']
	except:
		return False
	try:
		feature['Ratio_band1_0_band2_3_scale_2']
	except:
		return False
	try:
		feature['Ratio_band1_0_band2_3_scale_4']
	except:
		return False
	try:
		feature['Ratio_band1_1_band2_2_scale_1']
	except:
		return False
	try:
		feature['Ratio_band1_1_band2_2_scale_2']
	except:
		return False
	try:
		feature['Ratio_band1_1_band2_2_scale_4']
	except:
		return False
	try:
		feature['Ratio_band1_1_band2_3_scale_1']
	except:
		return False
	try:
		feature['Ratio_band1_1_band2_3_scale_2']
	except:
		return False
	try:
		feature['Ratio_band1_1_band2_3_scale_4']
	except:
		return False
	try:
		feature['Ratio_band1_2_band2_3_scale_1']
	except:
		return False
	try:
		feature['Ratio_band1_2_band2_3_scale_2']
	except:
		return False
	try:
		feature['Ratio_band1_2_band2_3_scale_4']
	except:
		return False
	try:
		feature['Patch'][feature['nRow']*feature['nCol']]
	except:
		return False

	return True 


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
		if feature_full(feature):
			with open(path_save + "feature_" + str(num_feature) + "_label_" + str(feature['Label']) + ".json", 'w') as outfile:  
					json.dump(feature, outfile)
			print("Saving in " + path_save + "feature_" + str(num_feature) + "_label_" + str(feature['Label']) + ".json")
			feature = {}
			num_feature += 1
		if prefix == "features.item.Label":
			feature['Label'] = value
		elif prefix == "features.item.nRow":
			feature['nRow'] = value
		elif prefix == "features.item.nCol":
			feature['nCol'] = value
		elif prefix == "features.item.Ratio_band1_0_band2_1_scale_1":
			feature['Ratio_band1_0_band2_1_scale_1'] = str(value)
		elif prefix == "features.item.Ratio_band1_0_band2_1_scale_2":
			feature['Ratio_band1_0_band2_1_scale_2'] = str(value)
		elif prefix == "features.item.Ratio_band1_0_band2_1_scale_4":
			feature['Ratio_band1_0_band2_1_scale_4'] = str(value)
		elif prefix == "features.item.Ratio_band1_0_band2_2_scale_1":
			feature['Ratio_band1_0_band2_2_scale_1'] = str(value)
		elif prefix == "features.item.Ratio_band1_0_band2_2_scale_2":
			feature['Ratio_band1_0_band2_2_scale_2'] = str(value)
		elif prefix == "features.item.Ratio_band1_0_band2_2_scale_4":
			feature['Ratio_band1_0_band2_2_scale_4'] = str(value)
		elif prefix == "features.item.Ratio_band1_0_band2_3_scale_1":
			feature['Ratio_band1_0_band2_3_scale_1'] = str(value)
		elif prefix == "features.item.Ratio_band1_0_band2_3_scale_2":
			feature['Ratio_band1_0_band2_3_scale_2'] = str(value)
		elif prefix == "features.item.Ratio_band1_0_band2_3_scale_4":
			feature['Ratio_band1_0_band2_3_scale_4'] = str(value)
		elif prefix == "features.item.Ratio_band1_1_band2_2_scale_1":
			feature['Ratio_band1_1_band2_2_scale_1'] = str(value)
		elif prefix == "features.item.Ratio_band1_1_band2_2_scale_2":
			feature['Ratio_band1_1_band2_2_scale_2'] = str(value)
		elif prefix == "features.item.Ratio_band1_1_band2_2_scale_4":
			feature['Ratio_band1_1_band2_2_scale_4'] = str(value)
		elif prefix == "features.item.Ratio_band1_1_band2_3_scale_1":
			feature['Ratio_band1_1_band2_3_scale_1'] = str(value)
		elif prefix == "features.item.Ratio_band1_1_band2_3_scale_2":
			feature['Ratio_band1_1_band2_3_scale_2'] = str(value)
		elif prefix == "features.item.Ratio_band1_1_band2_3_scale_4":
			feature['Ratio_band1_1_band2_3_scale_4'] = str(value)
		elif prefix == "features.item.Ratio_band1_2_band2_3_scale_1":
			feature['Ratio_band1_2_band2_3_scale_1'] = str(value)
		elif prefix == "features.item.Ratio_band1_2_band2_3_scale_2":
			feature['Ratio_band1_2_band2_3_scale_2'] = str(value)
		elif prefix == "features.item.Ratio_band1_2_band2_3_scale_4":
			feature['Ratio_band1_2_band2_3_scale_4'] = str(value)

		elif prefix == "features.item.Patch" and event == "start_array":
			patch = []
		elif prefix == "features.item.Patch.item":
			patch.append(value) 
		elif prefix == "features.item.Patch" and event == "end_array":
			feature['Patch'] = patch



if __name__ == '__main__':        
	json_filename = '../../json/features_set_0.json'
	# adapt you json path to file 
	path_save = '../../json/features_labelise/'
	#extract_patch_from_json(json_filename, path_save)
	example_regex_function(path_save, 14)
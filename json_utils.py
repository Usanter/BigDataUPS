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
	import csv 
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


def compute_mean_var_ratio(features_folder, csv_filename):
	'''
		compute_mean_std(feature_folder)
				compute the mean and the std of each label 
	'''
	nb_label = 17 

	Ratio_band1_0_band2_1_scale_1 = [[] for i in range(nb_label)]
	Ratio_band1_0_band2_1_scale_2 = [[] for i in range(nb_label)]
	Ratio_band1_0_band2_1_scale_4 = [[] for i in range(nb_label)]
	Ratio_band1_0_band2_2_scale_2 = [[] for i in range(nb_label)]
	Ratio_band1_0_band2_2_scale_4 = [[] for i in range(nb_label)]
	Ratio_band1_0_band2_3_scale_1 = [[] for i in range(nb_label)]
	Ratio_band1_0_band2_3_scale_2 = [[] for i in range(nb_label)]
	Ratio_band1_0_band2_3_scale_4 = [[] for i in range(nb_label)]
	Ratio_band1_1_band2_2_scale_1 = [[] for i in range(nb_label)]
	Ratio_band1_1_band2_2_scale_2 = [[] for i in range(nb_label)]
	Ratio_band1_1_band2_2_scale_4 = [[] for i in range(nb_label)]
	Ratio_band1_1_band2_3_scale_1 = [[] for i in range(nb_label)]
	Ratio_band1_1_band2_3_scale_2 = [[] for i in range(nb_label)]
	Ratio_band1_1_band2_3_scale_4 = [[] for i in range(nb_label)]
	Ratio_band1_2_band2_3_scale_1 = [[] for i in range(nb_label)]
	Ratio_band1_2_band2_3_scale_2 = [[] for i in range(nb_label)]
	Ratio_band1_2_band2_3_scale_4 = [[] for i in range(nb_label)]

	nb_feature_per_label = [0 for i in range(nb_label)]

	l = os.listdir(features_folder)
	for filename in l:
		print ("computing data in : " + features_folder + filename)
		with open(features_folder + filename, 'r') as f:
			feature = json.load(f)

		# update nb_feature_per_label 
		nb_feature_per_label[feature['Label']] += 1

		# update ratio band 
		Ratio_band1_0_band2_1_scale_1[feature['Label']].append(float(feature['Ratio_band1_0_band2_1_scale_1']))
		Ratio_band1_0_band2_1_scale_2[feature['Label']].append(float(feature['Ratio_band1_0_band2_1_scale_2']))
		Ratio_band1_0_band2_1_scale_4[feature['Label']].append(float(feature['Ratio_band1_0_band2_1_scale_4']))
		Ratio_band1_0_band2_2_scale_2[feature['Label']].append(float(feature['Ratio_band1_0_band2_2_scale_2']))
		Ratio_band1_0_band2_2_scale_4[feature['Label']].append(float(feature['Ratio_band1_0_band2_2_scale_4']))
		Ratio_band1_0_band2_3_scale_1[feature['Label']].append(float(feature['Ratio_band1_0_band2_3_scale_1']))
		Ratio_band1_0_band2_3_scale_2[feature['Label']].append(float(feature['Ratio_band1_0_band2_3_scale_2']))
		Ratio_band1_0_band2_3_scale_4[feature['Label']].append(float(feature['Ratio_band1_0_band2_3_scale_4']))
		Ratio_band1_1_band2_2_scale_1[feature['Label']].append(float(feature['Ratio_band1_1_band2_2_scale_1']))
		Ratio_band1_1_band2_2_scale_2[feature['Label']].append(float(feature['Ratio_band1_1_band2_2_scale_2']))
		Ratio_band1_1_band2_2_scale_4[feature['Label']].append(float(feature['Ratio_band1_1_band2_2_scale_4']))
		Ratio_band1_1_band2_3_scale_1[feature['Label']].append(float(feature['Ratio_band1_1_band2_3_scale_1']))
		Ratio_band1_1_band2_3_scale_2[feature['Label']].append(float(feature['Ratio_band1_1_band2_3_scale_2']))
		Ratio_band1_1_band2_3_scale_4[feature['Label']].append(float(feature['Ratio_band1_1_band2_3_scale_4']))
		Ratio_band1_2_band2_3_scale_1[feature['Label']].append(float(feature['Ratio_band1_2_band2_3_scale_1']))
		Ratio_band1_2_band2_3_scale_2[feature['Label']].append(float(feature['Ratio_band1_2_band2_3_scale_2']))
		Ratio_band1_2_band2_3_scale_4[feature['Label']].append(float(feature['Ratio_band1_2_band2_3_scale_4']))


	Ratio_band1_0_band2_1_scale_1_mean = [0.0 for i in range(nb_label)]
	Ratio_band1_0_band2_1_scale_2_mean = [0.0 for i in range(nb_label)]
	Ratio_band1_0_band2_1_scale_4_mean = [0.0 for i in range(nb_label)]
	Ratio_band1_0_band2_2_scale_2_mean = [0.0 for i in range(nb_label)]
	Ratio_band1_0_band2_2_scale_4_mean = [0.0 for i in range(nb_label)]
	Ratio_band1_0_band2_3_scale_1_mean = [0.0 for i in range(nb_label)]
	Ratio_band1_0_band2_3_scale_2_mean = [0.0 for i in range(nb_label)]
	Ratio_band1_0_band2_3_scale_4_mean = [0.0 for i in range(nb_label)]
	Ratio_band1_1_band2_2_scale_1_mean = [0.0 for i in range(nb_label)]
	Ratio_band1_1_band2_2_scale_2_mean = [0.0 for i in range(nb_label)]
	Ratio_band1_1_band2_2_scale_4_mean = [0.0 for i in range(nb_label)]
	Ratio_band1_1_band2_3_scale_1_mean = [0.0 for i in range(nb_label)]
	Ratio_band1_1_band2_3_scale_2_mean = [0.0 for i in range(nb_label)]
	Ratio_band1_1_band2_3_scale_4_mean = [0.0 for i in range(nb_label)]
	Ratio_band1_2_band2_3_scale_1_mean = [0.0 for i in range(nb_label)]
	Ratio_band1_2_band2_3_scale_2_mean = [0.0 for i in range(nb_label)]
	Ratio_band1_2_band2_3_scale_4_mean = [0.0 for i in range(nb_label)]

	Ratio_band1_0_band2_1_scale_1_var = [0.0 for i in range(nb_label)]
	Ratio_band1_0_band2_1_scale_2_var = [0.0 for i in range(nb_label)]
	Ratio_band1_0_band2_1_scale_4_var = [0.0 for i in range(nb_label)]
	Ratio_band1_0_band2_2_scale_2_var = [0.0 for i in range(nb_label)]
	Ratio_band1_0_band2_2_scale_4_var = [0.0 for i in range(nb_label)]
	Ratio_band1_0_band2_3_scale_1_var = [0.0 for i in range(nb_label)]
	Ratio_band1_0_band2_3_scale_2_var = [0.0 for i in range(nb_label)]
	Ratio_band1_0_band2_3_scale_4_var = [0.0 for i in range(nb_label)]
	Ratio_band1_1_band2_2_scale_1_var = [0.0 for i in range(nb_label)]
	Ratio_band1_1_band2_2_scale_2_var = [0.0 for i in range(nb_label)]
	Ratio_band1_1_band2_2_scale_4_var = [0.0 for i in range(nb_label)]
	Ratio_band1_1_band2_3_scale_1_var = [0.0 for i in range(nb_label)]
	Ratio_band1_1_band2_3_scale_2_var = [0.0 for i in range(nb_label)]
	Ratio_band1_1_band2_3_scale_4_var = [0.0 for i in range(nb_label)]
	Ratio_band1_2_band2_3_scale_1_var = [0.0 for i in range(nb_label)]
	Ratio_band1_2_band2_3_scale_2_var = [0.0 for i in range(nb_label)]
	Ratio_band1_2_band2_3_scale_4_var = [0.0 for i in range(nb_label)]

	# compute mean and var 
	for label in range(nb_label):
		Ratio_band1_0_band2_1_scale_1_mean[label] = np.mean(Ratio_band1_0_band2_1_scale_1[label])
		Ratio_band1_0_band2_1_scale_2_mean[label] = np.mean(Ratio_band1_0_band2_1_scale_2[label])
		Ratio_band1_0_band2_1_scale_4_mean[label] = np.mean(Ratio_band1_0_band2_1_scale_4[label])
		Ratio_band1_0_band2_2_scale_2_mean[label] = np.mean(Ratio_band1_0_band2_2_scale_2[label])
		Ratio_band1_0_band2_2_scale_4_mean[label] = np.mean(Ratio_band1_0_band2_2_scale_4[label])
		Ratio_band1_0_band2_3_scale_1_mean[label] = np.mean(Ratio_band1_0_band2_3_scale_1[label])
		Ratio_band1_0_band2_3_scale_2_mean[label] = np.mean(Ratio_band1_0_band2_3_scale_2[label])
		Ratio_band1_0_band2_3_scale_4_mean[label] = np.mean(Ratio_band1_0_band2_3_scale_4[label])
		Ratio_band1_1_band2_2_scale_1_mean[label] = np.mean(Ratio_band1_1_band2_2_scale_1[label])
		Ratio_band1_1_band2_2_scale_2_mean[label] = np.mean(Ratio_band1_1_band2_2_scale_2[label])
		Ratio_band1_1_band2_2_scale_4_mean[label] = np.mean(Ratio_band1_1_band2_2_scale_4[label])
		Ratio_band1_1_band2_3_scale_1_mean[label] = np.mean(Ratio_band1_1_band2_3_scale_1[label])
		Ratio_band1_1_band2_3_scale_2_mean[label] = np.mean(Ratio_band1_1_band2_3_scale_2[label])
		Ratio_band1_1_band2_3_scale_4_mean[label] = np.mean(Ratio_band1_1_band2_3_scale_4[label])
		Ratio_band1_2_band2_3_scale_1_mean[label] = np.mean(Ratio_band1_2_band2_3_scale_1[label])
		Ratio_band1_2_band2_3_scale_2_mean[label] = np.mean(Ratio_band1_2_band2_3_scale_2[label])
		Ratio_band1_2_band2_3_scale_4_mean[label] = np.mean(Ratio_band1_2_band2_3_scale_4[label])

		Ratio_band1_0_band2_1_scale_1_var[label] = np.var(Ratio_band1_0_band2_1_scale_1[label])
		Ratio_band1_0_band2_1_scale_2_var[label] = np.var(Ratio_band1_0_band2_1_scale_2[label])
		Ratio_band1_0_band2_1_scale_4_var[label] = np.var(Ratio_band1_0_band2_1_scale_4[label])
		Ratio_band1_0_band2_2_scale_2_var[label] = np.var(Ratio_band1_0_band2_2_scale_2[label])
		Ratio_band1_0_band2_2_scale_4_var[label] = np.var(Ratio_band1_0_band2_2_scale_4[label])
		Ratio_band1_0_band2_3_scale_1_var[label] = np.var(Ratio_band1_0_band2_3_scale_1[label])
		Ratio_band1_0_band2_3_scale_2_var[label] = np.var(Ratio_band1_0_band2_3_scale_2[label])
		Ratio_band1_0_band2_3_scale_4_var[label] = np.var(Ratio_band1_0_band2_3_scale_4[label])
		Ratio_band1_1_band2_2_scale_1_var[label] = np.var(Ratio_band1_1_band2_2_scale_1[label])
		Ratio_band1_1_band2_2_scale_2_var[label] = np.var(Ratio_band1_1_band2_2_scale_2[label])
		Ratio_band1_1_band2_2_scale_4_var[label] = np.var(Ratio_band1_1_band2_2_scale_4[label])
		Ratio_band1_1_band2_3_scale_1_var[label] = np.var(Ratio_band1_1_band2_3_scale_1[label])
		Ratio_band1_1_band2_3_scale_2_var[label] = np.var(Ratio_band1_1_band2_3_scale_2[label])
		Ratio_band1_1_band2_3_scale_4_var[label] = np.var(Ratio_band1_1_band2_3_scale_4[label])
		Ratio_band1_2_band2_3_scale_1_var[label] = np.var(Ratio_band1_2_band2_3_scale_1[label])
		Ratio_band1_2_band2_3_scale_2_var[label] = np.var(Ratio_band1_2_band2_3_scale_2[label])
		Ratio_band1_2_band2_3_scale_4_var[label] = np.var(Ratio_band1_2_band2_3_scale_4[label])

	# prepare in order to save in csv 
	data = [Ratio_band1_0_band2_1_scale_1_mean, \
	Ratio_band1_0_band2_1_scale_2_mean, \
	Ratio_band1_0_band2_1_scale_4_mean , \
	Ratio_band1_0_band2_2_scale_2_mean , \
	Ratio_band1_0_band2_2_scale_4_mean , \
	Ratio_band1_0_band2_3_scale_1_mean , \
	Ratio_band1_0_band2_3_scale_2_mean , \
	Ratio_band1_0_band2_3_scale_4_mean , \
	Ratio_band1_1_band2_2_scale_1_mean , \
	Ratio_band1_1_band2_2_scale_2_mean , \
	Ratio_band1_1_band2_2_scale_4_mean , \
	Ratio_band1_1_band2_3_scale_1_mean , \
	Ratio_band1_1_band2_3_scale_2_mean , \
	Ratio_band1_1_band2_3_scale_4_mean , \
	Ratio_band1_2_band2_3_scale_1_mean , \
	Ratio_band1_2_band2_3_scale_2_mean , \
	Ratio_band1_2_band2_3_scale_4_mean , \

	Ratio_band1_0_band2_1_scale_1_var, \
	Ratio_band1_0_band2_1_scale_2_var, \
	Ratio_band1_0_band2_1_scale_4_var, \
	Ratio_band1_0_band2_2_scale_2_var, \
	Ratio_band1_0_band2_2_scale_4_var, \
	Ratio_band1_0_band2_3_scale_1_var, \
	Ratio_band1_0_band2_3_scale_2_var, \
	Ratio_band1_0_band2_3_scale_4_var, \
	Ratio_band1_1_band2_2_scale_1_var, \
	Ratio_band1_1_band2_2_scale_2_var, \
	Ratio_band1_1_band2_2_scale_4_var, \
	Ratio_band1_1_band2_3_scale_1_var, \
	Ratio_band1_1_band2_3_scale_2_var, \
	Ratio_band1_1_band2_3_scale_4_var, \
	Ratio_band1_2_band2_3_scale_1_var, \
	Ratio_band1_2_band2_3_scale_2_var, \
	Ratio_band1_2_band2_3_scale_4_var]

	legend_row = ['label %d'%i for i in range(nb_label)]
	# save the csv 
	with open(csv_filename, "wb") as csv_file:
		writer = csv.writer(csv_file, delimiter=',')
		writer.writerow(legend_row)
		i = 0
		for line in data:
			writer.writerow(line)
			i += 1



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
	#json_filename = '../../json/part_features_set.json'
	#path_save = '../../json/feature_test/'
	json_filename = '../../json/features_set_0.json'
	path_save = '../../json/features_labelise/'

	#example_regex_function(path_save, 14)
	#extract_patch_from_json(json_filename, path_save)

	csv_filename = '../../json/mean_var/mean_var_ratio.csv'
	compute_mean_var_ratio(path_save, csv_filename)
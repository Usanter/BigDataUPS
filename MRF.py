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


# Image segmentation using MRF model
from PIL import Image
import numpy
from pylab import *
from scipy.cluster.vq import *
from scipy.signal import *
import scipy

from json_utils import *



def launch_MRF(nb_iter, nb_label, save_iter):
	'''
		launch_MRF(nb_iter, nb_label)
			launch au MRF segmentation optimisation 
				nb_iter : integer 
				nb_label : integer 
				save_iter : bool 
	'''

	# prepare img 
	image = create_image_test('../../json/features_labelise/', 37 * 37)
	# img : 37 * 37 patch : 2321 params 
	img = image.get_image_in_matrix()

	# prepare label_img
	label_img = getinitkmean(img, nb_label)
	if save_iter:
		scipy.misc.imsave('kmean_image.png',label_img * (256 / 17))

	# iterate 
	for it in range(nb_iter):
		print ("iteration " + str(it))
		# MRF ICM
		win_dim=32
		while (win_dim>4):
			print win_dim
			# compute local average  
			local_av = local_average(img, label_img, nb_label, win_dim) 
			# label the image 
			label_img = MRF(img, label_img, local_av, nb_label)
			win_dim = win_dim/2
		# save if necessary 
		if save_iter:
			scipy.misc.imsave('label_img_%s.png'%it,label_img * (256 / 17))

	print label_img







def MRF(img, label_img, local_av, nb_label):
	'''
		MRF(img, label_img, local_av, nb_label)
			MRF segmentation 
				img : ndarray 
				label_img : ndarray 
				local_av : ndarray
				nb_label : integer 
	'''
	(M,N)=img.shape[0:2]
	for i in range(M):
		for j in range(N):
			# Find segmentation level which has min energy (highest posterior)
			cost = [energy(k,i,j, img, label_img, local_av) for k in range(nb_label)]
			label_img[i,j] = cost.index(min(cost))
	return label_img





def energy(patch_label, i, j, img, label_img, local_av):
	'''
		energy(patch_label, i, j, img, label_img, local_av)
			compute energy function  
				patch_label : integer 
				i : integer 
				j : integer 
				img : ndarray
				label_img : matrix 
				local_av : ndarray
	'''
	beta = 0.5
	std = 7
	cl = clique(patch_label, i, j, label_img)
	closeness = numpy.linalg.norm(local_av[i,j,:,patch_label] - img[i,j,:])
	return beta*cl+closeness/std**2




def delta(label_1, label_2):
	'''
		delta(label_1, label_2)
			compute delta function 
				label_1 : integer 
				label_2 : integer
	'''
	if label_1 == label_2:
		return -1
	else:
		return 1


def local_average(img, label_img, nb_label, win_dim):
	'''
		local_average(img, label_img, nb_label, win_dim)
			compute local average 
				img : ndarray
				label_img : ndarray 
				nb_label : integer 
				wind_dim : integer 
	'''
	# Use correlation to perform averaging
	mask = numpy.ones((win_dim, win_dim))/win_dim**2

	# 4d array (512, 512, ncolors, nb_label)
	local_av = ones((img.shape+(nb_label,)))
	for i in range(img.shape[2]):	# loop through image channels
		for j in range(nb_label):	# loop through segmentation levels
			temp = (img[:,:,i] * (label_img == j))
			local_av[:,:,i,j] = fftconvolve(temp, mask, mode='same')
	return local_av


def clique(patch_label, i, j, label_img):
	'''
		clique(patch_label, i, j, label_img)
			compute potential of the clique  
				patch_label : integer
				i : integer 
				j : integer 
				label_img : matrix  
	'''
	M, N = label_img.shape[0:2]
	#find correct neighbors
	if i == 0 and j == 0:
		neighbor = [(0,1), (1,0)]
	elif i == 0 and j == N-1:
		neighbor=[(0,N-2), (1,N-1)]
	elif i == M-1 and j == 0:
		neighbor = [(M-1,1), (M-2,0)]
	elif i == M-1 and j == N-1:
		neighbor = [(M-1,N-2), (M-2,N-1)]
	elif i == 0:
		neighbor = [(0,j-1), (0,j+1), (1,j)]
	elif i == M-1:
		neighbor = [(M-1,j-1), (M-1,j+1), (M-2,j)]
	elif j == 0:
		neighbor = [(i-1,0), (i+1,0), (i,1)]
	elif j == N-1:
		neighbor = [(i-1,N-1), (i+1,N-1), (i,N-2)]
	else:
		neighbor = [(i-1,j), (i+1,j), (i,j-1), (i,j+1),\
				  (i-1,j-1), (i-1,j+1), (i+1,j-1), (i+1,j+1)]
	
	return sum(delta(patch_label, label_img[i]) for i in neighbor)



def getinitkmean(img, nb_label):
	obs = reshape(img,(img.shape[0]*img.shape[1],-1))	
	obs = whiten(obs)

	(centroids, label_img) = kmeans2(obs,nb_label)
	label_img = label_img.reshape(img.shape[0],img.shape[1])
	return label_img



launch_MRF(2, 17, True)
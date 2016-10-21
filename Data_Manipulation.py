import matplotlib.pyplot as plt
import numpy as np
import pydicom
import matplotlib
import matplotlib.pyplot as plt
import io
import os
from skimage.data import camera
from skimage.filters import threshold_otsu
from skimage import data
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import io
import numpy
""" change """
import pydicom as dicom
import os
import sys
import shutil
import sys
import pylab
from matplotlib import pyplot, cm
import skimage
import pydicom as dicom
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk
from skimage.filters.rank import median
from skimage.morphology import disk
from skimage.transform import (hough_line, hough_line_peaks,
							   probabilistic_hough_line)
from skimage import feature
from skimage import data
import math
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.color import label2rgb
import numpy as np
import matplotlib.pyplot as plt

from skimage import data, color
from skimage.transform import hough_circle
from skimage.feature import peak_local_max
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte

from skimage.draw import line, polygon, circle, circle_perimeter, ellipse


"""Here are the imports from wndcharm"""
from wndcharm.PyImageMatrix import PyImageMatrix
from wndcharm.FeatureVector import FeatureVector
import time



def filler(x,y,image):
	xlim=image.shape[0]
	ylim=image.shape[1]
	if image[x,y]==0:
		image[x,y]=100000
		if(x<xlim-1):
			image=filler(x+1,y,image)
		if(y<ylim-1):
			image=filler(x,y+1,image)
		return image
	else:
		return image

def flipex(right):
	h=extract_outline(right)
	right=np.fliplr(right)
	m=extract_outline(right)
	m=np.fliplr(m)
	return(h,m)

def extract_outline(right):
	(a,b,c)=resize(right)
	c=filler(0,0,c)
	c=filler(c.shape[0]-1,c.shape[1]-1,c)
	c=feature.canny(c)
	h=size(a,b,c)
	return h

def m_data(image):
	arr = []
	k = 0
	x = image.shape[0]
	y = image.shape[1]
	for i in range(x):
		for j in range(y):
			b = []
			if(image[i][j] != False and image[i][j+5] == False):
				b.append(i)
				b.append(j)
				arr.append(b)
				k = k + 1
				break
	return arr

def procrusutes(left,right):
	left_m = m_data(left)
	right_m = m_data(right)
	left_m = np.asarray(left_m)
	right_m = np.asarray(right_m)
	(lvar1,lvar2) = np.std(left_m,axis = 0)
	(rvar1,rvar2) = np.std(right_m,axis = 0)
	(lm1,lm2) = np.mean(left_m,axis= 0)
	(rm1,rm2) = np.mean(right_m,axis = 0)
	"""here I am cutting the longer border of breast, but we have to think whether 
	we shout cut or pad zeros to the border to make them equal length"""
	if(left_m.shape[0] > right_m.shape[0]):
		left_m = left_m[:right_m.shape[0],:]
	else:
		right_m = right_m[:left_m.shape[0],:]
	print(left_m.shape[0])
	print(right_m.shape[0])
	(left_m,right_m,c) = scipy.spatial.procrustes(left_m,right_m)
	plt.scatter(left_m[:,0],left_m[:,1], alpha=.1, s=2)
	plt.show()
	return (left_m,right_m,c,[lm1,lvar1,lm2,lvar2],[rm1,rvar1,rm2,rvar2])

def reconstruct(left,right):
	left = loadfile(left)
	right = loadfile(right)
	""" here we have to pass right images"""
	(left,right,c,d,e) = procrusutes(left,right)
	left[:,0] = (left[:,0] * d[1]) + d[0]
	left[:,1] = (left[:,1] * d[3]) + d[2]
	right[:,0] = (right[:,0] * e[1]) + e[0]
	right[:,1] = (right[:,1] * e[3]) + e[2]
	left = np.astype(left,np.int16)
	right = np.astype(right,np.int16)

	return left,right






def find_mole_markers(right):
	centers = []
	accums = []
	radii = []
	(a,b,c)=resize(right)
	image=np.zeros(c.shape)
	hough_radii = np.arange(5, 20, 1)
	edges = feature.canny(c, sigma=1, low_threshold=1500, high_threshold=2000)
	hough_res = hough_circle(edges, hough_radii)
	for radius, h in zip(hough_radii, hough_res):
		# For each radius, extract two circles
		num_peaks = 2
		peaks = peak_local_max(h, num_peaks=num_peaks)
		centers.extend(peaks)
		accums.extend(h[peaks[:, 0], peaks[:, 1]])
		radii.extend([radius] * num_peaks)
	for idx in np.argsort(accums)[::-1][:5]:
		center_x, center_y = centers[idx]
		radius = radii[idx]
		if radius>8:
			cx, cy = circle(center_y, center_x, radius)
			image[cy, cx] = 1000
	h=size(a,b,image)
	thresh(h)
	h=h/np.max(h)
	o=np.ones(h.shape)
	h=o-h
	return h


def resize(image,scale=10):
	vert=image.shape[0]
	horiz=image.shape[1]
	x=0
	y=0
	newarr=[]
	while(x<vert):
		newarr.append(image[x,:])
		x=x+scale
	xrem=vert-1-x+scale
	newimage=np.vstack(newarr)
	newarr=[]
	while(y<horiz):
		newarr.append(newimage[:,y])
		y=y+scale
	yrem=horiz-1-y+scale
	final=np.column_stack(newarr)
	return(xrem,yrem,final)


def size(xrem,yrem,image,scale=10):
	vert=image.shape[0]
	horiz=image.shape[1]
	x=0
	y=0
	newarr=[]
	while(x<vert):
		lim=0
		if x==vert-1:
			while(lim<=xrem):
				newarr.append(image[x,:])
				lim=lim+1
			break
		while(lim<scale):
			newarr.append(image[x,:])
			lim=lim+1
		x=x+1
	newimage=np.vstack(newarr)
	newarr=[]
	while(y<horiz):
		lim=0
		if y==horiz-1:
			while(lim<=yrem):
				newarr.append(newimage[:,y])
				lim=lim+1
			break
		while(lim<scale):
			newarr.append(newimage[:,y])
			lim=lim+1
		y=y+1
	final=np.column_stack(newarr)
	return final



def convert_to_8(array1):
	array1=array1/np.max(array1)
	array1=array1*255
	array1=array1.astype('B')
	return array1


def pad(window,padding=200):
	x=window.shape[1]
	y=window.shape[0]
	window[0:padding,:]=np.zeros(window[0:padding,:].shape)
	window[:,0:padding]=np.zeros(window[:,0:padding].shape)
	window[y-padding:y,:]=np.zeros(window[y-padding:y,:].shape)
	window[:,x-padding:x]=np.zeros(window[:,x-padding:x].shape)
	return window


def showpatch(coods,target):
	image=target[coods[2]:coods[3],coods[0]:coods[1]]
	io.imshow(image)
	io.show()

def display(target):
	io.imshow(target)
	io.show()

def overlay_labels(image, lbp, labels):
	mask = np.logical_or.reduce([lbp == each for each in labels])
	return label2rgb(mask, image=image, bg_label=0, alpha=0.5)


def highlight_bars(bars, indexes):
	for i in indexes:
		bars[i].set_facecolor('r')


def hist(ax, lbp):
	n_bins = lbp.max() + 1
	return ax.hist(lbp.ravel(), normed=True, bins=n_bins, range=(0, n_bins),
				   facecolor='0.5')

def plot_lbp(image):
	METHOD = 'uniform'
	radius = 3
	n_points = 8 * radius
	lbp = local_binary_pattern(image, n_points, radius, METHOD)
	fig, (ax_img, ax_hist) = plt.subplots(nrows=2, ncols=3, figsize=(9, 6))
	plt.gray()

	titles = ('edge', 'flat', 'corner')
	w = width = radius - 1
	edge_labels = range(n_points // 2 - w, n_points // 2 + w + 1)
	flat_labels = list(range(0, w + 1)) + list(range(n_points - w, n_points + 2))
	i_14 = n_points // 4            # 1/4th of the histogram
	i_34 = 3 * (n_points // 4)      # 3/4th of the histogram
	corner_labels = (list(range(i_14 - w, i_14 + w + 1)) +list(range(i_34 - w, i_34 + w + 1)))
	label_sets = (edge_labels, flat_labels, corner_labels)
	for ax, labels in zip(ax_img, label_sets):
		ax.imshow(overlay_labels(image, lbp, labels))

	for ax, labels, name in zip(ax_hist, label_sets, titles):
		counts, _, bars = hist(ax, lbp)
		highlight_bars(bars, labels)
		ax.set_ylim(ymax=np.max(counts[:-1]))
		ax.set_xlim(xmax=n_points + 2)
		ax.set_title(name)

	ax_hist[0].set_ylabel('Percentage')
	for ax in ax_img:
		ax.axis('off')

def hough(image):
	edges = feature.canny(image, 2, 1, 25)
	lines = probabilistic_hough_line(edges, threshold=10, line_length=5,line_gap=3)
	fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8,4), sharex=True, sharey=True)
	ax1.imshow(image, cmap=plt.cm.gray)
	ax1.set_title('Input image')
	ax1.set_axis_off()
	ax1.set_adjustable('box-forced')
	ax2.imshow(edges, cmap=plt.cm.gray)
	ax2.set_title('Canny edges')
	ax2.set_axis_off()
	ax2.set_adjustable('box-forced')
	ax3.imshow(edges * 0)
	for line in lines:
		p0, p1 = line
		ax3.plot((p0[0], p1[0]), (p0[1], p1[1]))
	ax3.set_title('Probabilistic Hough')
	ax3.set_axis_off()
	ax3.set_adjustable('box-forced')
	plt.show()


def kullback_leibler_divergence(p,q):
	p=np.asarray(p)
	q=np.asarray(q)
	filt=np.logical_and(p!=0,q!=0)
	return np.sum(p[filt]*np.log2(p[filt]/q[filt]))

def plot_comparison(original, filtered, filter_name):
	fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True, sharey=True)
	ax1.imshow(original, cmap=plt.cm.gray)
	ax1.set_title('original')
	ax1.axis('off')
	ax1.set_adjustable('box-forced')
	ax2.imshow(filtered, cmap=plt.cm.gray)
	ax2.set_title(filter_name)
	ax2.axis('off')
	ax2.set_adjustable('box-forced')

def loadfile(file):
	RefDs = dicom.read_file(file)
	f = RefDs.pixel_array 
	return f

def entropymap(f,ds=100):
	f = img_as_ubyte(f)
	#fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 4), sharex=True,sharey=True,subplot_kw={"adjustable": "box-forced"})
	#img0 = ax0.imshow(f, cmap=plt.cm.gray)
	#ax0.set_title("Image")
	#ax0.axis("off")
	#fig.colorbar(img0, ax=ax0)
	ent=entropy(f,disk(ds))
	#print ent.max()
	#img1 = ax1.imshow(ent, cmap=plt.cm.jet)
	#ax1.set_title("Entropy")
	#ax1.axis("off")
	#fig.colorbar(img1, ax=ax1)
	#fig.tight_layout()
	#plt.show()
	return ent

def dilate(phantom,sel=6):
	selem = disk(sel)
	dilated = dilation(phantom, selem)
	plot_comparison(phantom, dilated, 'dilation')
	return dilated

def thresh(ent,rh):
	thresh=ent.max()*rh
	binary=ent>thresh
	return binary

def mask(binary,image):
	mask=binary*image
	return mask

""" Thresholding the image"""
def symmetrize(left,right,hold):
	thpl=thresh(right,hold)
	thfl=thresh(left,hold)
	plot_image = np.concatenate((thfl, thpl), axis=1)
	return (plot_image,thfl,thpl)
	#plt.imshow(plot_image)
	#plt.show()

def learn_threshold(left,right):
	threshold=1
	th=[]
	sensitivity=[]
	while(threshold>0.5):
		l=thresh(left,threshold)
		r=thresh(right,threshold)
		sensitivity.append(abs(sum(sum(l))-sum(sum(r))))
		th.append(threshold)
		threshold=threshold-0.025
	return (max(sensitivity),th[sensitivity.index(max(sensitivity))])




def checkdiff(left,right):
	threshold=1##############
	sensitivity=abs(sum(sum(left))-sum(sum(right)))
	if sensitivity>threshold:
		if sum(sum(left))>sum(sum(right)):
			return(-1,sensitivity)
		else:
			return(1,sensitivity)
	else:
		return(0,sensitivity)



def horizontal(target,reference,coods):
	(cx1,cx2,cy1,cy2)=([],[],[],[])
	hor=target.shape[1]
	ts=hor/4
	a=np.hsplit(target,2)
	cy1.append(0)
	cy2.append(target.shape[0])
	cy1.append(0)
	cy2.append(target.shape[0])
	cx1.append(0)
	cx2.append(math.floor(target.shape[1]/2))
	cx1.append(math.floor(target.shape[1]/2))
	cx2.append(target.shape[1])
	b=np.hsplit(reference,2)
	c=np.hsplit(target,[ts,hor-ts])
	cy1.append(0)
	cy2.append(target.shape[0])
	cx1.append(ts)
	cx2.append(hor-ts)
	a.append(c[1])
	c=np.hsplit(reference,[ts,hor-ts])
	b.append(c[1])
	sen=[]
	(indicator,sensitivity)=checkdiff(a[0],b[0])
	sen.append(sensitivity)
	(indicator,sensitivity)=checkdiff(a[1],b[1])
	sen.append(sensitivity)
	(indicator,sensitivity)=checkdiff(a[2],b[2])
	sen.append(sensitivity)
	x_ref=coods[0]
	y_ref=coods[2]
	coods[0]=x_ref+cx1[sen.index(max(sen))]
	coods[1]=x_ref+cx2[sen.index(max(sen))]
	coods[2]=y_ref+cy1[sen.index(max(sen))]
	coods[3]=y_ref+cy2[sen.index(max(sen))]
	return (a[sen.index(max(sen))],b[sen.index(max(sen))],coods)


def vertical(target,reference,coods):
	(cx1,cx2,cy1,cy2)=([],[],[],[])
	ver=target.shape[0]
	ts=ver/4
	a=np.vsplit(target,2)
	cx1.append(0)
	cx2.append(target.shape[1])
	cx1.append(0)
	cx2.append(target.shape[1])
	cy1.append(0)
	cy2.append(math.floor(target.shape[0]/2))
	cy1.append(math.floor(target.shape[0]/2))
	cy2.append(target.shape[0])
	b=np.vsplit(reference,2)
	c=np.vsplit(target,[ts,ver-ts])
	cx1.append(0)
	cx2.append(target.shape[1])
	cy1.append(ts)
	cy2.append(ver-ts)
	a.append(c[1])
	c=np.vsplit(reference,[ts,ver-ts])
	b.append(c[1])
	sen=[]
	(indicator,sensitivity)=checkdiff(a[0],b[0])
	sen.append(sensitivity)
	(indicator,sensitivity)=checkdiff(a[1],b[1])
	sen.append(sensitivity)
	(indicator,sensitivity)=checkdiff(a[2],b[2])
	sen.append(sensitivity)
	x_ref=coods[0]
	y_ref=coods[2]
	coods[0]=x_ref+cx1[sen.index(max(sen))]
	coods[1]=x_ref+cx2[sen.index(max(sen))]
	coods[2]=y_ref+cy1[sen.index(max(sen))]
	coods[3]=y_ref+cy2[sen.index(max(sen))]
	return (a[sen.index(max(sen))],b[sen.index(max(sen))],coods)


def detect(target,reference,once,coods=[0,0,0,0]):
	if once==1:
		reference=np.fliplr(reference)
		once=0
		coods=[0,target.shape[1],0,target.shape[0]]
		target=pad(target)
		reference=pad(reference)
	x=target.shape[0]
	y=target.shape[1]
	if min(x,y)<800:
		return (target,coods)
	(target,reference,coods)=horizontal(target,reference,coods)
	(target,reference,coods)=vertical(target,reference,coods)
	(target,coods)=detect(target,reference,once,coods)
	return (target,coods)

def comp(a,b):
	if sum(sum(a))>sum(sum(b)):
		return -1
	else:
		return 1


def symdetect(left,right):
	threshold=1
	indicator=0
	while (threshold>=0.5):
		(image,leftim,rightim)=symmetrize(left,right,threshold)
		(check,sv)=checkdiff(leftim,rightim)
		if check==-1:
			print("Left\n")
			square=detect(leftim,rightim,1)
			indicator=-1
			break
		if check==1:
			print("Right\n")
			square=detect(rightim,leftim,1)
			indicator=1
			break
		threshold=threshold-0.025
	return (square,indicator)

indexl = [1,1,1,1,0,0,1,1,1,1,0,1,1]
patient = ["98","1626","3642","5424","10196","21107","26102","29374","29504","40034","51323","53559","67906"] 
l = ["100152","111359","121176","151846","151894","267643","203943","337668","317772","402752","485208","502860","673773"]
r = ["100151","111359","121174","151845","151892","267646","203941","337663","317769","402750","485201","502858","673770"]
#Fails for 11 th one
def find_patch(num=10):
	left=l[num]+".dcm"
	right=r[num]+".dcm"
	left=loadfile(left)
	right=loadfile(right)
	left=pad(left)
	right=pad(right)
	(sensitivity,th)=learn_threshold(left,right)
	print(th)
	(a,b,c)=symmetrize(left,right,th)
	if indexl[num]==1:
		(tar,coods)=new_detect(b,c,1)
		print("Left breast has cancer")
		print(coods)
		showpatch(coods,left)
		display(left)
		#showpatch(coods,right)
	else:
		(tar,coods)=new_detect(c,b,1)
		print("Right breast has cancer")
		print(coods)
		showpatch(coods,right)
		display(right)
		#showpatch(coods,left)
	return (tar,b,c,coods)


""" We have to provide target, reference, size of the window, shift to the window"""
def new_master(target,reference,size,shift):
	#sen=[]
	p_size = size
	shift = shift
	h_size = p_size/2
	(ver,hor) = (target.shape[0],target.shape[1])
	print(ver)
	print(hor)
	k = h_size + 5
	sen_max = 0
	x = 0
	y = 0
	while(k < ver):
		l = h_size+5
		while(l < hor):
			(indicator,sensitivity)=checkdiff(target[l-h_size:l+h_size,k-h_size:k+h_size],reference[l-h_size:l+h_size,k-h_size:k+h_size])
			#sen.append(sensitivity)
			print(sensitivity)
			l = l + shift
			#print (l,k)
			if(sensitivity > sen_max):
				sen_max= sensitivity
				(x,y) = (l,k)
		k = k + shift
	print(x,y)
	return (target[x-h_size:x+h_size,y-h_size:y+h_size],reference[x-h_size:x+h_size,y-h_size:y+h_size],[x,y])
		
			
	



def new_detect(target,reference,once):
	if once==1:
		reference=np.fliplr(reference)
		target=pad(target)
		reference=pad(reference)
		once=0
	size=400
	shift=100
	(target,reference,coods)=new_master(target,reference,size,shift)
	cp=[]
	cp.append(coods[1]-size/2)
	cp.append(coods[1]+size/2)
	cp.append(coods[0]-size/2)
	cp.append(coods[0]+size/2)
	return (target,cp)

"""Here alpha refers to which breast is having cancer 
if left breast is having cancer alpha is 1 and for right breast alpha is 0"""
def patch(left,right,alpha=1):
	left = loadfile(left)
	right = loadfile(right)
	left2 = np.fliplr(left)
	right2 = np.fliplr(right)
	(f,thresh) = learn_threshold(left,right)
	(l,m,n) = symmetrize(left,right,thresh)
	if(alpha == 1):
		(m1,n1) = new_detect(m,n,1)
		display(m1)#""" Here I have to change for righ code"""
		return (left[n1[2]:n1[3],n1[0]:n1[1]],right2[n1[2]:n1[3],n1[0]:n1[1]])
	else:
		(m1,n1) = new_detect(n,m,1)
		display(m1)
		return (right[n1[2]:n1[3],n1[0]:n1[1]],left2[n1[2]:n1[3],n1[0]:n1[1]])


def descriptors(image):
	(image) = image
	matrix = PyImageMatrix()
	matrix.allocate(image.shape[1], image.shape[0])
	numpy_matrix = matrix.as_ndarray()
	numpy_matrix[:] = image
	#fv = FeatureVector( name='FromNumpyMatrix', long=True, original_px_plane=matrix)
	fv = FeatureVector( name='numpy_matrix', long=True, original_px_plane=matrix)
	fv.GenerateFeatures(quiet=False, write_to_disk=False)
	return fv.values

def extract_descriptor(left,right,alpha):
	(image1,image2) = patch(left,right,alpha)
	display(image1)
	display(image2)
	a = descriptors(image1)
	b = descriptors(image2)
	return (a,b)



"""Extract Descriptors and train an SVM """
"""Extracting 9 patches with the co-ordinates including the center patch"""

def new_patch(left,right,patch_size,alpha=1):
	half_size = patch_size/2
	left = loadfile(left)
	width = left.shape[1]
	height = left.shape[0]
	right = loadfile(right)
	left2 = np.fliplr(left)
	right2 = np.fliplr(right)
	(f,thresh) = learn_threshold(left,right)
	(l,m,n) = symmetrize(left,right,thresh)
	patches =[]
	if(alpha == 1):
	   	(m1,n1) = newsquare_detect(m,n,1)
	   	useful_patches = []
	   	non_useful_patches = []
	   	intensity = [None] * 9
	   	for i in range(len(n1)):
	   		co = n1[i]
	   		if(co[3] <= width and co[1] <= height):
	   			#useful_patches[i] = left[co[2]:co[3],co[0]:co[1]]
	   			useful_patches.append(left[co[2]:co[3],co[0]:co[1]])
	   			non_useful_patches.append(right2[co[2]:co[3],co[0]:co[1]])
	   			intensity[i] = np.sum(np.sum(useful_patches[i]))
	   		else:
	   			#useful_patches[i] = 0
	   			useful_patches.append(0)
	   			non_useful_patches.append(0)
	   			intensity[i] = 0
	   	sorted_indices = np.argsort(intensity)
	   	""" This is for taking top five patches"""
	   	cancer_patches = []
	   	#length = intensity.shape[0]
	   	non_cancer_patches=[]
	   	length = len(intensity)
	   	for i in range(5):
	   		cancer_patches.append(useful_patches[sorted_indices[length-i-1]])
	   		non_cancer_patches.append(non_useful_patches[sorted_indices[length-i-1]])
		return (cancer_patches,non_cancer_patches)
	else:
		(m1,n1) = newsquare_detect(n,m,1)
		useful_patches = []
		non_useful_patches = []
	   	intensity = [None] * 9
	   	for i in range(len(n1)):
	   		co = n1[i]
	   		if(co[3] <= width and co[1] <= height):
	   			#useful_patches[i] = right[co[2]:co[3],co[0]:co[1]]
	   			useful_patches.append(right[co[2]:co[3],co[0]:co[1]])
	   			non_useful_patches.append(left2[co[2]:co[3],co[0]:co[1]])
	   			intensity[i] = np.sum(np.sum(useful_patches[i]))
	   		else:
	   			#useful_patches[i] = 0
	   			useful_patches.append(0)
	   			non_useful_patches.append(0)
	   			intensity[i] = 0
	   	sorted_indices = np.argsort(intensity)
	   	""" This is for taking top five patches which are having cancer"""
	   	cancer_patches = []
	   	non_cancer_patches = []
	   	print (len(non_useful_patches))
	   	length = 9
	   	"""I can change"""
	   	for i in range(5):
	   		cancer_patches.append(useful_patches[sorted_indices[length-i-1]])
	   		non_cancer_patches.append(non_useful_patches[sorted_indices[length-i-1]])
		return (cancer_patches,non_cancer_patches)


def newsquare_detect(target,reference,once):
	""" h_s is half_size
	    cood - tuple of all the patches around the center
	    all_coods - It will return the co-ordinates for all the nine patches what we want to extract"""
	if once==1:
		reference=np.fliplr(reference)
		target=pad(target)
		reference=pad(reference)
		once=0
	size=400
	h_s = size/4
	shift=100
	(target,reference,coods)=newsquare_master(target,reference,size,shift)
	print(coods)
	centers = [[0,0],[0,h_s],[h_s,-h_s],[-h_s,0],[-h_s,-h_s],[0,-h_s],[h_s,-h_s],[h_s,0],[h_s,h_s]]
	all_coods = []
	for i in range(len(centers)):
		s = []
		cp=[]
		sample = centers[i]
		#s[0] = coods[0] + sample[0]
		#s[1] = coods[1] + sample[1]
		s.append(coods[0] + sample[0])
		s.append(coods[1] + sample[1])
		cp.append(s[1]-h_s)
		cp.append(s[1]+h_s)
		cp.append(s[0]-h_s)
		cp.append(s[0]+h_s)
		all_coods.append(cp)
	return (target,all_coods)


def newsquare_master(target,reference,size,shift):
	p_size = size
	shift = shift
	h_size = p_size/2
	(ver,hor) = (target.shape[0],target.shape[1])
	#print(ver)
	#print(hor)
	k = h_size + 5
	sen_max = 0
	x = 0
	y = 0
	#print(k)
	while(k < ver):
		l = h_size+5
		while(l < hor):
			(indicator,sensitivity)=checkdiff(target[l-h_size:l+h_size,k-h_size:k+h_size],reference[l-h_size:l+h_size,k-h_size:k+h_size])
			l = l + shift
			if(sensitivity > sen_max):
				sen_max= sensitivity
				(x,y) = (l,k)
				#print(x,y)
		k = k + shift
	return (target[x-h_size:x+h_size,y-h_size:y+h_size],reference[x-h_size:x+h_size,y-h_size:y+h_size],[x,y])


"""new function for calculating descriptors and storing them as pickle files"""
def final_descriptors():
	once = 1
	indexl = [1,1,1,1,0,0,1,1,1,1,0,1,1]
	""" Change for patient 2"""
	patient = ["98","1626","3642","5424","10196","21107","26102","29374","29504","40034","51323","53559","67906"] 
	l = ["100152","111359","121176","151846","151894","267643","203943","337668","317772","402752","485208","502860","673773"]
	r = ["100151","111358","121174","151845","151892","267646","203941","337663","317769","402750","485201","502858","673770"]
	descriptor = []
	non_descriptor = []
	for i in range(0,len(indexl)):
		a = '/home/jagadeesh/Dream/images/' + l[i] + '.dcm'
		b = '/home/jagadeesh/Dream/images/' + r[i] + '.dcm'
		(c,d) = new_patch(a,b,400,indexl[i])
		for j in range(len(c)):
			desc = descriptors(c[j])
			non_desc = descriptors(d[j])
			if(once == 1):
				descriptor.append(desc)
				non_descriptor.append(non_desc)
				once = 0
			else:
				#descriptor = np.concatenate((descriptor,desc),axis=1)
				descriptor.append(desc)
				non_descriptor.append(non_desc)
	return (descriptor,non_descriptor)


		
			

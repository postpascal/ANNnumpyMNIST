import numpy as np 
import pickle
import os
from PIL import Image 
import matplotlib.pyplot as plt
# MAINTAINER: Keke Zhang,28 Sep 2017,opendayoff@163.com
def softmax(z):
	soft_sum=np.sum(np.exp(z/100), axis=1)
	y=np.exp(z/100)

	for i,j in enumerate(soft_sum):
		y[i]=y[i]/j
	return y

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    # return rgb[:,:,0]

def predict(image,w):
	x=image.reshape((1,785))
	y=softmax(np.dot(x,w))
	predict=np.argmax(y)
	print("predict:",predict)
	return predict

def show_image(image):
	plt.imshow(image)
	plt.show()

def read_images(path):
	image_path=[]
	for dir_entry in os.listdir(path):
		if '.DS_Store' == dir_entry:
			continue
		dir_entry_path = os.path.join(path,dir_entry)
		if os.path.isfile(dir_entry_path):
			image_path.append(dir_entry_path)
	return image_path

def path2image(path):
	image=np.asarray(Image.open(path).resize((28,28)))
	if len(image.shape)>2:
		image=255-rgb2gray(image)
	image=image.reshape((1,784))
	image=np.append(image,1)
	return image

if __name__=='__main__':
	folder_path="/Users/zhangkeke/Desktop/tupu/images"
	image_path=read_images(folder_path)
	correct=0
	with open("mnist_model",'rb') as file:
		w=pickle.load(file)
		for path in image_path:
			image=path2image(path)
			num=predict(image,w)

		
	



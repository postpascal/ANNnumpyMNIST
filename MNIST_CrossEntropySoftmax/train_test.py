import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
import sys
# MAINTAINER: Keke Zhang,28 Sep 2017,opendayoff@163.com
def read_images(file_name):
	with open(file_name,'rb') as file:
		im_buf=file.read()
		ind=0
		im=[i for i in im_buf][16:]
		size=int(len(im)/784)
		images=[im[x*784:x*784+784] for x in range(size)]
	return images
		
def read_labels(file_name):
	with open(file_name,'rb') as file:
		labels=file.read()
	return [i for i in labels][8:]

def show_image(image,label):

	image=np.asarray(image[:784]).reshape((28,28))
	plt.imshow(image)
	plt.show()

	# Y=XW+b

def train(images,labels):
	# x=[x 1]

	batch_num=int(len(images)/batch_size)
	w=init_norm(785,10)
	for epoch in range(10):
		print("<------------epoch--------->:",epoch+1)
		correct=0
		num=0
		for i in range(batch_num):
			x=np.asarray(images[i*batch_size:i*batch_size+batch_size]).reshape((batch_size,785))
			label=labels[i*batch_size:i*batch_size+batch_size]		
			y=softmax(np.dot(x,w))
			loss,w,correct,num=ce_loss(y,label,x,w,correct,num)

		print("training accuracy:",correct/num)
		print("training error:",1-correct/num)
		test(test_images,test_labels,w)
	return w
			

def test(images,labels,w):
	correct=0
	x=np.asarray(images).reshape((10000,785))
	y=softmax(np.dot(x,w))
	for label, y_i in zip(labels,y):
		predict=np.argmax(y_i)
		correct=correct+reference[label][predict]

	print("test accuracy:",correct/len(labels))
	print("test error:",1-correct/len(labels))


def ce_loss(y,labels,x,w,correct,num):
	loss=0
	for y_i,label,x_i in zip(y,labels,x):
		loss=-np.dot(reference[label],np.log(y_i))
		bp1=np.asarray(y_i-reference[label]).reshape((1,10))
		x_i=x_i.reshape(785,1)
		w=w-0.0005*np.dot(x_i,bp1)/batch_size

		predict=np.argmax(y_i)
		correct=correct+reference[label][predict]
		num=num+1
	
	return loss,w,correct,num

def init_norm(h,w):
	# weights=[w;b]
	weights=np.zeros((h,w))
	for i in range(h-1):
		weights[i]=np.random.normal(0, 0.001, w)
	return weights

def softmax(z):
	soft_sum=np.sum(np.exp(z/100), axis=1)
	y=np.exp(z/100)

	for i,j in enumerate(soft_sum):
		y[i]=y[i]/j
	return y

def onehot_labels():
	reference={}
	for i in range(10):
		flat=np.zeros((10))
		flat[i]=1
		reference[i]=flat
	return reference


def data_shuffle(images,labels):
	# x=[x 1]
	new_labels=[]
	for image,label in zip(images,labels):
		image.append(1)
		image.append(label)
	random.shuffle(images)
	for image in images:
		new_labels.append(image.pop())	
	return images,new_labels
if __name__=='__main__':
	if str(sys.argv[1])=="train":
		batch_size=100
		reference=onehot_labels()
		print("reading MNIST data")
		test_images=read_images("t10k-images-idx3-ubyte")
		test_labels=read_labels("t10k-labels-idx1-ubyte")
		
		training_images=read_images("train-images-idx3-ubyte")
		training_labels=read_labels("train-labels-idx1-ubyte")

		training_images,training_labels=data_shuffle(training_images,training_labels)
		test_images,test_labels=data_shuffle(test_images,test_labels)
		w=train(training_images,training_labels)
		with open("mnist_model",'wb') as file:
			print("saving weights as mnist_model")
			pickle.dump(w,file)
	elif str(sys.argv[1])=="test":
		reference=onehot_labels()
		print("reading test data")
		test_images=read_images("t10k-images-idx3-ubyte")
		test_labels=read_labels("t10k-labels-idx1-ubyte")
		test_images,test_labels=data_shuffle(test_images,test_labels)
		with open("mnist_model",'rb') as file:
			w=pickle.load(file)
			test(test_images,test_labels,w)


	else:
		print("incorrect arguments")
	

	



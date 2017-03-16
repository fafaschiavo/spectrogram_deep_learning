import os
import cPickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from sklearn import datasets, svm, metrics
from sklearn.datasets import load_sample_image

digits = datasets.load_digits()
# print "This is image 0:"
# print len(digits['images'])

n_samples = len(digits['images'])
data = digits['images'].reshape((n_samples, -1))
learning_data = data[:n_samples/2]
testing_data = data[n_samples/2:]
# print data

target_data = digits['target'][:n_samples/2]
testing_target_data = digits['target'][n_samples/2:]

print 'Here ------------------------------------------------------------'
print target_data

classifier = svm.SVC(gamma=0.001)
classifier.fit(learning_data, target_data)
predicted_data = classifier.predict(testing_data)

error_counter = 0
total_counter = 0
for index in predicted_data:
	total_counter = total_counter + 1
	if predicted_data[index] != testing_target_data[index]:
		error_counter = error_counter + 1

print error_counter
print total_counter
accuracy = float(error_counter)/float(total_counter)
accuracy = accuracy*100
print str(accuracy) + '%' 





# ------------------------------------------------ Pra importar imagens no formato certo
# new_image = mpimg.imread('image1.png')
# print new_image


# ------------------------------------------------ Pra shufle duas listas da mesma forma
# from random import shuffle

# list1_shuf = []
# list2_shuf = []
# index_shuf = range(len(list1))
# shuffle(index_shuf)
# for i in index_shuf:
#     list1_shuf.append(list1[i])
#     list2_shuf.append(list2[i])





# ------------------------------------------------ Pra salvar o classifier treinado
# import cPickle
# # save the classifier
# with open('my_dumped_classifier.pkl', 'wb') as fid:
#     cPickle.dump(gnb, fid)    

# # load it again
# with open('my_dumped_classifier.pkl', 'rb') as fid:
#     gnb_loaded = cPickle.load(fid)

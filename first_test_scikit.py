import os
import cPickle
from random import shuffle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import time

from sklearn import datasets, svm, metrics
from sklearn.datasets import load_sample_image

start_time = time.time()

data_target = []
data = []

int_code = 0
folder_name = 'black-metal'
files = os.listdir(folder_name + '/')

amount_done = 0
for image in files[:500]:
	new_image = mpimg.imread(folder_name + '/' + image)
	cropped_image = new_image[150:1065][240:400]
	data.append(cropped_image)
	data_target.append(int_code)
	amount_done = amount_done + 1
	print str(amount_done) + ' - ' + folder_name

int_code = 1
folder_name = 'tango'
files = os.listdir(folder_name + '/')
# print files

amount_done = 0
for image in files[:500]:
	new_image = mpimg.imread(folder_name + '/' + image)
	cropped_image = new_image[150:1065][240:400]
	data.append(cropped_image)
	data_target.append(int_code)
	amount_done = amount_done + 1
	print str(amount_done) + ' - ' + folder_name

n_samples = len(data)
data = np.asarray(data)
data = data.reshape((n_samples, -1))



data_shuf = []
data_target_shuf = []
index_shuf = range(len(data))
shuffle(index_shuf)
for i in index_shuf:
    data_shuf.append(data[i])
    data_target_shuf.append(data_target[i])




#separate learning and testing date
learning_data = data_shuf[:n_samples/2]
testing_data = data_shuf[n_samples/2:]

target_data = data_target_shuf[:n_samples/2]
testing_target_data = data_target_shuf[n_samples/2:]

# learning_data = data_shuf[:5]
# testing_data = data_shuf[5:]

# target_data = data_target_shuf[:5]
# testing_target_data = data_target_shuf[5:]





classifier = svm.SVC(gamma=0.001, verbose=True)
print 'Fitting...'
classifier.fit(learning_data, target_data)
print 'Predicting...'
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
print 'Porcentagem de erros - ' + str(accuracy*100) + '%'
accuracy = 100 - (accuracy*100)
print 'Porcentagem de acertos - ' + str(accuracy) + '%'

# print 'Now saving classifier...'
# save the classifier
# with open('classifiers/my_dumped_classifier_001_100.pkl', 'wb') as file_to_save:
#     cPickle.dump(classifier, file_to_save)
    
# print 'Save completed'




print("--- %s seconds ---" % (time.time() - start_time))



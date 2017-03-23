import os
import cPickle
from random import shuffle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import time

from sklearn import datasets, svm, metrics
from sklearn.datasets import load_sample_image

def crop_grayscale_flatten_conversion(image_to_convert):
	cropped_image = image_to_convert[228:710, 244:1410]
	
	conversion_array = get_conversion_grayscale(image_to_convert)
	greyscale_cropped_image = rgb2gray(cropped_image, conversion_array)
	data = greyscale_cropped_image
	data = np.asarray(data)
	data = data.reshape((1, -1))
	return data

def get_conversion_grayscale(new_image):
	a = np.array([new_image[148][1515][:3], new_image[606][1515][:3], new_image[1065][1515][:3]])
	b = np.array([1, 0.5, 0])
	x = np.linalg.solve(a, b)
	return x

def convert_to_greyscale(cropped_image, conversion_array):
	# converted_cropped_image = np.zeros(shape=(1,1166))
	print cropped_image.shape
	converted_cropped_image = np.random.rand(cropped_image.shape[0], cropped_image.shape[1])
	line_counter = 0
	row_counter = 0
	for line in cropped_image:
		row_counter = 0
		for pixel in line:
			new_pixel = pixel[0]*conversion_array[0] + pixel[1]*conversion_array[1] + pixel[2]*conversion_array[2]
			new_pixel = int(new_pixel*255)
			converted_cropped_image[line_counter][row_counter] = new_pixel
			row_counter = row_counter + 1
		line_counter = line_counter + 1

	return converted_cropped_image

def rgb2gray(cropped_image, conversion_array):

    r, g, b = cropped_image[:,:,0], cropped_image[:,:,1], cropped_image[:,:,2]
    gray = conversion_array[0] * r + conversion_array[1] * g + conversion_array[2] * b

    return gray

start_time = time.time()

number_of_samples = 10 #number oof samples for EACH CLASS!
precentage_of_learning = 0.75

data_target = []
data = []
learning_data =[]
target_data =[]
testing_data =[]
testing_target_data =[]

int_code = 0
folder_name = 'tango'
files = os.listdir(folder_name + '/')
for file in files:
	if file.startswith('.') or '.png' not in file:
		del files[files.index(file)]

amount_done = 0
for image in files[:number_of_samples]:
	new_image = mpimg.imread(folder_name + '/' + image)
	cropped_image = new_image[228:710, 244:1410]
	# cropped_image = new_image[610:710, 1310:1410]
	
	conversion_array = get_conversion_grayscale(new_image)
	greyscale_cropped_image = rgb2gray(cropped_image, conversion_array)

	# plt.subplot(2,1,1)
	# imgplot = plt.imshow(greyscale_cropped_image, cmap='gray')

	# plt.subplot(2,1,2)
	# plt.imshow(cropped_image)
	# plt.show()

	if amount_done < number_of_samples*precentage_of_learning:
		learning_data.append(greyscale_cropped_image)
		target_data.append(int_code)
	else:
		testing_data.append(greyscale_cropped_image)
		testing_target_data.append(int_code)

	# data.append(greyscale_cropped_image)
	# data_target.append(int_code)
	amount_done = amount_done + 1
	print str(amount_done) + ' - ' + folder_name

int_code = 1
folder_name = 'black-metal'
files = os.listdir(folder_name + '/')
for file in files:
	if file.startswith('.') or '.png' not in file:
		del files[files.index(file)]

amount_done = 0
for image in files[:number_of_samples]:
	new_image = mpimg.imread(folder_name + '/' + image)
	cropped_image = new_image[228:710, 244:1410]
	
	conversion_array = get_conversion_grayscale(new_image)
	greyscale_cropped_image = rgb2gray(cropped_image, conversion_array)

	if amount_done < number_of_samples*precentage_of_learning:
		learning_data.append(greyscale_cropped_image)
		target_data.append(int_code)
	else:
		testing_data.append(greyscale_cropped_image)
		testing_target_data.append(int_code)

	# data.append(greyscale_cropped_image)
	# data_target.append(int_code)
	amount_done = amount_done + 1
	print str(amount_done) + ' - ' + folder_name

int_code = 2
folder_name = 'jazz'
files = os.listdir(folder_name + '/')
for file in files:
	if file.startswith('.') or '.png' not in file:
		del files[files.index(file)]

amount_done = 0
for image in files[:number_of_samples]:
	new_image = mpimg.imread(folder_name + '/' + image)
	cropped_image = new_image[228:710, 244:1410]
	
	conversion_array = get_conversion_grayscale(new_image)
	greyscale_cropped_image = rgb2gray(cropped_image, conversion_array)

	if amount_done < number_of_samples*precentage_of_learning:
		learning_data.append(greyscale_cropped_image)
		target_data.append(int_code)
	else:
		testing_data.append(greyscale_cropped_image)
		testing_target_data.append(int_code)

	# data.append(greyscale_cropped_image)
	# data_target.append(int_code)
	amount_done = amount_done + 1
	print str(amount_done) + ' - ' + folder_name




n_samples = len(learning_data)
learning_data = np.asarray(learning_data)
learning_data = learning_data.reshape((n_samples, -1))

n_samples = len(testing_data)
testing_data = np.asarray(testing_data)
testing_data = testing_data.reshape((n_samples, -1))

learning_data_shuffle = []
target_data_shuffle = []
index_shuf = range(len(learning_data))
shuffle(index_shuf)
for i in index_shuf:
    learning_data_shuffle.append(learning_data[i])
    target_data_shuffle.append(target_data[i])





classifier = svm.SVC(gamma=0.001, verbose=True, C=100)
print 'Fitting...'
classifier.fit(learning_data_shuffle, target_data_shuffle)
print 'Predicting...'
predicted_data = classifier.predict(testing_data)

error_counter = 0
total_counter = 0
for index in predicted_data:
	if predicted_data[total_counter] != testing_target_data[total_counter]:
		error_counter = error_counter + 1
	total_counter = total_counter + 1

accuracy = float(error_counter)/float(total_counter)
print 'Porcentagem de erros - ' + str(accuracy*100) + '%'
accuracy = 100 - (accuracy*100)
print 'Porcentagem de acertos - ' + str(accuracy) + '%'




print "Classification report for classifier %s:\n%s\n" % (classifier, metrics.classification_report(testing_target_data, predicted_data))
print "Confusion matrix:\n%s" % metrics.confusion_matrix(testing_target_data, predicted_data)




my_own_test_image = mpimg.imread('samples' + '/' + 'black_metal_0AE1wd4ZOIDl1GpO8EBQfp.png')
my_own_test_image = crop_grayscale_flatten_conversion(my_own_test_image)
print 'black_metal_0AE1wd4ZOIDl1GpO8EBQfp.png'
print 'prediction:'
print classifier.predict(my_own_test_image)

my_own_test_image = mpimg.imread('samples' + '/' + 'black_metal_0AharcdMp1s3BgYTmfLzBo.png')
my_own_test_image = crop_grayscale_flatten_conversion(my_own_test_image)
print 'black_metal_0AharcdMp1s3BgYTmfLzBo.png'
print 'prediction:'
print classifier.predict(my_own_test_image)

my_own_test_image = mpimg.imread('samples' + '/' + 'black_metal_0AnfcHtCViS8FWZmzUK6fZ.png')
my_own_test_image = crop_grayscale_flatten_conversion(my_own_test_image)
print 'black_metal_0AnfcHtCViS8FWZmzUK6fZ.png'
print 'prediction:'
print classifier.predict(my_own_test_image)

my_own_test_image = mpimg.imread('samples' + '/' + 'tango_0B9veKH2yIJnM2a78lWYwg.png')
my_own_test_image = crop_grayscale_flatten_conversion(my_own_test_image)
print 'tango_0B9veKH2yIJnM2a78lWYwg.png'
print 'prediction:'
print classifier.predict(my_own_test_image)

my_own_test_image = mpimg.imread('samples' + '/' + 'tango_0ahYxXxa7BeyihIkGGlYcs.png')
my_own_test_image = crop_grayscale_flatten_conversion(my_own_test_image)
print 'tango_0ahYxXxa7BeyihIkGGlYcs.png'
print 'prediction:'
print classifier.predict(my_own_test_image)

my_own_test_image = mpimg.imread('samples' + '/' + 'tango_0b8qyxnbWOxO4a08Z2TY0X.png')
my_own_test_image = crop_grayscale_flatten_conversion(my_own_test_image)
print 'tango_0b8qyxnbWOxO4a08Z2TY0X.png'
print 'prediction:'
print classifier.predict(my_own_test_image)

my_own_test_image = mpimg.imread('samples' + '/' + 'jazz_0ADAFZE0bYIuaIyPNFl3LE.png')
my_own_test_image = crop_grayscale_flatten_conversion(my_own_test_image)
print 'jazz_0ADAFZE0bYIuaIyPNFl3LE.png'
print 'prediction:'
print classifier.predict(my_own_test_image)

my_own_test_image = mpimg.imread('samples' + '/' + 'jazz_0AEBs3sZ492afceldzJzF0.png')
my_own_test_image = crop_grayscale_flatten_conversion(my_own_test_image)
print 'jazz_0AEBs3sZ492afceldzJzF0.png'
print 'prediction:'
print classifier.predict(my_own_test_image)

my_own_test_image = mpimg.imread('samples' + '/' + 'jazz_0aWMVrwxPNYkKmFthzmpRi.png')
my_own_test_image = crop_grayscale_flatten_conversion(my_own_test_image)
print 'jazz_0aWMVrwxPNYkKmFthzmpRi.png'
print 'prediction:'
print classifier.predict(my_own_test_image)


# print 'Now saving classifier...'
# save the classifier
# with open('classifiers/my_dumped_classifier_001_100.pkl', 'wb') as file_to_save:
#     cPickle.dump(classifier, file_to_save)
    
# print 'Save completed'




print("--- %s seconds ---" % (time.time() - start_time))



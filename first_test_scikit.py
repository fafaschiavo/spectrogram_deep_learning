import os
import cPickle
from random import shuffle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import time

from sklearn import datasets, svm, metrics
from sklearn.datasets import load_sample_image

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

data_target = []
data = []

int_code = 0
folder_name = 'tango'
files = os.listdir(folder_name + '/')
for file in files:
	if file.startswith('.'):
		del files[files.index(file)]

amount_done = 0
for image in files[:200]:
	new_image = mpimg.imread(folder_name + '/' + image)
	cropped_image = new_image[228:710, 244:1410]
	# cropped_image = new_image[610:710, 1310:1410]
	
	conversion_array = get_conversion_grayscale(new_image)
	# greyscale_cropped_image = convert_to_greyscale(cropped_image, conversion_array)
	greyscale_cropped_image = rgb2gray(cropped_image, conversion_array)

	# plt.subplot(2,1,1)
	# imgplot = plt.imshow(greyscale_cropped_image, cmap='gray')

	# plt.subplot(2,1,2)
	# plt.imshow(cropped_image)
	# plt.show()

	data.append(greyscale_cropped_image)
	data_target.append(int_code)
	amount_done = amount_done + 1
	print str(amount_done) + ' - ' + folder_name

int_code = 1
folder_name = 'black-metal'
files = os.listdir(folder_name + '/')
for file in files:
	if file.startswith('.'):
		del files[files.index(file)]

amount_done = 0
for image in files[:200]:
	new_image = mpimg.imread(folder_name + '/' + image)
	cropped_image = new_image[228:710, 244:1410]
	
	conversion_array = get_conversion_grayscale(new_image)
	# greyscale_cropped_image = convert_to_greyscale(cropped_image, conversion_array)
	greyscale_cropped_image = rgb2gray(cropped_image, conversion_array)

	data.append(greyscale_cropped_image)
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



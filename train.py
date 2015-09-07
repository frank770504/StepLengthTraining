import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class sl_training_data:
	def __init__(self, speed, height, age, gender, steplength):
		self.speed = speed
		self.height = height
		self.age = age
		self.gender = gender
		self.steplength = steplength

def read_training_data(name):
	f = open(name, 'r')
	# get item name
	names = f.readline()
	names = names.strip();
	names = names.split(" ");
	training_nplist = np.array([])
	# get the training data: A, and value vector: B
	# The last column is B, and A is the rest part
	for line in f.readlines():
		tmp = line.strip();
		if not tmp or tmp.isspace(): #skip the space line
			continue
		tmp = tmp.split(" ")
		ind = names.index('speed')
		speed = float(tmp[ind])
		ind = names.index('height')
		height = float(tmp[ind])
		ind = names.index('steplength')
		steplength = float(tmp[ind])
		ind = names.index('age')
		age = float(tmp[ind])
		ind = names.index('gender') # gender is a string 'Male'/'Female'
		gender = tmp[ind]
		#convert to float
		training_tmp = sl_training_data(speed, height, age, gender, steplength)
		training_tmp = np.array([training_tmp])
		training_nplist = np.hstack([training_nplist, training_tmp])\
			if training_nplist.size>0 else training_tmp
	return training_nplist

def get_training_matrix(training_data_nplist):
	A = np.array([])
	B = np.array([])
	c = 0
	for t_data in training_data_nplist:
		tA = np.array([t_data.speed, t_data.height, 1])
		tB = np.array([t_data.steplength])
		A = np.vstack([A, tA]) if A.size else tA
		B = np.vstack([B, tB]) if B.size else tB
		# make sure A ans B is ok to use
	return (A, B)

def linear_regression(A, B):
	# doing psuedo inverse
	pA = np.linalg.pinv(A)

	# Ax = B
	# x = pinv(A)*B
	x = np.matrix(pA)*np.matrix(B)
	return x

def get_mean_square_error(training_data_nplist, x):
	A, B = get_training_matrix(training_data_nplist)
	preB = np.matrix(A)*np.matrix(x)
	Err = np.square(preB - B)
	MSE = np.sum(Err) / Err.shape[0]
	return MSE

def print_result(x):
	mapper = []
	names = ['speed', 'height', 'steplength']
	for i in range(0, len(names)-1, 1):
	       tarr = str(x.item(i))+"*"+names[i]
	       mapper.append(tarr)

	mapper.append(str(x.item(len(names)-1)))
	mapper.append(names[-1])

	formatter = "({})"
	for i in range(0, len(mapper)-2, 1):
	       formatter += " + ({})"
	formatter += " = {}"

	# print the result
	print "Formula of Linear Regression"
	print formatter.format(*mapper)


'''
main scrip
'''

f_names = sys.argv;
f_names = f_names[1:];

training_data_nplist = np.array([])

for name in f_names:
	training_nplist = read_training_data(name)
	training_data_nplist = np.hstack([training_data_nplist, training_nplist])\
		if training_data_nplist.size else training_nplist

A, B = get_training_matrix(training_data_nplist)

x = linear_regression(A, B)

print_result(x)

MSE = get_mean_square_error(training_data_nplist, x)

print 'mean square error is {}'.format(MSE)

#TODO analyze the data with age and gender

sp_ind = 0

minA = np.mean(A, axis=0)
minA[sp_ind] = np.amin(A[:,sp_ind]) - 10;
minB = np.matrix(minA)*np.matrix(x)
maxA = np.mean(A, axis=0)
maxA[sp_ind] = np.amax(A[:,sp_ind]) + 10
maxB = np.matrix(maxA)*np.matrix(x)

#plot a speed vs step length figure
f = plt.figure()
plt.plot( A[:,sp_ind], B , '.')
plt.plot([minA.item(0), maxA.item(0)],[minB.item(0), maxB.item(0)], 'r')
plt.title('speed(bpm) vs step length')
plt.xlabel('speed(bpm)')
plt.ylabel('step length(m)')
#plt.show()
plt.savefig('sp_vs_sl.png')

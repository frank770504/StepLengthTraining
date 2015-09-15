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

def get_age_list(training_data_nplist):
	age_nplist = np.array([])
	for d in training_data_nplist:
		tmp = np.array(d.age)
		age_nplist = np.hstack([age_nplist, tmp])\
			if age_nplist.size > 0 else tmp
	age_nplist = np.unique(age_nplist)
	return age_nplist

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

def train_one_set(training_data_nplist):
	A, B = get_training_matrix(training_data_nplist)
	x = linear_regression(A, B)
	MSE = get_mean_square_error(training_data_nplist, x)
	return (x, MSE, A, B)

def get_age_split_nplist(training_data_nplist, th):
	L = np.array([])
	R = np.array([])
	for d in training_data_nplist:
		dnp = np.array([d])
		if d.age > th:
			R = np.hstack([R, dnp]) if R.size > 0 else dnp
		elif d.age < th:
			L = np.hstack([L, dnp]) if L.size > 0 else dnp
	return (L, R)

def get_gender_split_nplist(training_data_nplist):
	F = np.array([])
	M = np.array([])
	for d in training_data_nplist:
		dnp = np.array([d])
		if d.gender == 'Female':
			F = np.hstack([F, dnp]) if F.size > 0 else dnp
		elif d.gender == 'Male':
			M = np.hstack([M, dnp]) if M.size > 0 else dnp
	return (F, M)

def age_decision_stump(training_data_nplist):
	age_nplist = get_age_list(training_data_nplist)
	tmp1 = np.hstack([0, age_nplist[:-2]])
	tmp2 = age_nplist[1:]
	stump_nplist = (tmp1 + tmp2)*0.5
	stump_result = np.array([])
	for th in stump_nplist:
		L, R = get_age_split_nplist(training_data_nplist, th)
		xL, RMSL, A, B = train_one_set(L) if L.size > 0 else (-1, 0, 0, 0)
		xR, RMSR, A, B = train_one_set(R) if R.size > 0 else (-1, 0, 0, 0)
		RMS = RMSL + RMSR
		one_set = np.array([th, RMS])
		stump_result = np.vstack([stump_result, one_set])\
			if stump_result.size > 0 else one_set
	stump_result = stump_result[stump_result[:,1].argsort()]
	return stump_result[0,:]

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

def plot_sp_sl(A, B, x, name):
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
	plt.savefig('sp_vs_sl_{}.png'.format(name))

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

F, M = get_gender_split_nplist(training_data_nplist)

stump = age_decision_stump(M)

L, R = get_age_split_nplist(M, stump[0])
xL, MSEL, AL, BL = train_one_set(L) if L.size > 0 else (-1, 0, 0, 0)
xR, MSER, AR, BR = train_one_set(R) if R.size > 0 else (-1, 0, 0, 0)

print 'For Male'
print 'The stump age is {}'.format(stump[0])
print ""
if L.size <= 0:
	print 'all the age is larger than {}'.format(stump[0])
else:
	print 'The curve for age < {}: '.format(stump[0])
	print_result(xL)
	print 'mean square error is {}'.format(MSEL)
print ""
print ""
print 'The curve for age > {}: '.format(stump[0])
print_result(xR)
print 'mean square error is {}'.format(MSER)

plot_sp_sl(AL, BL, xL, 'Male_L')
plot_sp_sl(AR, BR, xR, 'Male_R')

print ""
stump = age_decision_stump(M)

L, R = get_age_split_nplist(M, stump[0])
xL, MSEL, AL, BL = train_one_set(L) if L.size > 0 else (-1, 0, 0, 0)
xR, MSER, AR, BR = train_one_set(R) if R.size > 0 else (-1, 0, 0, 0)

print 'For Female'
print 'The stump age is {}'.format(stump[0])
print ""
if L.size <= 0:
	print 'all the age is larger than {}'.format(stump[0])
else:
	print 'The curve for age < {}: '.format(stump[0])
	print_result(xL)
	print 'mean square error is {}'.format(MSEL)
print ""
print ""
print 'The curve for age > {}: '.format(stump[0])
print_result(xR)
print 'mean square error is {}'.format(MSER)

plot_sp_sl(AL, BL, xL, 'Female_L')
plot_sp_sl(AR, BR, xR, 'Female_R')

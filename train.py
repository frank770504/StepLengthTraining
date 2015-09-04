import os
import sys
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

A = np.array([])
B = np.array([])

f_names = sys.argv;
f_num = len(sys.argv);
f_names = f_names[1:f_num];


for name in f_names:
	f = open(name, 'r')

	# get item name
	names = f.readline()
	names = names.strip();
	names = names.split(" ");

	# get the training data: A, and value vector: B
	# The last column is B, and A is the rest part
	for line in f.readlines():
		t = line.strip();
		t = t.split(" ")
		tA = np.array(t[:len(names)-1])
		tA = np.append(tA, '1')
		tB = np.array(t[len(names)-1])
		A = np.vstack([A,tA]) if A.size else tA
		B = np.vstack([B,tB]) if B.size else tB

A = A.astype(np.float)
A = np.matrix(A)
B = B.astype(np.float)
B = np.matrix(B)

# doing psuedo inverse
pA = np.linalg.pinv(A)

# Ax = B
# x = pinv(A)*B
x = pA*B

prB = A*x

Err = np.square(prB - B)

MSE = np.sum(Err) / Err.shape[0]

mapper = []

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
print "Mean square error for this training data is {}".format(MSE)

sp_ind = names.index('speed')
hg_ind = names.index('height')


minA = np.mean(A, axis=0)
minA[0,sp_ind] = np.amin(A[:,sp_ind]) - 10;
minB = minA*x
maxA = np.mean(A, axis=0)
maxA[0,sp_ind] = np.amax(A[:,sp_ind]) + 10
maxB = maxA*x

#plot a speed vs step length figure
f = plt.figure(figsize=(8,5))
plt.plot( A[:,sp_ind], B , '.')
plt.plot([minA.item(0), maxA.item(0)],[minB.item(0), maxB.item(0)], 'r')
plt.title('speed(bpm) vs step length')
plt.xlabel('speed(bpm)')
plt.ylabel('step length(m)')
#plt.show()
plt.savefig('sp_vs_sl.png')

import os
import sys
import numpy as np

A = np.array([])
B = np.array([])

f_names = sys.argv;
f_num = len(sys.argv);
f_names = f_names[1:f_num];

f = open(f_names[0], 'r')

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

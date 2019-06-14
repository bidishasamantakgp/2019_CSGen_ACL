import sys

original = open(sys.argv[1]).readlines()
converted = open(sys.argv[2]).readlines()

threshold = float(sys.argv[3])
#for o in original:
for (o, c) in zip(original, converted):
	score = float(o.split("\t")[4])
	#print score
	if score> threshold:
		#print score
		print c.strip()

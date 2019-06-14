from nltk import Tree
import sys
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def corpus2trees(text):
	""" Parse the corpus and return a list of Trees """
	rawparses = text.split("=====")
	print len(rawparses)
	trees = []
 	listtrees = []
	for rp in rawparses:
		
		if not rp.strip():
			continue
		rplist = rp.strip().split("\n")
		if len(rplist) == 0:
			continue
		#print "rplist", rplist
		trees = []
		for i in range(len(rplist)):
			print rplist[i]			
			trees.append(Tree.fromstring(rplist[i]))
		
		listtrees.append(trees)
 
 
	return listtrees

def getsegments(listtree):
	segments = []
	sid = SentimentIntensityAnalyzer()
	for tree in listtree:
		total_len = len(tree.leaves())
		for s in tree.subtrees():
			segment = ' '.join([x for x in s.leaves() if x != ","])
			polarity = abs(sid.polarity_scores(segment)['compound'])
			if s.label()=='VP' or s.label()=='NP': 
			#if s.label()=='VP' or s.label()=='NP' or s.label() == 'SBAR' or polarity > 0.5:
				len_leaves = len(s.leaves())
				#if len_leaves <= total_len/2 : 	
				segments.append(segment)
				#segments.append(' '.join([x for x in s.leaves() if x != ","]))
				
        				
	return segments

if __name__=="__main__":
	f = open(sys.argv[1])
	if not isinstance(f, basestring):
		content = f.read()
	trees =  corpus2trees(content)
	#print trees
	#'''
	for t in trees:
		#print t
		with open(sys.argv[2], 'a') as f:
			f.write("["+','.join(["\""+x+"\"" for x in getsegments(t)])+"]\n")
			#print getsegments(t)
	#'''

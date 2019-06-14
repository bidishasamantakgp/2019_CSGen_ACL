import urllib2
import urllib
import sys
import json
import ast
import re
import time
import argparse
from transliterate_util import *
from collections import defaultdict
import numpy as np

def parseargument():
 	parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
 	parser.add_argument('--synthetic_file', type=str, default='attention.csv',
                        help='synthetic_file which stores the synthetic sentences')
 	parser.add_argument('--align_file', type=str, default='judgements.txt',
                        help='file which stores the judgements')
 	parser.add_argument('--out_file', type=str, default='out',
                        help='file to be written')
 	parser.add_argument('--source_lang', type=str, default='en',
                        help='source language hi, en')
 	parser.add_argument('--target_lang', type=str, default='bn',
                        help='target language to be converted en, hi')
	args = parser.parse_args()
	return args

# the format is <linenumber>, <hindi>, <english>, <codemixed>, <scroes>* , <ens>, <ene>, <his>, <hie>
def getSynthetic(filename):
	indexlist = []
	englishpos = defaultdict(list)
	hindipos = defaultdict(list)
	codemixed = defaultdict(list)
	with codecs.open(filename, encoding='utf-8') as f:
		for line in f:
			#print line
			tokens = line.split("\t")
			#print tokens
			index = int(tokens[0])
			indexlist.append(index)
			codemixed[index].append(tokens[3].strip())
	return (indexlist,codemixed)

def getjudgements(filename):
	judgements = []
	with open(filename, 'r') as f:
		for line in f:
			if len(line.strip()) != 1:
				judgements.append(-1)	
			else:
				judgements.append(int(line.strip()))
	return judgements
	 

def alignSentiment(hparams, indexlist, codemixed, judgements):
	for (i,j) in zip(indexlist, judgements):
		#if i> 6500:
		#	break
		sentences = codemixed[i]
		judgement = j
		if judgement == -1:
			continue
		for sentence in sentences:
			#transliterated = sentence.strip()
			#'''
			hparams.data = sentence.replace(',', '')
			#print hparams.data
			trans_list = transliterate(hparams)
			if len(trans_list) > 0:
				transliterated = trans_list[0]
			else:
				transliterated = sentence
			#print transliterated
			#'''
			try:			
				with open(hparams.out_file, 'a') as fw:	
				#with codecs.open(hparams.out_file, 'a', 'utf-8') as fw:
					fw.write(transliterated + '\t' + str(judgement) + '\n')
			except:
				continue

if __name__=="__main__":
	hparams = parseargument()
	indexlist, codemixed  = getSynthetic(hparams.synthetic_file)
	indexlist = list(set(indexlist))
	judgements = getjudgements(hparams.align_file)
	alignSentiment(hparams, indexlist, codemixed, judgements)
        


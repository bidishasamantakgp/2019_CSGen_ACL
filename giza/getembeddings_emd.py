import sys
import codecs
import argparse
import operator
from collections import defaultdict
import json
from emd import emd
import re
import ast
import copy
import numpy as np

# This file actually contains 
def load_embeddings(endictfile,hidictfile):
        endict = defaultdict()
        hidict = defaultdict()
        
	with codecs.open(endictfile,'r', 'utf-8') as f:
                for line in f:
                        tokens = line.split()
                        #print tokens
			endict[tokens[0]] = [float(x) for x in tokens[-300:]]

        with codecs.open(hidictfile, encoding='utf-8') as f:
                for line in f:
                        tokens = line.split()
                        hidict[tokens[0]] = [float(x) for x in tokens[-300:]]

        return (endict,hidict)


def calculate_emd(hidict, endict, ensentence, hindisentence, probdict, idf_dict, idf_dict_hi):
        x = []
        y = []
        #print('Inside EMD', ensentence)
        en_idf = 0.0
	for word in ensentence:
                word = word.lower()
		if word == '\'re':
			word ='are'
                if word not in ('!','.',':', ';', ','):
                        #print('ENWORD', word)
                        try:
                                x.append(endict[word])
				en_idf += 1.0/(idf_dict[word]+1)
                        except:
                                #print("except", word)
                                continue
                                #print('Error', word)
	hi_idf = 0.0
	for word in hindisentence:
                if word not in ('!','.',':', ';', ','):
                        #print('HIWORD', word)
                        try:
                                y.append(hidict[word])
				hi_idf += 1.0/(idf_dict_hi[word]+1)
                        except:
                                #print("except", word)
                                continue
                                #print('Error', word)
	distance=np.zeros((len(ensentence), len(hindisentence)))
	for en in range(len(ensentence)):
		for hi in range(len(hindisentence)):
			#print "Debug",probdict[ensentence[en]]
			try:
				distance[en][hi] = (1.0 - probdict[ensentence[en]][hindisentence[hi]])
			except:
				distance[en][hi] = 1.0
        distVal= 99
        if len(y) > 0 and len(x)> 0:
                distVal = emd(np.array(x),np.array(y), D=distance) 
        return distVal

def parsearguments():
        parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--src', type=str, default='en',
                        help='source language')
        parser.add_argument('--tgt', type=str, default='hi',
                        help='target language')
	parser.add_argument('--giza_prefix', type=str, default='answer.A3.final',
                        help='data file to store maping of hindi and english')
        parser.add_argument('--segment_file', type=str, default='segment.txt',
                        help='file to store the segmentation information genearted by a grammar')
        parser.add_argument('--map_file', type=str, default='answer.actual.ti.final',
                        help='file to store the mapping')
	parser.add_argument('--sentence_prefix', type=str, default='english_senetence.txt',
                        help='file containing the sentences')
	parser.add_argument('--sample_sent', type=str, default='sample.txt',
                        help='file containing the sampled parallel corpus')
	parser.add_argument('--embedding_prefix', type=str, default='embedding',
                        help='multilingual word embedding')
 	parser.add_argument('--output_file', type=str, default='output.txt',
                        help='output to be stored')
		
	hparams = parser.parse_args()
        return hparams

def get_idfdict(filename):
	freq_dict = defaultdict(int)
	with codecs.open(filename, encoding="utf-8") as f:
		lines = f.readlines()
	for line in lines:	
		words = line.split()
		for word in words:
			freq_dict[word] += 1
	return freq_dict
	

def get_probdict(filename):
  probdict = defaultdict(defaultdict)
  with codecs.open(filename, encoding="utf-8") as f:
  #with open(filename) as f:
      lines = f.readlines()
      for line in lines:
          tokens = line.split()
	  #print tokens[0], tokens[2]
          probdict[tokens[0].replace('.', '').strip()][tokens[1].strip()] = float(tokens[2])
  return probdict

def get_stopword(filename):
  with codecs.open(filename, encoding="utf-8") as f:
    lines = f.readlines()
  return lines

def getembeddings(segments, engsentence, hindisentence, mapping, probdict, idf_dict, idf_dict_hi, endict, hidict, outputfile, index):
  
  enlen = len(engsentence)
  hindilen = len(hindisentence)
  newenglishsentence = copy.copy(engsentence)
  newhindisentence = copy.copy(hindisentence)

  for segment in segments:
          segmentlist = segment.strip().replace('-LSB-','[').replace('-RSB-',']').split()
          if len(segmentlist) == enlen:
		continue
	  
	  random_seed1 = -1

          count = 0
          for word in engsentence:
                if segmentlist[0] in word:
                        break
                count += 1
          random_seed1 = count
          #random_seed1 = engsentence.index(segmentlist[0])
          random_seed2 = -1
          reverselist = copy.copy(engsentence)
          reverselist.reverse()
          count = 0
          for word in reverselist:
                count += 1
                if segmentlist[-1] in word:
                        break
                #count += 1
          random_seed2 = enlen - count

          #mapping = ''
          hindisegment = ''
          indexlist = []
          newenglishsentence = copy.copy(engsentence)
          newhindisentence = copy.copy(hindisentence)
	  segment_dict = defaultdict()
          for l in range(1,max(len(segmentlist),hindilen-1) + 1):
                     for k in range(hindilen - l):
                        j = min(k + l -1, hindilen-1)

                        indexlist = []
                        count = 0
                        probmul = 1
                        overallenglist = []
                        score = 0
			emd_in = calculate_emd(hidict, endict, segmentlist, hindisentence[k:j+1], probdict, idf_dict, idf_dict_hi)
                        engoutseg = engsentence[0:max(0,random_seed1)]+ engsentence[random_seed2+1:]
                        hioutseg = hindisentence[0:k]+ hindisentence[j+1:]
                        emd_out = calculate_emd(hidict, endict, engoutseg, hioutseg, probdict, idf_dict, idf_dict_hi)
                        score = emd_in + emd_out
			newsentence = [' '.join(engsentence[0:max(0,random_seed1)]), ' '.join(hindisentence[k:j+1]), ' '.join(engsentence[random_seed2+1:])]
                        segment_dict[' '.join(newsentence)] = (score, emd_in, emd_out, random_seed1 - 1, enlen - random_seed2 - 1, k, j)
                  # ascending order sorting
          newenglishsentence[random_seed1:random_seed2+1] = [x.upper() for x in newenglishsentence[random_seed1: random_seed2+1]]
	  sorted_candidates = sorted(segment_dict.items(), key=operator.itemgetter(1,0))
	  with codecs.open(outputfile, 'a', 'utf-8') as csvfile:
                        for (candidate, (score, score_in, score_out, rs1, rs2, k , j)) in sorted_candidates[:1]:
                                csvfile.write(str(index)+'\t'+' '.join(hindisentence)+'\t'+' '.join(newenglishsentence)+'\t'+candidate+'\t'+str(score)+'\t'+str(score_in)+'\t'+str(score_out)+'\t'+str(rs1)+"\t"+str(rs2)+"\t"+str(k)+"\t"+str(j)+'\n')
			csvfile.write("\n")



def getsegment(segment_file):
    i = 0
    map_dict = defaultdict()
    listsegment = []
    with codecs.open(segment_file, 'r', 'utf-8') as f:
        lines = f.readlines()
        for line in lines[483:]:
                segment_sr = ast.literal_eval(line.strip())
                listsegment.append(segment_sr)
    return listsegment

def getsentences(filename):
        f = codecs.open(filename, 'r', 'utf-8')
        srcsenlist = []
        tgtsenlist = []
        indexlist = []

        for line in f.readlines():
                tokens = line.split("|||")
                indexlist.append(int(tokens[0].split("\t")[0]))
                src = tokens[0].split("\t")[1]
                tgt = tokens[1].strip()
                srcsenlist.append(src.strip())
                tgtsenlist.append(tgt.strip())
        return (srcsenlist, tgtsenlist, indexlist)

def getlinedict(filename, lineindex):
   f = codecs.open(filename, 'r', 'utf-8')
   lines = f.read().split("\n")
   map_ = defaultdict()
   for i in lineindex:
          map_dict = defaultdict()
          engsentence = lines[3 * (i) + 1].strip()
          hinditokens = lines[3 * (i) + 2].split()
          indexlist = []
          key = ''
          hiindex = -1
          for j in range(len(hinditokens)):
            token = hinditokens[j]
            if token.isdigit() and (hinditokens[j+1]!="({"):
              indexlist.append(int(token)-1)
            else:
              if token == "})":
                  if key != "NULL":
                    map_dict[hiindex] = indexlist
                  indexlist = []
              elif(token == "({"):
                continue
              else:
                key = token
                if token != "NULL":
                  hiindex += 1
          map_[i] = map_dict
   return map_

if __name__=="__main__":

      hparams = parsearguments()
      
      sentence = hparams.sentence_prefix
      idf_dict = get_idfdict(sentence + "." + hparams.src)
      idf_dict_hi = get_idfdict(sentence + "." + hparams.tgt)
      #print("IDF dict loading done")
      dict_file = hparams.embedding_prefix

      probdict = get_probdict(hparams.map_file)
      (endict, hidict) = load_embeddings(dict_file + '.' + hparams.src, dict_file+'.' + hparams.tgt)
      
      #print("Embedding loding done")      
      englist, hilist, indexlist = getsentences(hparams.sample_sent)
      listsegment = getsegment(hparams.segment_file)

      sentence_dict = getlinedict(hparams.giza_prefix, indexlist)

      count = 0
      i = 0  
      sentenceprev = ''
      
      for (segments, ensen, hisen, index) in zip(listsegment, englist, hilist, indexlist):
                print(segments, ensen, hisen, index, sentence_dict[index], sentence_dict[index])
                getembeddings(segments, ensen.split(), hisen.split(), sentence_dict[index], probdict, idf_dict, idf_dict_hi, endict, hidict, hparams.output_file, index)
		count +=1

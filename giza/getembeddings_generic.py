import sys
import codecs
import argparse
import operator
from collections import defaultdict
import re
import ast
import copy
import math 

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
        parser.add_argument('--sentence', type=str, default='english_senetence.txt',
                        help='file containing the sentences')	
        parser.add_argument('--sample_sent', type=str, default='sample.txt',
                        help='file containing the sampled parallel corpus')	
	parser.add_argument('--output_file', type=str, default='output.txt',
                        help='output file to store results')
	hparams = parser.parse_args()
        
	return hparams


def balanced_sigmoid(score, lang):
	return ( 1 / (1 + math.exp(-score)))

def get_idfdict(filename):
        freq_dict = defaultdict(int)
        idf_dict = defaultdict(float)
        with codecs.open(filename, 'r', 'utf-8') as f:
        #with codecs.open(filename, encoding="utf-8") as f:
                lines = f.readlines()
        count = 0.0
        for line in lines:
                count += 1
                words = line.replace(' \'', '\'').split()
                for word in set(words):
                        freq_dict[word] += 1
        for word in freq_dict.keys():
                idf_dict[word] = math.log(count / freq_dict[word])

        return idf_dict

def get_probdict(filename):
  probdict = defaultdict(defaultdict)
  with codecs.open(filename, encoding="utf-8") as f:
      lines = f.readlines()
      for line in lines:
          tokens = line.split()
          #print("debug tokens", tokens[2])
	  if len(tokens) != 3:
		continue
	  probdict[tokens[0].replace('.', '').strip()][tokens[1].strip()] = float(tokens[2])
  return probdict

def get_stopword(filename):
  with codecs.open(filename, encoding="utf-8") as f:
    lines = f.readlines()
  return lines


def getembeddings(segments, engsentence, hindisentence, mapping, probdict, idf_dict, idf_dict_hi, output_file, index):
  
  enlen = len(engsentence)

  hindilen = len(hindisentence)
  newenglishsentence = copy.copy(engsentence)
  newhindisentence = copy.copy(hindisentence)

  for segment in segments:
	  segmentlist = segment.strip().split()	
	  random_seed1 = -1
	
	  count = 0
          try:
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
	  except:
		continue
	  random_seed2 = enlen - count 
          hindisegment = ''
          indexlist = []
          newenglishsentence = copy.copy(engsentence)
          newhindisentence = copy.copy(hindisentence)
	  segment_dict = defaultdict()
	  for l in range(1,hindilen-1):
                     for k in range(hindilen - l):
                        j = min(k + l -1, hindilen-1)

                        indexlist = []
                        count = 0
                        probmul = 1
                        overallenglist = []
                        score = 1.0
			eng_idf = 0.0
			hi_idf = 0.0
			for x in range(random_seed1, random_seed2+1):
				try:
					eng_idf += balanced_sigmoid(idf_dict[engsentence[x]], 'en')
				except:
					print "Error", x, segmentlist,engsentence
			for i in range(k,j+1):
                                try:
					hindiword = hindisentence[i]
                                	engindexlist = mapping[i]
                                	overallenglist.extend(engindexlist)
                                except:
					continue
				try:
                                        hi_idf += balanced_sigmoid(idf_dict_hi[hindiword], 'hi')
				except:
					hi_idf += 0.0
				

				scoresum = 0.0
				for x in range(random_seed1, random_seed2+1):
						if x in mapping[i]:
							scoresum += 1.0
						if hindiword in probdict[engsentence[x]].keys():
							scoresum +=  probdict[engsentence[x]][hindiword] * balanced_sigmoid(idf_dict[engsentence[x]], 'en')
				score *= scoresum * balanced_sigmoid(idf_dict_hi[hindisentence[i]], 'hi')
			#score *= 1.0/(abs(hi_idf-eng_idf)+0.00001)
			newenglishsentence[random_seed1:random_seed2+1] = [x.upper() for x in newenglishsentence[random_seed1: random_seed2+1]]
			newsentence = [' '.join(engsentence[0:max(0,random_seed1)]), ' '.join(hindisentence[k:j+1]), ' '.join(engsentence[random_seed2+1:])]
                        segment_dict[' '.join(newsentence)] = (score, accuracy, abs(hi_idf-eng_idf), random_seed1 - 1, enlen - random_seed2 - 1, k, j)
          newenglishsentence[random_seed1:random_seed2+1] = [x.upper() for x in newenglishsentence[random_seed1: random_seed2+1]]
          sorted_candidates = sorted(segment_dict.items(), key=operator.itemgetter(1,0), reverse = True)
	  with codecs.open(output_file,'a','utf-8') as csvfile:
                        for (candidate, (score, accuracy, extra, rs1, rs2, k , j)) in sorted_candidates[:1]:
                                csvfile.write(str(index) + '\t'+' '.join(hindisentence)+'\t'+' '.join(newenglishsentence)+'\t'+candidate+'\t'+str(score)+'\t'+str(accuracy)+'\t'+str(extra)+'\t'+str(rs1)+"\t"+str(rs2)+"\t"+str(k)+"\t"+str(j)+'\n')


def getsegment(segment_file):
    i = 0
    map_dict = defaultdict()
    listsegment = []
    with codecs.open(segment_file, 'r', 'utf-8') as f:
        lines = f.readlines()
	for line in lines:
        	segment_sr = ast.literal_eval(line.strip())
		#if len(segment_sr) > 3 and len(segment_sr) < 10:
		listsegment.append(segment_sr)
    return listsegment

def getsentences(filename):
        #print filename
	f = codecs.open(filename, 'r', 'utf-8')
        srcsenlist = []
        tgtsenlist = []
	indexlist = []

        for line in f.readlines():
		#print line
                tokens = line.split("|||")
		#print tokens
		indexlist.append(int(tokens[0].split("\t")[0]))
                src = tokens[0].split("\t")[1]
		#.strip().replace('\'', " \'").replace('.', ' .').replace('-LSB-','[').replace('-RSB-',']')
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
          #print("Debug eng",i, engsentence)
          hinditokens = lines[3 * (i) + 2].split()
          #print("Debug hin", hinditokens)
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
      idf_dict = get_idfdict(hparams.sentence+'.'+hparams.src)
      
      probdict = get_probdict(hparams.map_file)
      idf_dict_hi = get_idfdict(hparams.sentence+'.'+hparams.tgt)
      englist, hilist, indexlist = getsentences(hparams.sample_sent)
       
      listsegment = getsegment(hparams.segment_file)
      sentence_dict = getlinedict(hparams.giza_prefix, indexlist)

      count = 0
      i = 0
      for (segments, ensen, hisen, index) in zip(listsegment, englist, hilist, indexlist):
		getembeddings(segments, ensen.split(), hisen.split(), sentence_dict[index], probdict, idf_dict, idf_dict_hi, hparams.output_file, index)
		count +=1
	

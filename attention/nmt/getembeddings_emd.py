"""For training NMT models."""
from __future__ import print_function

import json
import operator
import collections
import math
import os
import random
import time
import numpy as np
import csv
import copy
import ast
import urllib2
import urllib
import re
import argparse
from collections import defaultdict
import codecs
import tensorflow as tf
from emd import emd

from . import attention_model
from . import gnmt_model
from . import inference
from . import model as nmt_model
from . import model_helper
from .utils import iterator_utils
from .utils import misc_utils as utils
from .utils import nmt_utils
from .utils import vocab_utils

class TrainModel(
    collections.namedtuple("TrainModel", ("graph", "model", "iterator",
                                          "skip_count_placeholder"))):
  pass


def create_model(
	model_creator, hparams, scope=None, single_cell_fn=None,
    model_device_fn=None):
  
  """Create train graph, model, and iterator."""
  src_file = "%s.%s" % (hparams.train_prefix, hparams.src)
  tgt_file = "%s.%s" % (hparams.train_prefix, hparams.tgt)
  src_vocab_file = hparams.src_vocab_file
  tgt_vocab_file = hparams.tgt_vocab_file

  graph = tf.Graph()

  with graph.as_default():
    src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(
        src_vocab_file, tgt_vocab_file, hparams.share_vocab)

    src_dataset = tf.contrib.data.TextLineDataset(src_file)
    tgt_dataset = tf.contrib.data.TextLineDataset(tgt_file)
    skip_count_placeholder = tf.placeholder(shape=(), dtype=tf.int64)
    

    iterator = iterator_utils.get_iterator(
        src_dataset,
        tgt_dataset,
        src_vocab_table,
        tgt_vocab_table,
        batch_size=hparams.batch_size,
        sos=hparams.sos,
        eos=hparams.eos,
        source_reverse=hparams.source_reverse,
        random_seed=hparams.random_seed,
        num_buckets=hparams.num_buckets,
        src_max_len=hparams.src_max_len,
        tgt_max_len=hparams.tgt_max_len,
        skip_count=skip_count_placeholder)

    # Note: One can set model_device_fn to
    # `tf.train.replica_device_setter(ps_tasks)` for distributed training.
    with tf.device(model_device_fn):
      model = model_creator(
          hparams,
          iterator=iterator,
          mode=tf.contrib.learn.ModeKeys.TRAIN,
          source_vocab_table=src_vocab_table,
          target_vocab_table=tgt_vocab_table,
          scope=scope,
          single_cell_fn=single_cell_fn)
  
  return TrainModel(
      graph=graph,
      model=model,
      iterator=iterator,
      skip_count_placeholder=skip_count_placeholder)
  #return (graph, model, iterator, skip_count_placeholder)

def load_embeddings(endictfile,hidictfile):
	
	endict = defaultdict()
	hidict = defaultdict()
        
	with codecs.open(endictfile,'r', 'utf-8') as f:
                for line in f:
			tokens = line.split()
			endict[tokens[0]] = [float(x) for x in tokens[-300:]]
        
	with codecs.open(hidictfile, encoding='utf-8') as f:
		for line in f:
			tokens = line.split()
			hidict[tokens[0]] = [float(x) for x in tokens[-300:]]
		
        return (endict,hidict)


def get_idfdict(filename):
        #'''
	freq_dict = defaultdict(int)
        #with open(filename) as f:
	with codecs.open(filename, encoding="utf-8") as f:
                lines = f.readlines()
        for line in lines:
                words = line.replace(' \'', '\'').split()
                for word in words:
                        freq_dict[word] += 1
        return freq_dict

def balanced_sigmoid(score, lang):
        return  (1 / (1 + math.exp(-score)))

def calculate_emd(hidict, endict, ensentence, hindisentence, probdict, idf_dict, idf_dict_hi, enidlist, hiidlist):
        x = []
        y = []
	xin = []
	yin = []
	en_idf = 0.0
	for en in enidlist:
		word = ensentence[en]
                if word == '\'re':
                        word ='are'
                if word not in ('!','.',':', ';', ','):
                        try:
                                x.append(endict[word])
				xin.append(en)
                        except:
                                continue
	hi_idf = 0.0
	for hi in hiidlist:
		word = hindisentence[hi]
                if word not in ('!','.',':', ';', ','):
                        try:
                                y.append(hidict[word])
				yin.append(hi)
                        except:
                                continue
	distance=np.zeros((len(xin), len(yin)))
	for en in range(len(xin)):
                for hi in range(len(yin)):
			try:
				distance[en][hi] = (1 - probdict[yin[hi],0][xin[en]]) 
                        except:
                                distance[en][hi] = 1
        distVal= 99
        
	if len(y) > 0 and len(x)> 0:
		distVal = emd(np.array(x),np.array(y), D=distance) 
        return distVal

def score(hindisentence, hindisegments, enlen):
        if len(hindisegments) == 0:
		return 0	
	discontinuous = [] 
	contiguous = []

	lencont = 0
	lendis = 0

        hilen = len(hindisentence)
	for i in range(hindisegments[0], hindisegments[-1]+1):
		if i in hindisegments:
			lencont += 1
			if lendis != 0:
				discontinuous.append(lendis)
				lendis = 0
		else:
			lendis += 1
			if lencont != 0:
				contiguous.append(lencont)
				lencont = 0 
	if lendis != 0:
        	discontinuous.append(lendis)
	if lencont != 0:
		contiguous.append(lencont)
         
	
	contiguous = [(x+0.0)/enlen for x in contiguous]
	denom = np.average(discontinuous) * len(discontinuous) / (hindisegments[-1] - hindisegments[0]+1) 
	if len(discontinuous) == 0:
		denom = 1
	sc = 1.0/denom
	return sc

	
def getembeddings(hparams,segments, engsentence, hindisentence, idf_dict, idf_dict_hi, endict, hidict, output_file, index, scope=None, target_session="", single_cell_fn=None):
  """Train a translation model."""
  log_device_placement = hparams.log_device_placement
  out_dir = hparams.out_dir
  num_train_steps = hparams.num_train_steps
  steps_per_stats = hparams.steps_per_stats
  steps_per_external_eval = hparams.steps_per_external_eval
  steps_per_eval = 10 * steps_per_stats
  if not steps_per_external_eval:
    steps_per_external_eval = 5 * steps_per_eval

  if not hparams.attention:
    model_creator = nmt_model.Model
  elif hparams.attention_architecture == "standard":
    model_creator = attention_model.AttentionModel
  elif hparams.attention_architecture in ["gnmt", "gnmt_v2"]:
    model_creator = gnmt_model.GNMTModel
  else:
    raise ValueError("Unknown model architecture")

  #graph, model, iterator, skip_count_placeholder = 
  train_model = create_model(model_creator, hparams, scope, single_cell_fn)
  model_dir = hparams.out_dir
  config_proto = utils.get_config_proto(log_device_placement=log_device_placement)

  sess = tf.Session(
      target=target_session, config=config_proto, graph=train_model.graph)
  with train_model.graph.as_default():
    loaded_model, global_step = model_helper.create_or_load_model(
       train_model.model, model_dir, sess, "train")
  sess.run(
      train_model.iterator.initializer,
      feed_dict={train_model.skip_count_placeholder: 0})
  step_result = loaded_model.getembeddings(sess)
  encoder_outputs, decoder_outputs, encoder_inputs, decoder_inputs, history = step_result
  enlen = len(engsentence)

  hindilen = len(hindisentence)
  #print("Hindi len", hindilen, history.shape)
  newenglishsentence = copy.copy(engsentence)
  newhindisentence = copy.copy(hindisentence)

  name = -1
  #segmentname = ['SR']
  for segment in segments:
                  segmentlist = segment.strip().split()
                  print(engsentence, segmentlist)
                  if len(segmentlist) == enlen:
                        continue

                  segmentlist = segment.strip().split()
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
                  print("count", count, enlen)
                  random_seed2 = enlen - count
                  #reverselist.index(segmentlist[-1]) - 1
                  print("random", random_seed1, random_seed2)

		  '''
		  random_seed1 = -1
		  random_seed2 = -1
		  try:
		  	random_seed1 = engsentence.index(segmentlist[0])
		  	random_seed2 = -1
		  	reverselist = copy.copy(engsentence)
                  	reverselist.reverse()
                  	random_seed2 = enlen - reverselist.index(segmentlist[-1]) - 1
		  except:
			continue
		  '''
		  #random_seed2 = random_seed1 + len(segmentlist) - 1	 			
		  mapping = ''
  		  hindisegment = ''
		  indexlist = []
		  newenglishsentence = copy.copy(engsentence)
		  newhindisentence = copy.copy(hindisentence)
		  segment_dict = defaultdict() 
                  for l in range(1,max(len(segmentlist),hindilen-1) + 1):
                     for k in range(hindilen - l):
		       try:
                        j = min(k + l -1, hindilen-1)
			mapping = ''
                  	indexlist = []
                  	newenglishsentence = copy.copy(engsentence)
                  	newhindisentence = copy.copy(hindisentence)
			count = 0
			probmul = 1
			overallenglist = []
			#enlen * 0.3
			sumscore = 0.0
			score = 1.0
                        #print("DebugKJ",k,j)
			emd_in = calculate_emd(hidict, endict, engsentence, hindisentence, history, idf_dict, idf_dict_hi,range(random_seed1, random_seed2+1), range(k,j+1))
                        engoutseg = engsentence[0:max(0,random_seed1)]+ engsentence[random_seed2+1:]
                        hioutseg = hindisentence[0:k]+ hindisentence[j+1:]
                        emd_out = calculate_emd(hidict, endict, engsentence, hindisentence, history, idf_dict, idf_dict_hi, range(0,max(0,random_seed1))+range(random_seed2+1,enlen), range(0,k)+range(j+1, hindilen))
                        #score = emd_in * min(abs((j+1.0-k)-(random_seed2+1 - random_seed1))/(random_seed2+1 - random_seed1),1.0)+ emd_out * min(abs((hindilen + 1.0-(j+1 -k)) - (enlen+1-(random_seed2 +1 - random_seed1)))/(enlen+1-(random_seed2 +1 - random_seed1)),1.0)
			score = emd_in + emd_out
			#score = emd_in *1.0/(j+1.0-k)
			#/(random_seed2+1 - random_seed1))
			#score +=emd_out * 1.0 /(hindilen + 1.0-(j+1 -k))
			#/(enlen+1-(random_seed2 +1 - random_seed1)))
                        newsentence = [' '.join(engsentence[0:max(0,random_seed1)]), ' '.join(hindisentence[k:j+1]), ' '.join(engsentence[random_seed2+1:])]
                        #segment_dict[str(k)+_+] = (score, emd_in, emd_out, random_seed1 - 1, enlen - random_seed2 - 1, ' '.join(newsentence))
			segment_dict[' '.join(newsentence)] = (score, emd_in, emd_out, random_seed1 - 1, enlen - random_seed2 - 1, k, j)
		       except:
				continue
		  newenglishsentence[random_seed1:random_seed2+1] = [x.upper() for x in newenglishsentence[random_seed1: random_seed2+1]]
          	  sorted_candidates = sorted(segment_dict.items(), key=operator.itemgetter(1,0))
		  with codecs.open(output_file,'a','utf-8') as csvfile:
                        for (candidate, (score, score_in, score_out, rs1, rs2, k, j)) in sorted_candidates[:1]:
                                csvfile.write(str(index) + '\t'+' '.join(hindisentence)+'\t'+' '.join(newenglishsentence)+'\t'+candidate+'\t'+str(score)+'\t'+str(score_in)+'\t'+str(score_out)+'\t'+str(rs1)+'\t'+str(rs2)+'\t'+str(k)+'\t'+str(j)+'\n')
                        #csvfile.write("\n")
		  #break

def getsegment(segment_file, englist):
    map_dict = defaultdict()
    listenglish = []
    listhindi = []
    listsegment = []
    englishlist = []
    hindilist = []
    with open(segment_file, 'r') as f:
        lines = f.readlines()[483:]
        for (line, engsent) in zip(lines, englist):
                tokens = line.strip()
                segment_sr = ast.literal_eval(tokens)
                list_temp = []
                englishwords = engsent.split()
                for seg in segment_sr:
                        list_temp.append(seg)
                list_temp = list_temp[:min(10, len(list_temp))]

                listsegment.append(list_temp)
    return listsegment

def getsentences(filename):
        f = codecs.open(filename, encoding='utf-8')
        srcsenlist = []
        tgtsenlist = []
	indexlist = []
        for line in f.readlines():
                tokens = line.split("|||")
		indexlist.append(int(tokens[0].split("\t")[0]))
                src = tokens[0].split("\t")[1]
		#src = tokens[0].replace('\'', " \'").replace('.', ' .').replace('-LSB-','[').replace('-RSB-',']')
		#src = tokens[0].split("\t")[1].strip().replace('\'', " \'").replace('.', ' .').replace('-LSB-','[').replace('-RSB-',']')
                tgt = tokens[1].strip()
                srcsenlist.append(src.strip())
                tgtsenlist.append(tgt.strip())
        return (srcsenlist, tgtsenlist, indexlist)


def parsearguments():
        parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--segment_file', type=str, default='segment.txt',
                        help='file to store the segmentation information genearted by a grammar')
        parser.add_argument('--model_dir', type=str, default='model/',
                        help='directory which has the trained model')
        parser.add_argument('--sentence_prefix', type=str, default='sentence',
                        help='file containing the corpus')
        parser.add_argument('--sample_sent', type=str, default='sample.txt',
                        help='file containing the corpus')
        parser.add_argument('--output_file', type=str, default='sample.txt',
                        help='output to be stored')
        parser.add_argument('--embedding_prefix', type=str, default='embedding',
                        help='multilingual word embedding')
	hparams = parser.parse_args()
        return hparams

#def main_embeddings(hparams, scope=None, target_session="", single_cell_fn=None):
if __name__=="__main__":
      hparams_local = parsearguments()
      out_dir = hparams_local.model_dir
      hparams = utils.load_hparams(out_dir)
      hparams.src_max_len = 300
      hparams.tgt_max_len = 300
      hparams.num_embeddings_partitions = 0
      hparams.get_embeddings = True
      sentence = hparams_local.sentence_prefix
      idf_dict = get_idfdict(sentence + "." + hparams.src)
      idf_dict_hi = get_idfdict(sentence + "." + hparams.tgt)


      dict_file = hparams_local.embedding_prefix
      (endict, hidict) = load_embeddings(dict_file+"."+hparams.src, dict_file+"."+hparams.tgt) 
      #print('endict', endict['the']) 
      hparams.train_prefix = hparams.train_prefix.replace("train", "test")
      
      segment_file = hparams_local.segment_file
      (englist, hilist, indexlist) = getsentences(hparams_local.sample_sent)
      listsegment = getsegment(segment_file, englist)
      output_file = hparams_local.output_file
      count = 0
      i = 0
      for (segments, engsentence, hindisentence, index) in zip(listsegment, englist, hilist, indexlist):
		with codecs.open(hparams.train_prefix+"."+hparams.src, 'w', encoding='utf-8') as f:
                        f.write(engsentence+'\n')
                with codecs.open(hparams.train_prefix+"."+hparams.tgt, 'w', encoding='utf-8') as f:
                        f.write(hindisentence+'\n')
		getembeddings(hparams, segments, engsentence.split(), hindisentence.split(), idf_dict, idf_dict_hi, endict, hidict, hparams_local.output_file, index)

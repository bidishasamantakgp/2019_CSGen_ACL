"""For training NMT models."""
from __future__ import print_function

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

def balanced_sigmoid(score, lang):
	val = (1 / (1 + math.exp(-score)))
	val = (val - 0.5) /(0.5)
	return val
	
def get_idfdict(filename):
        freq_dict = defaultdict(int)
	idf_dict = defaultdict(float)
        with open(filename) as f:
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

def score(hindisentence, hindisegments, enlen):
	#engsentecne -> list of tokens in engsentence
	#engsegments -> list of lists for contiguous segments
	#hindisentence -> list of tokens in hindi sentence
	#hindisegments -> index for segments
        if len(hindisegments) == 0:
		return 0	
	discontinuous = [] 
	contiguous = []

	lencont = 0
	lendis = 0

        hilen = len(hindisentence)
        #print(hilen, hindisegments[-1])
	#for i in range(hilen):
	for i in range(hindisegments[0], hindisegments[-1]+1):
	        #print(i)
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
         
        #print("hindi segments", hindisegments)
	#print("len segments", hilen, contiguous, discontinuous)
	
	contiguous = [(x+0.0)/enlen for x in contiguous]
	#discontinuous = [(x+0.0)/hilen for x in discontinuous]
	denom = np.average(discontinuous) * len(discontinuous) / (hindisegments[-1] - hindisegments[0]+1) 
	if len(discontinuous) == 0:
		denom = 1
	#sc = (np.sum(contiguous) * 1.0) / denom
	sc = 1.0/denom
	#print(contiguous, np.average(discontinuous), sc)
	return sc

	
def getembeddings(hparams,segments, engsentence, hindisentence, idf_dict, idf_dict_hi, output_file, index, scope=None, target_session="", single_cell_fn=None):
#def getembeddings(hparams, scope=None, target_session="", single_cell_fn=None):
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
  #print("iterator soucre", iterator.source.eval(session=sess), iterator.source.shape) 
  step_result = loaded_model.getembeddings(sess)
  encoder_outputs, decoder_outputs, encoder_inputs, decoder_inputs, history = step_result
  print("encoder input shape",encoder_inputs.shape, encoder_inputs[0][0])
  print("encoder output shape",encoder_outputs.shape, encoder_outputs[0])
  #print(encoder_inputs)
  print("decoder input shape",decoder_inputs.shape, decoder_inputs[0][0])
  #print(decoder_inputs)
  print("decoder_outputs_shape",decoder_outputs.rnn_output.shape)
  print("history_shape",history.shape, history[0])
  
  enlen = len(engsentence)
  hindilen = len(hindisentence)
  newenglishsentence = copy.copy(engsentence)
  newhindisentence = copy.copy(hindisentence)
  
  name = -1
  
  for segment in segments:
		  #segmentlist = segment.replace('-LSB-','[').replace('-RSB-',']').strip().split()
		  segmentlist = segment.strip().split()
		  print(engsentence, segmentlist)
		  if len(segmentlist) == enlen:
			continue
		  
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
		  print("count", count, enlen)
          	  random_seed2 = enlen - count
          	  #reverselist.index(segmentlist[-1]) - 1
          	  print("random", random_seed1, random_seed2)

		  '''
		  '''
			 			
		  #print("DEBUG", segmentlist, engsentence, random_seed1, random_seed2)
		  mapping = ''
  		  hindisegment = ''
		  indexlist = []
		  newenglishsentence = copy.copy(engsentence)
		  newhindisentence = copy.copy(hindisentence)
		  segment_dict = defaultdict() 
                  for l in range(1,hindilen-1):
		  #for l in range(1,max(len(segmentlist),hindilen-1) ):
                     for k in range(hindilen - l):
                  #for l in range(5):
                  #   for k in range(5,7):
     
		       try: 
			j = min(k + l -1, hindilen-1) 
			mapping = ''
                  	#hindisegment = ''
                  	indexlist = []
                  	newenglishsentence = copy.copy(engsentence)
                  	newhindisentence = copy.copy(hindisentence)
			count = 0
			probmul = 1
			overallenglist = []
			#enlen * 0.3
			sumscore = 0.0
			score = 1.0
			eng_idf_score = 0.0
                        for enindex in range(random_seed1, random_seed2+1):
				eng_idf_score += balanced_sigmoid(idf_dict[engsentence[enindex]], 'en')

			hi_idf_score = 0.0
			#print("Debug index", k, j+1, hindilen, hindisentence)
			#for i in range(5,10):
			for i in range(k,j+1):
				
                        	hindiword = hindisentence[i]
                        	engindexlist = history[i,0].argsort()[-int(enlen * 0.3):]
				engindexlist = history[i,0].argsort()[-1]
				#overallenglist.extend(engindexlist)
				overallenglist.append(engindexlist)
				#sumscore += history[i,0][engindex]
				try:
					hi_idf_score += balanced_sigmoid(idf_dict_hi[hindisentence[i]], 'hi')
					#hi_idf_score += idf_dict_hi[hindisentence[i]]
					#hi_idf_score += 1.0/ (idf_dict_hi[hindisentence[i]]+1)
				except:
					hi_idf_score += 0.0

				sumscore = 0.0
                        	for enindex in range(random_seed1, random_seed2+1):
					#print("Debug:",enindex, engsentence[enindex], engsentence)
					try:
						sumscore += history[i,0][enindex] * balanced_sigmoid(idf_dict[engsentence[enindex]], 'en')  
					except:
						sumscore += history[i+1,0][enindex]
				score *= sumscore * balanced_sigmoid(idf_dict_hi[hindisentence[i]], 'hi')
                        
			#score  = score / abs(len(segmentlist) - (k+1 - j)+0.1)
			print("Debug acc", score, k,j+1,hi_idf_score, eng_idf_score)
			#score = score / abs(hi_idf_score - eng_idf_score+ 0.00001)
			#print("DEBUG ACC", TP, FP, FN, overallenglist, random_seed1, random_seed2, segmentlist, engsentence)
                        #score *= accuracy
		  	newsentence = [' '.join(engsentence[0:max(0,random_seed1)]), ' '.join(hindisentence[k:j+1]), ' '.join(engsentence[random_seed2+1:])]
                  	segment_dict[' '.join(newsentence)] = (score, accuracy, overallenglist, random_seed1 - 1, enlen - random_seed2 - 1, k, j)	
		       except:
				continue
                  newenglishsentence[random_seed1:random_seed2+1] = [x.upper() for x in newenglishsentence[random_seed1: random_seed2+1]]
		  newenglishsentence[random_seed1:random_seed2+1] = [x.upper() for x in newenglishsentence[random_seed1: random_seed2+1]]
          	  sorted_candidates = sorted(segment_dict.items(), key=operator.itemgetter(1,0), reverse=True)
		  with open(output_file,'a') as csvfile:
                        for (candidate, (score, accuracy, overlist, rs1, rs2, k, j)) in sorted_candidates[:1]:
                                csvfile.write(str(index)+'\t'+' '.join(hindisentence)+'\t'+' '.join(newenglishsentence)+'\t'+candidate+'\t'+str(score)+'\t'+str(rs1)+'\t'+str(rs2)+'\t'+str(k)+'\t'+str(j)+'\n')
                        #csvfile.write("\n")

def getsegment(segment_file, englist):
    map_dict = defaultdict()
    listenglish = []
    listhindi = []
    listsegment = []
    englishlist = []
    hindilist = []
    with open(segment_file, 'r') as f:
        lines = f.readlines()
        for (line, engsent) in zip(lines, englist):
		tokens = line.strip()
		segment_sr = ast.literal_eval(tokens)
		list_temp = []
		englishwords = engsent.split()
		for seg in segment_sr:
			#print("Debug seg", len(seg.split()), len(englishwords), len(englishwords)*1.0/4, len(englishwords)*1.0/3)
			#if (len(seg.split()) <= len(englishwords)*1.0/2 ) and (len(seg.split())>= len(englishwords)*1.0/4) :
                	list_temp.append(seg)	
		#list_temp = list_temp[:min(10, len(list_temp))]	
		
		listsegment.append(list_temp)
    return listsegment

def getsentences(filename):
	f = open(filename)
	srcsenlist = []
	tgtsenlist = []
	indexlist = []
	for line in f.readlines():
		tokens = line.split("|||")
		indexlist.append(int(tokens[0].split("\t")[0]))
                src = tokens[0].split("\t")[1]
		src = tokens[0].split("\t")[1].strip().replace('\'', " \'").replace('.', ' .').replace('-LSB-','[').replace('-RSB-',']')
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
        parser.add_argument('--output_file', type=str, default='output.txt',
                        help='output to be stored')
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

      hparams.train_prefix = hparams.train_prefix.replace("train", "test")
      segment_file = hparams_local.segment_file

      (englist, hilist, indexlist) = getsentences(hparams_local.sample_sent)
      listsegment = getsegment(segment_file, englist)
      output_file = hparams_local.output_file

      count = 0
      i = 0
      #for (segments, engsentence, hindisentence, index) in zip(listsegment[12:], englist[12:], hilist[12:], indexlist[12:]):
      for (segments, engsentence, hindisentence, index) in zip(listsegment, englist, hilist, indexlist):
		count += 1
		#print("Segments", segments)
		with open(hparams.train_prefix+"."+hparams.src, 'w') as f:
                        f.write(engsentence+'\n')
                with open(hparams.train_prefix+"."+hparams.tgt, 'w') as f:
                        f.write(hindisentence+'\n')
                getembeddings(hparams, segments, engsentence.split(), hindisentence.split(), idf_dict, idf_dict_hi, output_file, index)

from transliterate_util import * 
import urllib2
import urllib
import sys
import json
import ast
import re
import time
import argparse
#import enchant
reload(sys)
sys.setdefaultencoding('utf8')

def parseargument():
 	parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--source_lang', type=str, default='en',
                        help='source language hi, en')
        parser.add_argument('--target_lang', type=str, default='hi',
                        help='target language to be converted en, hi')
	parser.add_argument('--file_name', type=str, default='hi',
                        help='file to be converted en, hi')
        parser.add_argument('--out_file', type=str, default='out',
                        help='file to be written')
	args = parser.parse_args()
	return args

if __name__=="__main__":
	args = parseargument()
	#d = enchant.Dict("en_US")
	#fw = codecs.open(args.out_file, 'a', 'utf-8')
	fw = open(args.out_file, 'w')

	f = codecs.open(args.file_name, 'r', 'utf-8')
	#f = open(args.file_name, 'r')
	for line in f:
		line = line.strip()
		if len(line) == 0:
			fw.write("\n")
			continue
		#words = line.split()
		#transword = word
		
		#data = ''
		#for word in words:
		#		if not d.check(word):
		#		data += ' '+word
		
		#if not d.check(word):
		[sentence, tag] = line.strip().split("\t")
		args.data = sentence.replace(',', '')
		#.encode('utf-8')
                #print hparams.data
                trans_list = transliterate(args)
                #print trans_list
		if len(trans_list) > 0:
                     transliterated = trans_list[0]
                else:
                     transliterated = sentence #.decode('utf-8').strip()
		#try:
		#transliterated = transliterated.decode('utf-8').strip()
		print transliterated+"\t"+tag
		#fw.write(transliterated.strip()+"\t"+tag+"\n")
		#except:
		#	fw.write("\n")
		

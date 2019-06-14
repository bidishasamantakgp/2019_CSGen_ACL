import urllib2
import urllib
import sys
import json
import ast
import re
import time
import argparse 
import unicodedata
import codecs

def transliterate(args):
	proxy = urllib2.ProxyHandler({"https": "https://172.16.2.30:8080/", "http": "http://172.16.2.30:8080/"})
        opener = urllib2.build_opener(proxy)
        urllib2.install_opener(opener)
        user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
        headers={'User-Agent':user_agent,}

	translatedList = args.data.replace(',','').split()
	#for x in translatedList: print x
	transliterated = []
	i = 0
	count = 0
	while i < len(translatedList):
		translated = translatedList[i : 30 + i]
		i = i + 30
		#print len(translated[0])
		translated = ' '.join([word for word in translated if len(word)>0])
        	#print translated
        	url = "http://www.google.com/transliterate?langpair="+args.target_lang+"|"+args.source_lang+"&text="+urllib.quote(translated.encode('utf8'))
        	#print url
		#request=urllib2.Request(url,None,headers)
		#response = urllib2.urlopen(request)
        	request = ''
        	response = ''
        	try:
                	request=urllib2.Request(url,None,headers)
                	response = urllib2.urlopen(request)
        	except:
                	#print 'sleeping'
                	time.sleep(30)
			#i = i - 30
			#count = count + 1
			if count < 6:
				i = i-30
				count = count + 1
				#continue
			else:
				count = 0
			continue
                	#request=urllib2.Request(url,None,headers)
                	#response = urllib2.urlopen(request)
		count = 0
		data = response.read()
		
                if len(data) == 0:
			return
		try:
        		#print url, translated, data
        		formatted = ast.literal_eval(data)
			if len(formatted) > 0:
        			transliterated += formatted[0]['hws']
		except:
			continue
	return transliterated



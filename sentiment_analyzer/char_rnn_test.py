from keras.models import model_from_json
from keras.utils import np_utils
import numpy as np
import h5py
import pickle
from copy import deepcopy
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score, precision_recall_fscore_support
import sys

Masterdir = './'
Datadir = 'data/'
Modeldir = 'models/'
Featuredir = 'features/'

batch_size = 128
numclasses = 3

def accuracy(original, predicted):
	print("F1 score is: " + str(f1_score(original, predicted, average='macro')))
	print(precision_recall_fscore_support(original, predicted, average='micro'))
        target_names = ['negative', 'neutral', 'positive']
        print(classification_report(original, predicted, target_names=target_names))
        #classification_report()
        scores = confusion_matrix(original, predicted)
	print scores
	print np.trace(scores)/float(np.sum(scores))

h5f = h5py.File(Masterdir+Datadir+'Xtest_'+experiment_details+'.h5','r')
X_test = h5f['dataset'][:]
h5f.close()
print(X_test.shape)

inp = open(Masterdir+Datadir+'Ytest_'+experiment_details+'.pkl', 'rb')
y_test=pickle.load(inp)
inp.close()
y_test=np.asarray(y_test).flatten()
y_test2 = np_utils.to_categorical(y_test, numclasses) 
print(y_test.shape)
f = open(Masterdir+Modeldir+experiment_details+'_architecture.json','r+')
json_string = f.read()
f.close()
model = model_from_json(json_string)

model.load_weights(Masterdir+Modeldir+experiment_details+'_weights.h5')
model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['accuracy'])
	

score, acc = model.evaluate(X_test, y_test2, batch_size=batch_size)

y_pred = model.predict_classes(X_test, batch_size=batch_size)
num2char = {0: ' ', 1: 'a', 2: 'c', 3: 'b', 4: 'e', 5: 'd', 6: 'g', 7: 'f', 8: 'i', 9: 'h', 10: 'k', 11: 'j', 12: 'm', 13: 'l', 14: 'o', 15: 'n', 16: 'q', 17: 'p', 18: 's', 19: 'r', 20: 'u', 21: 't', 22: 'w', 23: 'v', 24: 'y', 25: 'x', 26: 'z'}
#print(X_test, y_pred)
'''

'''
accuracy(y_test,y_pred)

print('Accuracy is: '+str(acc))

h5f = h5py.File(Masterdir+Datadir+'Xtrain_'+experiment_details+'.h5','r')
X_train = h5f['dataset'][:]
h5f.close()
'''

'''

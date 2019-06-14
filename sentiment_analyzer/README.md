# Steps to train nmt and run attention based model on your data

## Train nmt with parallel corpus

`git clone https://github.com/drimpossible/Sub-word-LSTM.git'


## Data

`mkdir data/ model/ features/`
 

## Train model
` cp char_rnn_train.py Sub-word-LSTM'
` THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32, allow_gc=False python char_rnn_train.py`

` cp ordinal_categorical_crossentropy.py char_rnn_train_OCC.py Sub-word-LSTM`

` THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32, allow_gc=False python char_rnn_train.py`


## Test model

` python char_rnn_test.py`

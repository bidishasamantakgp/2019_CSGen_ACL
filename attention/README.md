# Steps to train nmt and run attention based model on your data

## Train nmt with parallel corpus

- python -m nmt.nmt     --attention=scaled_luong     --src=en --tgt=hi     --vocab_prefix=data/vocab/vocab      --train_prefix=data/train     --dev_prefix=data/test      --test_prefix=data/test  --out_dir=data/nmt_model     --num_train_steps=150000     --steps_per_stats=100     --num_layers=4     --num_units=300     --dropout=0.2     --metrics=bleu --encoder_type=bi


## Data

- mkdir data/
 

## Generate synthetic text similarity based

- python -m nmt.getembeddings_generic --segment_file ../data/segments.txt --model_dir data/nmt_model/ --sentence_prefix ../data/train --sample_sent ../data/sample.txt --output_file data/<output>.tsv 

## Generate synthetic text similarity based

-  python -m nmt.getembeddings_generic --segment_file ../data/segments.txt --model_dir data/nmt_model/ --sentence_prefix ../data/train --sample_sent ../data/sample.txt --output_file data/<output>.tsv --embedding_prefix ../data/embedding/<embedding_file>


# Steps to install and run giza on your data

## Data

`mkdir data/source data/target'
 
## Follow the steps to train GIZA on your data

- `https://okapiframework.org/wiki/index.php/GIZA++_Installation_and_Running_Tutorial'

## Generate synthetic text similarity based
`python getembeddings_generic.py --giza_prefix data/<prefix>.A3.final --segment_file data/segments.txt --map_file data/<prefix>.ti.final --sentence ../data/train --sample_sent ../data/sample.txt --output_file data/<output>.tsv'

## Generate synthetic text similarity based
`python getembeddings_emd.py --giza_prefix data/<prefix>.A3.final --segment_file data/segments.txt --map_file data/<prefix>.ti.final --sentence ../data/train --sample_sent ../data/sample.txt --embedding_prefix ../data/embedding/<embedding_file>  --output_file data/<output>.tsv'

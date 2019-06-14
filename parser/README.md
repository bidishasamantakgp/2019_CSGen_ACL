# Steps to  generate segments

## Generate segment file

`java -cp ".\*"  edu.stanford.nlp.parser.shiftreduce.demo.ShiftReduceDemo -model edu/stanford/nlp/models/srparser/englishSR.ser.gz > parsed_tree.txt`

`python parsefile.py parsed_tree.txt ../data/segment.txt`



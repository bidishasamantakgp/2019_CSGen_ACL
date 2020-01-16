# Steps to install and run giza on your data
# Requirements:
	- `Install Pyemd by pip install -e git+https://github.com/garydoranjr/pyemd.git#egg=pyemd,follow https://github.com/garydoranjr/pyemd`
	- `Download embedding word vector file for english nad hindi from https://fasttext.cc/docs/en/aligned-vectors.html`

## Note:
	- Here we use source language as english and target language as Hindi
## Steps to run the GIZA:
###  GIZA++ Installation and Running Tutorial:
- Follow https://okapiframework.org/wiki/index.php/GIZA++_Installation_and_Running_Tutorial
- Generate vcb (vocabulary) files and snt (sentence) files, containing the list of vocabulary and aligned sentences, respectively.
- You can find an good explaination at https://masatohagiwara.net/using-giza-to-obtain-word-alignment-between-bilingual-sentences.html
- Generate [prefix].A3.final and [prefix].ti.final, which contain the actual Viterbi alignment and the lexical translation table, respectively.
- You can find an good explaination at https://masatohagiwara.net/using-giza-to-obtain-word-alignment-between-bilingual-sentences.html



### Steps to generate the segment file:

#### 1. Download the below Jar files
	        1. stanford-corenlp-3.7.0.jar 
         2. stanford-english-corenlp-models.jar
         3. stanford-srparser-2014-10-23-models.jar

#### 2. Run the command 
- java -cp "./*"  edu.stanford.nlp.parser.shiftreduce.demo.ShiftReduceDemo -model edu/stanford/nlp/models/srparser/englishSR.ser.gz >> `parsed tree file name` to generate the parse tree of your source language dataset.
   - Note: Here Source language dataset file path is hardcoded in the java programe to /tmp/english.txt, although you can change it at your convinience.

#### 3. After that run the python file parsefiles.py as 
    - python parsefiles.py <parsed tree txt file generated in step 2> <sentence segment file name>

### Need to create a file containing source and target sentence pair:
	- Format <Line no.> <tab> <Source sentence><|||><Target sentence>, 
	- Ex. 12	Browse the various methods of the current accessible|||इस समय जिसे प्राप्त किया गया हो, उसकी विभिन्न विधियों (मेथड) में विचरण करें

### Finally run the below command to Generate synthetic text based on similarity:
	-  python getembeddings_generic.py --giza_prefix <path to the [prefix].A3.final> --segment_file <path to the segment file> --map_file <path to the [prefix].ti.final file> --sentence <path to the source and target sentence corpus> --sample_sent <path to the source target sentence pair file> --output_file <output file name>
	-  Note: --sentence <path to the source and target sentence corpus> for this put your source and target file name same with different language id extension. Ex. if you two source and language file in data folder with name train_data_en_hi.en and train_data_en_hi.hi then in the command put --sentence ./data/train_data_en_hi 

### Run the below command to Generate synthetic text based on dissimilarity:
	- python getembeddings_generic.py --giza_prefix <path to the [prefix].A3.final> --segment_file <path to the segment file> --map_file <path to the [prefix].ti.final file> --sentence <path to the source and target sentence corpus> --sample_sent <path to the source target sentence pair file> --output_file <output file name> --embedding_prefix <path to the source and target language embedding file>
	- Note: --embedding_prefix <path to the source and target language embedding file> for this put your source and target file name same with different language id extension. Ex. if you two source and language file in data folder with name embedding_en_hi.en and embedding_en_hi.hi then in the command put --embedding_prefix ./embedding/embedding_en_hi

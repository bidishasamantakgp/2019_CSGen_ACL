# Improved Sentiment Detection via Label Transfer from Monolingual to Synthetic Code-Switched Text
Paper : https://arxiv.org/abs/1906.05725

## Required packages
- `tensorflow 1.13`

## Dataset
- Please contact the authors

## Quick Start
The following directories provides scripts for generating synthetic texts:

- `parser` contains code to identify a segment for a given sentence

- `attention` contains code for attention score based synthetic data generation

- `giza` contains code for giza score based synthetic data generation

- `alignsentiment` contains code to transfer sentiment score from the monolingual corpus to the generated corpus

The following directory provides scripts for sentiment analysis:

- `sentiment_analyzer` Code to run sentiment classifier

## Contact
bidisha@iitkgp.ac.in

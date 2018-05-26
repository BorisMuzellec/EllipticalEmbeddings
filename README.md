# Elliptical Embeddings

This repository contains Python code for computing embeddings in the Wasserstein space of elliptical distributions, as in

>[Boris Muzellec and Marco Cuturi, *Generalizing Point Embeddings using the Wasserstein Space of Elliptical Distributions*](https://arxiv.org/abs/1805.07594)

While the code it contains is functional and allows to reproduce results form the paper, it is still under the process of being refactored. A final version will be made available shortly.

## Dependencies

`python 2.7.5, cupy, cython `

## Training Data

The skipgram model presented in the paper was trained on a concatenation of ukWaC and WaCkypedia_EN, both of which can be requested [here](http://wacky.sslmit.unibo.it/doku.php?id=download). 

We use all words appearing more than 100 times after tokenization (this is customisable).

The wordnet dataset can be obtained from `nltk`. The extraction of the hypernymy transitive closure can be performed using code from the Poincaré embeddings[repository](https://github.com/facebookresearch/poincare-embeddings/tree/master/wordnet).

## Usage

Prior to training your first embeddings, it is necessary to compile the cython files. You can do this by running the following command:

`python setup.py build_ext --inplace`
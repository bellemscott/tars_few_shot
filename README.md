# Information Extraction Final Project
Author: Belle Scott

## Description
This project implements a few-shot TARS model, which is part of the Flair framework (find documentation here - https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_10_TRAINING_ZERO_SHOT_MODEL.md)

Datasets used in this project:
TREC 6 - https://emilhvitfeldt.github.io/textdata/reference/dataset_trec.html
SQuAD - https://rajpurkar.github.io/SQuAD-explorer/

## Instructions to run
To train the model with TREC, run `model.py`
To predict on the stanford dataset, run `predict.py`


Citations
@inproceedings{halder2020coling,
  title={Task Aware Representation of Sentences for Generic Text Classification},
  author={Halder, Kishaloy and Akbik, Alan and Krapac, Josip and Vollgraf, Roland},
  booktitle = {{COLING} 2020, 28th International Conference on Computational Linguistics},
  year      = {2020}
}
@inproceedings{akbik2019flair,
  title={FLAIR: An easy-to-use framework for state-of-the-art NLP},
  author={Akbik, Alan and Bergmann, Tanja and Blythe, Duncan and Rasul, Kashif and Schweter, Stefan and Vollgraf, Roland},
  booktitle={{NAACL} 2019, 2019 Annual Conference of the North American Chapter of the Association for Computational Linguistics (Demonstrations)},
  pages={54--59},
  year={2019}
}

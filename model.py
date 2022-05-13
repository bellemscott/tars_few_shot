from flair.data import Corpus
from flair.datasets import TREC_6
from flair.models import TARSClassifier
from flair.trainers import ModelTrainer

# Define label names
label_name_map = {'ENTY': 'question about entity',
                  'DESC': 'question about description',
                  'ABBR': 'question about abbreviation',
                  'HUM': 'question about person',
                  'NUM': 'question about number',
                  'LOC': 'question about location'
                  }

# Grab the corpus
corpus: Corpus = TREC_6(label_name_map=label_name_map)

# Type of prediction
label_type = 'question_class'

# Make a label dictionary
label_dict = corpus.make_label_dictionary(label_type=label_type)

# Load off base model
tars = TARSClassifier.load("tars-base")

# Switch to a new task (TARS can do multiple tasks so you must define one)
tars.add_and_switch_to_new_task(task_name="question classification",
                                label_dictionary=label_dict,
                                label_type=label_type,
                                )

# 7. initialize the text classifier trainer
trainer = ModelTrainer(tars, corpus)

# 8. start the training
trainer.train(base_path='resources/taggers/trec',  # path to store the model artifacts
              learning_rate=0.02,  # use very small learning rate
              mini_batch_size=16,
              mini_batch_chunk_size=4,  # optionally set this if transformer is too much for your machine
              max_epochs=5,  # terminate after 10 epochs
              )
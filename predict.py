from flair.models import TARSClassifier
from flair.trainers import ModelTrainer
from flair.data import Sentence
from flair.datasets import TREC_6
import pandas as pd
from IPython.display import display

# Predict additional sentences
def gather_questions():
  df = pd.read_json('data/squad_test.json')
  data = df['data'].to_dict()
  count = 0
  test_questions = []
  for i in range(10):
    test_paragraphs = data[i]['paragraphs']
    for item in test_paragraphs:
      test_questions.append(item['qas'][0]['question'])

  return test_questions

def write_to_file(results):
  with open(r'results.txt', 'w') as f:
    for result in results:
      to_write = str(result) + "\n"
      f.write(to_write)

# 1. Load the trained model
tars = TARSClassifier.load('resources/taggers/trec/best-model.pt')

# Load known tasks to the model
existing_tasks = tars.list_existing_tasks()
print(f"Existing tasks are: {existing_tasks}")

# Switch to task we just created in model.py
tars.switch_to_task("question classification")

test_set = gather_questions()
results = []
for question in test_set:
  sentence = Sentence(question)
  tars.predict(sentence)
  results.append(sentence)

write_to_file(results)
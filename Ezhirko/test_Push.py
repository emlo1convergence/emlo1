import pytest
import pandas as pd
from gitignore_parser import parse_gitignore

matches = parse_gitignore('Ezhirko/.gitignore')
# you have NOT uploaded data
def test_data_folder_not_present():
    assert matches('Ezhirko/Data')

# you have NOT uploaded best-model-parameters.pt
def test_model_file_not_present():
    assert matches('Ezhirko/best-model-parameters.pt')

# your accuracy of the model is more than 70%
def test_model_accuracy_above_seventy():
    metrix_df = pd.read_csv('Ezhirko/Metrics.csv')
    acc = max(metrix_df['TestAccuracy'])
    assert acc > 70.0

# your accuracy of the cat class is more than 70%
def test_cat_class_accuracy_above_seventy():
    metrix_df = pd.read_csv('Ezhirko/Metrics.csv')
    acc = max(metrix_df['CatAccuracy'])
    assert acc > 70.0
    
# your accuracy of the dog class is more than 70%
def test_dog_class_accuracy_above_seventy():
    metrix_df = pd.read_csv('Ezhirko/Metrics.csv')
    acc = max(metrix_df['DogAccuracy'])
    assert acc > 70.0

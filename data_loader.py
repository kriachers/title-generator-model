import pandas as pd
import datasets
import os
from datasets import Dataset

"""
FILE READING
"""

#Here we define the size of dataset to train and test the model
DATASET_SIZE = 1000

current_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_directory, 'small_medium_articles.csv')

title_df = pd.read_csv(file_path)
title_df = title_df.head(DATASET_SIZE)

"""
DROP NA
"""

title_df = title_df.dropna(subset=["title"])
title_df = title_df[title_df['title'].apply(lambda x: isinstance(x, str))]

'''
TEST/TRAIN SPLITTING
'''

title_dataset = Dataset.from_pandas(title_df)
title_dataset = title_dataset.train_test_split(test_size=0.2)


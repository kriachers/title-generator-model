# Title Generator for Articles: Fine-tuned T5-small Model

This project contains code for a T5-small model ([link](https://huggingface.co/google/t5-small)) fine-tuned to generate titles for articles.

## Project Content:

- **data_loader.py** - Loads the data.
- **data_preprocess.py** - Preprocesses (tokenizes) the data.
- **train.py** - Trains and evaluates the model.
- **predict.py** - Runs the model on examples from the test set.

## Dataset for Training

The training used a truncated version of the Medium Articles Dataset ([link](https://www.kaggle.com/datasets/fabiochiusano/medium-articles)). The original dataset, which consists of 150,000 rows, was truncated to 20,000 rows and saved as `small_medium_articles.csv`.

## Installation and Usage

1. **Clone the repository:**

   ```bash
   git clone https://github.com/kriachers/title-generator-model.git
   cd title-generator-model
   ```

2. **Install the required packages:**

```
pip install -r requirements.txt
```
3. **Data Loading:**

Use data_loader.py to load the dataset:
```
python data_loader.py
```

4. **Data Preprocessing:**
Preprocess the dataset using data_preprocess.py:

```
python data_preprocess.py
```

5. **Model Training and Evaluation:**


Train and evaluate the model using train.py:

```
python train.py
```

5. **Generate Titles::**


Use predict.py to generate titles for the test set examples:

```
python predict.py
```




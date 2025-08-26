# Fake News Detection with Naive Bayes

This project implements a machine learning model to classify news articles as **fake** or **real** using the **Naive Bayes** algorithm. It leverages natural language processing (NLP) techniques to preprocess text data and build an effective classification model.

## Project Structure

├── Fake.csv # Dataset containing fake news articles

├── True.csv # Dataset containing real news articles

├── final_data.csv # Combined and preprocessed dataset

├── final_model.pkl # Trained Naive Bayes model

├── final_vector.pkl # TF-IDF vectorizer

├── Fake News Detection.ipynb # Jupyter Notebook for model training and evaluation

├── Fake News Detection.py # Python script for model training and evaluation

└── README.md # Project documentation

## Approach

1. **Data Collection**: Used publicly available datasets of labeled news articles.
2. **Preprocessing**: Cleaned and tokenized text, removed stopwords, applied TF-IDF vectorization.
3. **Modeling**: Trained a Naive Bayes classifier to distinguish between fake and real news.
4. **Evaluation**: Measured performance using accuracy, precision, recall, and F1-score.

## Getting Started

### Prerequisites

Install the required Python packages:

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn
Running the Model
Clone the repository:

git clone https://github.com/Nabin0727/fake-news-naive_bayes_algorithm.git
cd fake-news-naive_bayes_algorithm
Run the Jupyter Notebook:

jupyter notebook Fake\ News\ Detection.ipynb
Or execute the Python script:

python Fake\ News\ Detection.py
```
## Results
The Naive Bayes model achieves high accuracy in classifying news articles as fake or real.

## Future Enhancements
Create a Flask API for real-time predictions.

Compare with other classifiers like SVM and Random Forest.

Explore deep learning approaches for improved accuracy.

Fake News Detection with Naive Bayes

This project implements a machine learning model to classify news articles as fake or real using the Naive Bayes algorithm. It leverages natural language processing (NLP) techniques to preprocess text data and build an effective classification model.

📁 Project Structure
├── Fake.csv                # Dataset containing fake news articles
├── True.csv                # Dataset containing real news articles
├── final_data.csv          # Combined and preprocessed dataset
├── final_model.pkl         # Trained Naive Bayes model
├── final_vector.pkl        # TF-IDF vectorizer
├── Fake News Detection.ipynb # Jupyter Notebook for model training and evaluation
├── Fake News Detection.py   # Python script for model training and evaluation
└── README.md               # Project documentation

🧠 Approach

Data Collection: Utilized publicly available datasets containing labeled news articles.

Preprocessing: Cleaned and tokenized text data, removing stopwords and applying TF-IDF vectorization.

Modeling: Implemented the Naive Bayes algorithm to train the classification model.

Evaluation: Assessed model performance using metrics such as accuracy, precision, recall, and F1-score.

🚀 Getting Started
Prerequisites

Ensure you have the following Python packages installed:

pandas

numpy

scikit-learn

nltk

matplotlib

seaborn

You can install them using pip:

pip install pandas numpy scikit-learn nltk matplotlib seaborn

Running the Model

Clone the repository:

git clone https://github.com/Nabin0727/fake-news-naive_bayes_algorithm.git
cd fake-news-naive_bayes_algorithm


Run the Jupyter Notebook:

jupyter notebook Fake\ News\ Detection.ipynb


Or execute the Python script:

python Fake\ News\ Detection.py

📈 Results

The Naive Bayes model achieved an accuracy of approximately XX%, demonstrating its effectiveness in classifying news articles.

🔧 Future Enhancements

Implement a Flask API to serve the model for real-time predictions.

Explore other classification algorithms like SVM and Random Forest for comparison.

Integrate deep learning models for improved performance.

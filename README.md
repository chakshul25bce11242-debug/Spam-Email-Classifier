# 🛡️ Campus Spam & Phishing Email Classifier

## Overview
This project is an automated Machine Learning tool designed to classify text messages and emails as either "Spam" (malicious/phishing) or "Ham" (safe). It was built as a capstone project to demonstrate the practical application of Natural Language Processing (NLP) and supervised learning.

## How it Works
The classifier uses a **Multinomial Naive Bayes** algorithm. Before training, the raw email text is preprocessed and converted into numerical data using **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization, allowing the model to weigh the importance of specific words commonly used in phishing attacks.

## Prerequisites
To run this project, you will need Python installed on your machine along with the following libraries:
- `pandas`
- `scikit-learn`

You can install the dependencies using:
`pip install pandas scikit-learn`

## Setup and Usage

1. **Clone the repository:**
   `git clone [Insert Your Repo URL Here]`
2. **Download the Dataset:**
   Ensure you have a spam dataset (like the Kaggle SMS Spam Collection) saved as `spam.csv` in the root directory of the project.
3. **Run the Script:**
   Navigate to the folder in your terminal and run:
   `python classifier.py`
4. **Test it Yourself:**
   Once the model trains (which takes less than a second), the terminal will open an interactive prompt. You can paste any email text into the terminal, and the model will instantly predict whether it is `SPAM 🚨` or `SAFE ✅`. 

Type `quit` to exit the program.

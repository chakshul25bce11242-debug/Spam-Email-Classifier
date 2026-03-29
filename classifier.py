import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def load_and_prep_data(filepath):
    # Load dataset
    print("Loading data...")
    df = pd.read_csv(filepath, encoding='latin-1')
    df = df[['v1', 'v2']] # Adjust these column names based on your downloaded CSV
    df.columns = ['label', 'text']
    
    # Convert labels to binary (spam=1, ham=0)
    df['label'] = df['label'].map({'spam': 1, 'ham': 0})
    return df

def train_model(df):
    print("Training model...")
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
    
    # Vectorize text using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    
    # Evaluate
    predictions = model.predict(X_test_vec)
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(classification_report(y_test, predictions, target_names=['Ham (0)', 'Spam (1)']))
    
    return model, vectorizer

def predict_email(model, vectorizer, email_text):
    # Transform the new email text and predict
    vec_text = vectorizer.transform([email_text])
    prediction = model.predict(vec_text)
    return "SPAM 🚨" if prediction[0] == 1 else "SAFE (HAM) ✅"

if __name__ == "__main__":
    # 1. Setup (Make sure you have a spam.csv file in the same folder)
    # Using a placeholder path here. Update it to your actual file.
    dataset_path = 'spam.csv' 
    
    try:
        df = load_and_prep_data(dataset_path)
        trained_model, tfidf_vectorizer = train_model(df)
        
        # 2. Interactive Testing Loop
        print("\n--- Campus Phishing Checker Ready ---")
        while True:
            test_email = input("\nPaste an email to check (or type 'quit' to exit): ")
            if test_email.lower() == 'quit':
                break
            result = predict_email(trained_model, tfidf_vectorizer, test_email)
            print(f"Result: {result}")
            
    except FileNotFoundError:
        print(f"Error: Could not find '{dataset_path}'. Please download a spam dataset and update the path.")
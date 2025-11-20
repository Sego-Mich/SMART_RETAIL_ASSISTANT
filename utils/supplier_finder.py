import re
import json
import string
import nltk
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from keras.models import load_model 

# Helper: Text preprocessing
def preprocess_text(text, abbr_dict, pattern=None):
    # 1. Lowercase
    text = text.lower()

    # 2. Remove pattern (e.g., URLs)
    if pattern:
        text = re.sub(pattern, '', text)

    # 3. Expand abbreviations
    words = text.split()
    words = [abbr_dict.get(word, word) for word in words]
    text = " ".join(words)

    # 4. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 5. Tokenize
    tokens = word_tokenize(text)

    # 6. Remove stopwords
    stop_words = set(nltk_stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]

    # 7. Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return " ".join(tokens)

# Helper: Clean text (fallback)
def clean_text(text):
    stop_words = ENGLISH_STOP_WORDS
    words = re.findall(r'\b\w+\b', text.lower())
    return [word for word in words if word not in stop_words]

# Main function
def find_suppliers(text, df):
    print("Finding Suppliers...")

    try:
        print("Using Model...")

        # Load abbreviation dictionary
        with open('abbr.txt', 'r') as f:
            abbr_dict = json.load(f)

        # Load fitted vectorizer
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)

        # Load fitted label encoder
        with open("label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)

        # Preprocess input
        cleaned_text = preprocess_text(text, abbr_dict)
        user_vec = vectorizer.transform([cleaned_text])  # Ensure match with training vectorizer
        print("here") 
        import os

        from keras.models import load_model

        best_model = load_model("checkpoints/best_model_overall.h5")
        print("step 100")
        predicted_prob = best_model.predict(user_vec.toarray())
        predicted_class = predicted_prob.argmax(axis=1)
        predicted_label = label_encoder.inverse_transform(predicted_class)[0]

        print("Predicted Label:", predicted_label)

        # Match in DataFrame
        matches = df[df["Product Name"] == predicted_label]
        return matches[["Supplier/Vendor", "Unit Price"]].iloc[:5].drop_duplicates()

    except Exception as e:
        print(f"Model failed. Falling back to dataset search. Reason: {e}")
        user_words = clean_text(text)

        def match_row(desc):
            desc = str(desc).lower()
            return any(word in desc for word in user_words)

        matches = df[df["Item Description"].apply(match_row)]
        return matches[["Supplier/Vendor", "Unit Price", "Item Description"]].drop_duplicates()

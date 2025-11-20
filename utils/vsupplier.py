import pandas as pd
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re
import json
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
from tensorflow.keras.models import load_model


def find_suppliers(text, df):
    print("Finding Suppliers")
    try:
        print("Using Model")
        # Load abbreviation dictionary once
        with open('abbr.txt', 'r') as f:
            abbr_dict = json.load(f)
        vectorizer = ""
        label_encoder = ""

        def preprocess_text(text, pattern=None):
            # 1. Lowercase
            text = text.lower()

            # 2. Remove pattern if provided (e.g., URLs)
            if pattern:
                text = re.sub(pattern, '', text)

            # 3. Expand abbreviations
            words = text.split()
            words = [abbr_dict.get(word, word) for word in words]

            # 4. Remove punctuation
            text = " ".join(words)
            text = text.translate(str.maketrans('', '', string.punctuation))

            # 5. Tokenize
            tokens = word_tokenize(text)

            # 6. Remove stopwords
            stop_words = set(stopwords.words('english'))
            tokens = [w for w in tokens if w not in stop_words]

            # 7. Lemmatize
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(w) for w in tokens]

            return " ".join(tokens)

        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)

        with open("label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        user_vec = vectorizer.transform([preprocess_text(text)])  # must match training vectorizer
        print(text)
        best_model = load_model("checkpoints/best_model_overall.keras")
        predicted_prob = best_model.predict(user_vec.toarray())
        predicted_class = predicted_prob.argmax(axis=1)
        predicted_label = label_encoder.inverse_transform(predicted_class)[0]
        print(predicted_label)
        matches = df[df["Product Name"] == predicted_label]

        return matches[["Supplier/Vendor", "Unit Price"]].iloc[:5].drop_duplicates()
    except:
        print("Using Dataset")
        user_words = text.lower()

        def match_row(desc):
            desc = str(desc).lower()
            return any(word in desc for word in user_words)

        matches = df[df["Item Description"].apply(match_row)]
        return matches[["Supplier/Vendor", "Unit Price", "Item Description"]].drop_duplicates()


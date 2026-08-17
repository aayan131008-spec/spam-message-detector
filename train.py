import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

data = {
    "text": [
        "Congratulations! You've won a $1,000 Walmart gift card. Click here to claim.",
        "Hey, are we still meeting for coffee at 4 PM today?",
        "URGENT! Your bank account has been compromised. Log in immediately at http://fakebank.com",
        "Don't forget to submit your assignment before midnight.",
        "Free entry in a £2,000 weekly draw! Text WIN to 80082 now.",
        "Can you send me the lecture notes from yesterday's class?",
        "WINNER!! As a valued customer you have been selected to receive a £900 prize reward!",
        "I'm running a bit late, see you in 10 minutes."
    ],
    "label": [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Spam, 0 = Legitimate (Ham)
}

df = pd.DataFrame(data)

pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('model', MultinomialNB())
])

pipeline.fit(data['text'], data['label'])

joblib.dump(pipeline, "spam_pipeline.joblib")

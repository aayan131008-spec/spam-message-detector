import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

data = {
    "text": [
        # --- SPAM MESSAGES ---
        "URGENT! Your mobile number has won a £2,000 cash prize. Call 09061701461 to claim now.",
        "Congratulations! You've been selected for a $1,000 Walmart gift card. Click here: bit.ly/claim-gift",
        "FREE ENTRY: Win an iPhone 15 Pro today! Text WIN to 80082. Terms apply.",
        "Alert: Unusual login activity detected on your bank account. Verify your identity immediately at fakebank-security.com",
        "Your package delivery is on hold due to unpaid customs fee of $2.99. Pay now at postal-track-service.info",
        "Get rich quick! Earn $5,000 a week working from home with no experience needed. Reply YES for details.",
        "Final Notice: Your tax refund of $450 is ready to be transferred. Submit your debit card details now.",
        "You have 1 unread message from a hot local single! Click to view profile.",
        "Exclusive offer! Get 80% off all designer watches today only. Shop at luxury-deals-discount.net",
        "Dear customer, your Netflix subscription has expired. Update payment details at netflix-billing-update.com",
        "CLAIM NOW! You have 50,000 unused reward points expiring tonight. Redeem at points-reward-store.com",
        "Loan approved! Get up to $10,000 in your bank account in 15 minutes. No credit check required.",
        "Hot deal! Crypto signal group guarantees 500% profit daily. Join Telegram group now.",
        "WARNING: Your computer is infected with 3 viruses! Download anti-virus software now.",
        "You won a free cruise trip to Bahamas! Call 1-800-555-0199 immediately to secure your cabin.",

        # --- LEGITIMATE / HAM MESSAGES ---
        "Hey, are we still meeting for lunch at 1 PM today?",
        "Don't forget to bring the physics lab notes to class tomorrow.",
        "Can you please send me the updated PDF file when you get a chance?",
        "I'm running a few minutes late due to traffic, see you soon!",
        "Your verification code for logging in is 482910. Do not share this code with anyone.",
        "Hi Mom, I reached home safely. Will call you after dinner.",
        "Your appointment with Dr. Smith is confirmed for Thursday at 3:00 PM.",
        "Hey, did you finish the assignment for CS? Let me know if you need help.",
        "The store is out of milk, should I get almond milk instead?",
        "Your ride with Uber is arriving in 3 minutes. Driver: Ahmed (White Toyota).",
        "Thanks for picking up the groceries earlier, I really appreciate it!",
        "Let's plan a gaming session this weekend if everyone is free.",
        "Your food delivery order #4092 has been picked up and is on the way.",
        "Good luck with your exam today! You've got this.",
        "Hi, just checking in to see if you received my previous email about the project."
    ],
    "label": [
        # 1 = Spam (First 15)
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        # 0 = Legitimate (Next 15)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]
}

df = pd.DataFrame(data)

pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('model', MultinomialNB())
])

pipeline.fit(data['text'], data['label'])

joblib.dump(pipeline, "spam_pipeline.joblib")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 21:48:27 2025

@author: xudawang
"""
import pandas as pd
import nltk
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('wordnet', quiet=True)

the_path = "/Users/xudawang/Desktop/combined_dataset.json"
df = pd.read_json(the_path, lines=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

contexts = df['Context']
contexts_clean = [clean_text(text) for text in contexts]

sia = SentimentIntensityAnalyzer()

mental_health_keywords = {
    'Hopelessness': ['hopeless', 'worthless', 'empty', 'isolated', 'trapped', 'anxious', 'worried', 'fear', 'scared',
                    'overwhelmed', 'stressed', 'burnout', 'pressure', 'exhausted'],
    'Depression': ['depressed', 'sad', 'miserable', 'suicide', 'self-harm', 'kill', 'dying', 'death', 'end it all'],
    'Positive': ['hopeful', 'happy', 'joyful', 'excited', 'optimistic', 'grateful', 'content', 'peaceful']
}

negations = ['not', 'never', 'no', "don't", "doesn't", "isn't", "wasn't", "weren't", "can't", "couldn't", "dont", "doesnt", "isnt", "wasnt", "werent", "cant", "couldnt"]
safe_words = ['want', 'wish', 'prefer', 'would like']
intensifiers = ['very', 'extremely', 'super', 'incredibly', 'totally']

def detect_negation_with_keywords(text):
    words = text.split()
    for i, word in enumerate(words):
        if word in negations:
            for j in range(i+1, min(i+5, len(words))):
                for category, keywords in mental_health_keywords.items():
                    if words[j] in keywords:
                        window = words[max(0, i-3):j+1]
                        if any(safe in window for safe in safe_words):
                            return 'Neutral'
    return False

def detect_keywords(text):
    counts = {emotion: 0 for emotion in mental_health_keywords}
    words = text.split()
    for word in words:
        for emotion, keywords in mental_health_keywords.items():
            if word in keywords:
                counts[emotion] += 1
    return counts

def assign_label_final(text):
    common_neutral_phrases = ["im fine", "im okay", "im alright", "im good"]
    if any(phrase in text for phrase in common_neutral_phrases):
        return 'Neutral'
    
    negation_check = detect_negation_with_keywords(text)
    if negation_check == 'Neutral':
        return 'Neutral'
    
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    
    keywords_detected = detect_keywords(text)
    
    if any(word in text.split() for word in intensifiers):
        compound *= 1.2
    
    dominant_keyword = max(keywords_detected, key=keywords_detected.get)
    keyword_count = keywords_detected[dominant_keyword]
    
    if keyword_count > 0:
        if dominant_keyword == 'Positive' and compound > 0.1:
            return 'Positive'
        elif dominant_keyword == 'Depression':
            return 'Depression'
        elif dominant_keyword == 'Hopelessness':
            return 'Hopelessness'
    
    if compound <= -0.5:
        return 'Depression'
    elif -0.5 < compound <= -0.1:
        return 'Hopelessness'
    elif -0.1 < compound <= 0.2:
        return 'Neutral'
    else:
        return 'Positive'

emotion_to_message = {
    'Depression': "It's brave to seek help. You're not alone. :)",
    'Hopelessness': "Even in darkness, hope is possible. :)",
    'Positive': "Keep shining. You're doing great! :)",
    'Neutral': "If you ever want to talk, I'm here. :)"
}

def predict_wellbeing_message(text):
    cleaned_input = clean_text(text)
    pred_label = assign_label_final(cleaned_input)
    encouragement = emotion_to_message.get(pred_label, "You are valued. It's okay to seek support. :)")
    return pred_label, encouragement

labels = [assign_label_final(text) for text in contexts_clean]
df['CleanedContext'] = contexts_clean
df['Label'] = labels

print(df[['Context', 'Label']].head())
print("Label distribution:")
print(df['Label'].value_counts())

print("\nTesting examples with the four-category system:")
test_examples = [
    "I want to kill myself",
]

for example in test_examples:
    pred_label, message = predict_wellbeing_message(example)
    print(f"\nInput: {example}")
    print(f"Predicted Label: {pred_label}")
    print(f"Message: {message}")
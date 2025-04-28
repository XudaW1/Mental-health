#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 16:58:32 2025

@author: xudawang
"""
import streamlit as st
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('sentiment/vader_lexicon')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('wordnet', quiet=True)

# Initialize components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()

# Define mental health lexicons
mental_health_keywords = {
    'Hopelessness': ['hopeless', 'worthless', 'empty', 'isolated', 'trapped', 'anxious', 'worried', 'fear', 'scared',
                    'overwhelmed', 'stressed', 'burnout', 'pressure', 'exhausted'],
    'Depression': ['depressed', 'sad', 'miserable', 'suicide', 'self-harm', 'kill', 'dying', 'death', 'end it all'],
    'Positive': ['hopeful', 'happy', 'joyful', 'excited', 'optimistic', 'grateful', 'content', 'peaceful']
}

# Define negations and safe words
negations = ['not', 'never', 'no', "don't", "doesn't", "isn't", "wasn't", "weren't", "can't", "couldn't", 
            "dont", "doesnt", "isnt", "wasnt", "werent", "cant", "couldnt"]
safe_words = ['want', 'wish', 'prefer', 'would like']
intensifiers = ['very', 'extremely', 'super', 'incredibly', 'totally']

# Text preprocessing function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

# Negation detection function
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

# Keyword detection function
def detect_keywords(text):
    counts = {emotion: 0 for emotion in mental_health_keywords}
    words = text.split()
    for word in words:
        for emotion, keywords in mental_health_keywords.items():
            if word in keywords:
                counts[emotion] += 1
    return counts

# Classification function
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

# Generate supportive message based on emotional state
def get_supportive_message(emotional_state):
    messages = {
        'Depression': "It's brave to seek help. You're not alone. :)",
        'Hopelessness': "Even in darkness, hope is possible. :)",
        'Positive': "Keep shining. You're doing great! :)",
        'Neutral': "If you ever want to talk, I'm here. :)"
    }
    return messages.get(emotional_state, "You are valued. It's okay to seek support. :)")

# Analyze text function
def analyze_text(text):
    cleaned_text = clean_text(text)
    emotional_state = assign_label_final(cleaned_text)
    supportive_message = get_supportive_message(emotional_state)
    
    # Get detailed analysis
    sentiment_scores = sia.polarity_scores(cleaned_text)
    keyword_counts = detect_keywords(cleaned_text)
    
    return {
        'emotional_state': emotional_state,
        'supportive_message': supportive_message,
        'sentiment_scores': sentiment_scores,
        'keyword_counts': keyword_counts
    }

# Streamlit app
def main():
    st.set_page_config(
        page_title="Mental Health Sentiment Analyzer",
        page_icon="❤️",
        layout="centered"
    )
    
    st.title("Mental Health Sentiment Analyzer")
    st.write("A lightweight NLP tool to analyze emotional states in short text messages")
    
    # Text input
    user_input = st.text_area("Enter your text:", height=150, max_chars=500)
    
    # Analysis button
    if st.button("Analyze"):
        if user_input.strip():
            with st.spinner("Analyzing..."):
                analysis = analyze_text(user_input)
                
                # Display result with appropriate styling
                emotional_state = analysis['emotional_state']
                colors = {
                    'Positive': '#4CAF50',  # Green
                    'Neutral': '#2196F3',   # Blue
                    'Hopelessness': '#FF9800',  # Orange
                    'Depression': '#F44336'  # Red
                }
                
                # Display emotional state
                st.markdown(f"### Detected emotional state: <span style='color:{colors[emotional_state]}'>{emotional_state}</span>", unsafe_allow_html=True)
                
                # Display supportive message
                st.markdown(f"#### {analysis['supportive_message']}")
                
                # Show detailed analysis in an expandable section
                with st.expander("See detailed analysis"):
                    st.subheader("Sentiment Scores")
                    
                    # Create sentiment score bars
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Negative", f"{analysis['sentiment_scores']['neg']:.2f}")
                    col2.metric("Neutral", f"{analysis['sentiment_scores']['neu']:.2f}")
                    col3.metric("Positive", f"{analysis['sentiment_scores']['pos']:.2f}")
                    col4.metric("Compound", f"{analysis['sentiment_scores']['compound']:.2f}")
                    
                    # Show keyword detection
                    st.subheader("Detected Emotional Keywords")
                    
                    # Only show non-zero keyword counts
                    keywords_found = {k: v for k, v in analysis['keyword_counts'].items() if v > 0}
                    if keywords_found:
                        for category, count in keywords_found.items():
                            st.write(f"- {category}: {count} keyword(s)")
                    else:
                        st.write("No specific emotional keywords detected")
        else:
            st.error("Please enter some text to analyze")
    
    # Information section
    st.markdown("---")
    st.subheader("About this tool")
    st.write("""
    This tool uses a lexicon-based NLP approach to analyze emotional states in text. 
    It does not use machine learning, store your data, or send information to external servers.
    All analysis happens directly in your browser.
    
    **Note**: This is not a diagnostic tool and should not replace professional mental health support.
    """)
    
    # Resources section
    st.markdown("---")
    st.subheader("Mental Health Resources")
    
    resource_col1, resource_col2 = st.columns(2)
    
    with resource_col1:
        st.markdown("**Crisis Text Line**: Text HOME to 741741")
        st.markdown("**National Suicide Prevention Lifeline**: 1-800-273-8255")
    
    with resource_col2:
        st.markdown("**SAMHSA's National Helpline**: 1-800-662-4357")
        st.markdown("**International Association for Suicide Prevention**: [Resources](https://www.iasp.info/resources/Crisis_Centres/)")

if __name__ == "__main__":
    main()
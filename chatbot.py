import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import re
import difflib
import random
import os

class CustomerServiceChatbot:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.qna_database = {}
        self.trained = False
        self.categories = []
        
        print("Initializing Customer Service Chatbot...")
        self.load_qna_responses()
        if not self.load_saved_model():
            self.train_model()
    
    def load_qna_responses(self):
        """Load Q&A responses from CSV"""
        try:
            qna_df = pd.read_csv('Ques_Ans List.csv')
            print(f"Loaded {len(qna_df)} Q&A pairs")
            
            for _, row in qna_df.iterrows():
                if pd.notna(row['User Question']) and pd.notna(row['Chatbot Answer']):
                    question = str(row['User Question']).lower().strip()
                    answer = str(row['Chatbot Answer']).strip()
                    self.qna_database[question] = answer
            
            print(f"Built Q&A database: {len(self.qna_database)} entries")
            
        except Exception as e:
            print(f"Warning: Could not load Q&A data: {e}")
    
    def preprocess_text(self, text):
        """Clean text for ML processing"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def train_model(self):
        """Train ML model on customer data"""
        print("Training ML model...")
        
        try:
            # Load training data
            train_df = pd.read_csv('train_sample_data.csv')
            print(f"Loaded training data: {len(train_df)} records")
            
            # Use the columns we identified
            text_col = 'Consumer complaint narrative'
            label_col = 'Product'
            
            # Clean data
            clean_df = train_df.dropna(subset=[text_col, label_col]).copy()
            clean_df['clean_text'] = clean_df[text_col].apply(self.preprocess_text)
            clean_df = clean_df[clean_df['clean_text'].str.len() > 10]
            
            print(f"Clean training samples: {len(clean_df)}")
            
            X = clean_df['clean_text']
            y = clean_df[label_col]
            
            self.categories = list(y.unique())
            print(f"Categories: {len(self.categories)}")
            
            # Create features
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2
            )
            X_vec = self.vectorizer.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_vec, y, test_size=0.2, random_state=42
            )
            
            # Train model
            self.model = LogisticRegression(max_iter=1000, random_state=42)
            self.model.fit(X_train, y_train)
            
            # Test accuracy
            test_acc = accuracy_score(y_test, self.model.predict(X_test))
            print(f"Test accuracy: {test_acc:.3f}")
            
            # Save model
            joblib.dump(self.model, 'chatbot_model.pkl')
            joblib.dump(self.vectorizer, 'chatbot_vectorizer.pkl')
            joblib.dump(self.categories, 'categories.pkl')
            
            self.trained = True
            print("Model training completed!")
            
        except Exception as e:
            print(f"Training error: {e}")
            self.trained = False
    
    def load_saved_model(self):
        """Load previously saved model"""
        try:
            self.model = joblib.load('chatbot_model.pkl')
            self.vectorizer = joblib.load('chatbot_vectorizer.pkl')
            self.categories = joblib.load('categories.pkl')
            self.trained = True
            print("Loaded saved model successfully")
            return True
        except:
            print("No saved model found")
            return False
    
    def find_qna_match(self, user_message):
        """Find matching Q&A response"""
        user_clean = user_message.lower().strip()
        
        # Direct match
        if user_clean in self.qna_database:
            return self.qna_database[user_clean], 1.0
        
        # Find best match
        best_match = None
        best_score = 0
        
        for question, answer in self.qna_database.items():
            similarity = difflib.SequenceMatcher(None, user_clean, question).ratio()
            if similarity > best_score and similarity > 0.6:
                best_score = similarity
                best_match = answer
        
        return best_match, best_score
    
    def predict_intent(self, user_message):
        """Predict intent using ML model"""
        if self.trained and self.model:
            try:
                clean_text = self.preprocess_text(user_message)
                text_vec = self.vectorizer.transform([clean_text])
                prediction = self.model.predict(text_vec)[0]
                confidence = max(self.model.predict_proba(text_vec)[0])
                return prediction, confidence
            except:
                pass
        
        # Fallback rules
        msg_lower = user_message.lower()
        if any(word in msg_lower for word in ['balance', 'account']):
            return 'Checking or savings account', 0.7
        elif any(word in msg_lower for word in ['card', 'credit']):
            return 'Credit card or prepaid card', 0.8
        else:
            return 'General inquiry', 0.5
    
    def generate_response(self, user_message):
        """Generate response using Q&A and ML"""
        # Try Q&A first
        qna_response, qna_score = self.find_qna_match(user_message)
        if qna_response and qna_score > 0.7:
            return qna_response
        
        # Use ML prediction
        intent, confidence = self.predict_intent(user_message)
        
        if confidence < 0.3:
            return "I'm not sure I understand. Could you please rephrase your question?"
        
        # Intent-based responses
        responses = {
            'checking or savings account': "I can help you with your checking or savings account. What specific issue are you experiencing?",
            'credit card': "I'll assist you with your credit card concern. What problem are you facing?",
            'credit reporting': "I can help with credit reporting issues. What do you need assistance with?"
        }
        
        intent_lower = intent.lower()
        for key, response in responses.items():
            if key in intent_lower:
                return response
        
        return f"I'll help you with your {intent} inquiry. Could you provide more details?"
    
    def get_stats(self):
        """Get chatbot statistics"""
        return {
            'trained': self.trained,
            'categories': len(self.categories) if self.categories else 0,
            'qna_pairs': len(self.qna_database)
        }

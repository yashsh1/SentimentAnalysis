import pickle

# Load vectorizer and model
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
model = pickle.load(open('sentiment_model.pkl', 'rb'))

# Example: take input from terminal
text = input("Enter text to analyze: ")

# Transform and predict
X = vectorizer.transform([text])
prediction = model.predict(X)

if(prediction[0] == 1):
    print("Sentiment: Positive 😁")
else:   
    print("Sentiment: Negative 😞")

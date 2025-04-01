import pandas as pd
import numpy as np
import nltk
import tkinter as tk
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from tkinter import ttk, messagebox

# Load dataset
df = pd.read_csv('chat.csv')

# Clean data
df['text'] = df['text'].fillna('')
df['label'] = pd.to_numeric(df['label'], errors='coerce')
df = df.dropna(subset=['label'])
df = df[df['label'].isin([0, 1])]

if df.empty:
    raise ValueError("No valid data left after cleaning! Check CSV for formatting issues.")

texts = df['text']
labels = df['label']

# Preprocessing (preserve more raw text)
def preprocess_text(text):
    tokens = word_tokenize(str(text).lower())
    # No stopword removal to keep context (e.g., "im," "ya")
    tokens = [t for t in tokens if t.isalnum() or t in ['😍', '🙄', '-w-', '!', '...']]  # Keep informal tokens
    return ' '.join(tokens)

processed_texts = [preprocess_text(t) for t in texts]

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')  # Handle OOV explicitly
tokenizer.fit_on_texts(processed_texts)
X_sequences = tokenizer.texts_to_sequences(processed_texts)
max_length = 40  # Reduced for shorter replies
X_padded = pad_sequences(X_sequences, maxlen=max_length, padding='post')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_padded, labels, test_size=0.2, random_state=42)

# Build LSTM model with more regularization
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=32, input_length=max_length))
model.add(LSTM(32, return_sequences=False))  # Smaller LSTM
model.add(Dropout(0.5))  # Higher dropout
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train with early stopping
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.2, callbacks=[early_stopping], verbose=1)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Model accuracy: {accuracy:.2%}")

# GUI
class InterestPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Interest Predictor")
        self.root.geometry("400x300")
        
        ttk.Label(root, text="Enter her message:").grid(row=0, column=0, padx=10, pady=10)
        self.text_input = tk.Text(root, height=4, width=30)
        self.text_input.grid(row=0, column=1, padx=10, pady=10)
        
        ttk.Button(root, text="Predict", command=self.make_prediction).grid(row=1, column=0, columnspan=2, pady=10)
        
        self.result_var = tk.StringVar()
        ttk.Label(root, text="Prediction:").grid(row=2, column=0, padx=10, pady=10)
        ttk.Label(root, textvariable=self.result_var).grid(row=2, column=1, padx=10, pady=10)

    def make_prediction(self):
        try:
            user_text = self.text_input.get("1.0", tk.END).strip()
            if not user_text:
                raise ValueError("Please enter a message!")
            
            processed_input = preprocess_text(user_text)
            print(f"Raw input: {user_text}")
            print(f"Processed input: {processed_input}")
            input_sequence = tokenizer.texts_to_sequences([processed_input])
            print(f"Tokenized sequence: {input_sequence}")
            input_padded = pad_sequences(input_sequence, maxlen=max_length, padding='post')
            print(f"Padded input: {input_padded}")
            
            probability = model.predict(input_padded, verbose=0)[0][0]
            prediction = 1 if probability >= 0.5 else 0
            confidence = probability if prediction == 1 else 1 - probability
            
            result = "Interested" if prediction == 1 else "Uninterested"
            self.result_var.set(f"{result} - {confidence:.2%} confidence")
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = InterestPredictorApp(root)
    root.mainloop()
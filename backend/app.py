from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import pickle
import numpy as np

# Load model & tokenizer
model = tf.keras.models.load_model("text_generator_model.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_seq_len = model.input_shape[1] + 1

app = FastAPI(title="Next Word Prediction API")

class TextRequest(BaseModel):
    text: str
    n_words: int = 1

def predict_next_words(seed_text, n_words=1):
    for _ in range(n_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = tf.keras.preprocessing.sequence.pad_sequences(
            [token_list], maxlen=max_seq_len-1, padding='pre'
        )
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)

        for word, index in tokenizer.word_index.items():
            if index == predicted:
                seed_text += " " + word
                break
    return seed_text

@app.post("/predict")
def predict(req: TextRequest):
    result = predict_next_words(req.text, req.n_words)
    return {"input": req.text, "prediction": result}

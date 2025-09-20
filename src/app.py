import numpy as np
from preprocess import preprocess
import keras
from training import model, tokenizer, max_length, padding_type, trunc_type
import gradio as gr
import time

def predict_answer(message, history):
    history.append({'role': 'user', 'content': message})
    question = preprocess(message)
    sequence = tokenizer.texts_to_sequences([question])
    padded_sequence = keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    pred = model.predict(padded_sequence)[0]
    idx = np.argmax(pred)
    answer = tokenizer.index_word[idx]
    history.append({'role': 'assistant', 'content': answer})
    time.sleep(0.2)
    print(answer)
    yield answer
        
gr.ChatInterface(
    predict_answer,
    type='messages',
    chatbot=gr.Chatbot(type="messages", height=400, elem_id="chatbot"),
    title="Chatbot",
    description="Ask the Chatbot any question"
    ).launch(share=True)
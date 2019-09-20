import pickle

import flask
import numpy as np
import tensorflow as tf
from flask import jsonify, request
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = flask.Flask(__name__)

model, tokenizer = None, None
labels = [
    "Авто",
    "Товары для здоровья",
    "Электроника",
    "Бытовая техника",
    "Строительство и ремонт",
    "Товары для дома",
    "Детские товары",
    "Досуг и развлечения",
    "Компьютерная техника",
    "Товары для красоты",
    "Одежда, обувь и аксессуары",
    "Продукты",
    "Спорт и отдых",
    "Дача, сад и огород",
    "Товары для животных",
]

MAX_SEQUENCE_LENGTH = 10


def preprocess_text(v):
    v = tokenizer.texts_to_sequences(v)
    return pad_sequences(v, maxlen=MAX_SEQUENCE_LENGTH)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    x = preprocess_text(data["input"])
    y = model.predict(x)

    result = []
    for item in y:
        index = np.argmax(item)
        result.append({
            "label": labels[index],
            "confidence": item.tolist()[index],
        })

    return jsonify(result)


if __name__ == "__main__":
    model = tf.keras.models.load_model("goods_classifier.h5")
    tokenizer = pickle.load(open("tokenizer.pickle", "rb"))
    app.run()

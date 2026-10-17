import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
BINS = {
    "cardboard": "Papier / carton",
    "paper": "Papier / carton",
    "glass": "Verre",
    "metal": "Métal",
    "plastic": "Plastique",
    "trash": "Déchets non recyclables",
}
SEUIL = 0.6

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("waste_model.keras")

model = load_model()

st.title("Smart Waste Sorting")
st.write("Montre un déchet à la caméra : l'IA devine sa catégorie.")

mode = st.radio("Source de l'image", ["Caméra", "Envoyer une photo"], horizontal=True)
if mode == "Caméra":
    photo = st.camera_input("Prends une photo")
else:
    photo = st.file_uploader("Choisis une photo", type=["jpg", "jpeg", "png"])

if photo:
    img = Image.open(photo).convert("RGB")
    st.image(img, width=300)
    x = np.expand_dims(np.array(img.resize((224, 224)), dtype="float32"), 0)
    probs = model.predict(x, verbose=0)[0]
    best = int(np.argmax(probs))
    conf = float(probs[best])

    if conf < SEUIL:
        st.warning(f"Je ne suis pas sûre ({conf:.0%}). Essaie un seul objet, sur fond uni.")
    else:
        st.success(f"{CLASSES[best]} ({conf:.0%}) → poubelle : {BINS[CLASSES[best]]}")
    st.bar_chart({c: float(p) for c, p in zip(CLASSES, probs)})
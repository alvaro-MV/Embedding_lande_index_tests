import openai
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from getpass import getpass
import time

# Configura tu clave de API
if "OPENAI_API_KEY" not in os.environ:
	os.environ["OPENAI_API_KEY"] = getpass("api key: ")

# --- Funciones principales ---

def separar_en_parrafos(ruta_fichero):
    parrafos = []
    with open(ruta_fichero, 'r', encoding='utf-8') as f:
        contenido = f.read()
    
    # Separar por líneas en blanco (una o más)
    bloques = contenido.strip().split('\n\n')
    for bloque in bloques:
        parrafo = bloque.strip().replace('\n', ' ')
        if parrafo:  # evitar párrafos vacíos
            parrafos.append(parrafo)
    
    return parrafos
		
def generate_embeddings(text, model='text-embedding-3-small'):
    response = openai.embeddings.create(
        input=text,
        model=model,
        encoding_format="float"
    )
    return response.data[0].embedding

def softmax(x):
    x = np.array(x)
    e_x = np.exp(x - np.max(x))  # mayor estabilidad numérica
    return e_x / e_x.sum()

def lande_diversity(p):
    """Calcula el índice de diversidad de Lande a partir de un vector de probabilidad."""
    D = np.sum(p ** 2)  # Índice Simpson
    N = len(p)
    return (N / (N - 1)) * (1 - D)

# --- Textos de ejemplo (expositivos) ---

texts = {
    "Cat": separar_en_parrafos("data/cat.txt"),
    # "Coffee": [
    #     "Coffee is a brewed drink prepared from roasted coffee beans, the seeds of berries from certain Coffea species.",
    #     "From the coffee fruit, the seeds are separated to produce a stable, raw product: unroasted green coffee.",
    #     "The beans are roasted and then ground into a fine powder and brewed to make coffee.",
    #     "It is one of the most popular drinks in the world, and can be prepared and presented in a variety of ways."
    # ],
    # "Machine Learning": [
    #     "Machine learning is a field of artificial intelligence that uses statistical techniques to give computer systems the ability to learn from data.",
    #     "These algorithms build a mathematical model based on sample data, known as training data.",
    #     "The discipline draws from fields like statistics, computer science, and cognitive science.",
    #     "Common applications include recommendation systems, image recognition, and natural language processing."
    # ]
}

# --- Experimentación ---

results = []

for topic, paragraphs in texts.items():
    cumulative_text = ""
    for i in range(1, len(paragraphs) + 1):
        cumulative_text = " ".join(paragraphs[:i])
        embedding = generate_embeddings(cumulative_text)
        probs = softmax(embedding)
        lande = lande_diversity(probs)
        results.append({
            "Topic": topic,
            "Paragraphs": i,
            "Lande Diversity": lande
        })
        time.sleep(1)  # Evita rate limits

df = pd.DataFrame(results)

# --- Visualización ---

plt.figure(figsize=(10, 6))
for topic in df["Topic"].unique():
    subset = df[df["Topic"] == topic]
    plt.plot(subset["Paragraphs"].to_numpy(), subset["Lande Diversity"].to_numpy(), marker="o", label=topic)

plt.xlabel("Number of Paragraphs")
plt.ylabel("Lande Diversity Index")
plt.title("Embedding Diversity Collapse with Growing Text Input")
plt.legend()
plt.grid(True)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.show()

import torch
import openai

def generate_embeddings(element, model = 'text-embedding-3-small'):
    embedding = openai.embeddings.create(
        input= element,
        model=model,
        encoding_format="float",

        ).data[0].embedding
    return embedding

def get_embedding_sample(sample, column):
  lista = []
  for split, el in sample.iterrows():
      lista.append(generate_embeddings(el[column]))
  embeddings_elements = torch.Tensor(lista)
  return embeddings_elements

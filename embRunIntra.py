import embGet
import torch
import numpy as np
import embUtils as utils

roleplay_task_indication = """
 This dataset contians conversations between a user and an agen, calles system.
  The assistant does a roleplay trying to adjust to a real character, either human or non-human, or fictitious.
  The assistant uses a distinctive way of answering, in relation to language usage and personality traits.
"""

def join_next_chunk(Abstract, chunks, step):
    Abstract += chunks[step] + "|"
    return Abstract

def run_intra(chunks, steps = 10):
  incrementText = ""
  lande_measure = []
  i = 0
  while i < steps:
      incrementText = " ".join(chunks[:i])
      text_embedding = np.array(embGet.generate_embeddings(incrementText))
      la = utils.lande_intra_index(text_embedding)
      print(la)
      lande_measure.append(la)
      i += 1
  return lande_measure

def run_expositive_mayeutic(updating, client):
    incrementText = ""
    lande_measure = []
    i = 0
    for i in range(len(updating)):
        question = client.get_questions(incrementText, roleplay_task_indication)
        answer = client.correct_questions(question, updating)
        incrementText += " " + answer
        tensor = torch.Tensor(
            embGet.generate_embeddings(incrementText)).unsqueeze(0)
        la = utils.lande_intra_index(tensor.numpy()[0])
        print(la)
        lande_measure.append(la)
        i += 1
    return lande_measure

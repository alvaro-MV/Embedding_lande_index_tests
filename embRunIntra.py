import embGet
import numpy as np
import embUtils as utils

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

def run_on_expositive_texts_intra(abstract_chunks, n_batches = 8):
    return run_intra(abstract_chunks, n_batches)

import embRun
import embUtils as utils

def run_intra(embeddings_el, df, updating, update_fn, client, steps = 10):
  incrementText = ""
  lande_measure = []
  i = 0
  while i < steps:
      incrementText = update_fn(client, incrementText, df, updating, i)    
      la = utils.lande_intra_index(embeddings_el)
      print(la)
      lande_measure.append(la)
      i += 1
  return lande_measure

def run_on_expositive_texts_intra(abstract_chunks, title_embs, n_batches = 8):
    return run_intra(title_embs, None, abstract_chunks, embRun.update_expo_abstract, None, n_batches)
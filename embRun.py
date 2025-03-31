import torch
import embUtils as utils
from embUtils import Chat
import embGet


roleplay_task_indication = """
 This dataset contians conversations between a user and an agen, calles system.
  The assistant does a roleplay trying to adjust to a real character, either human or non-human, or fictitious.
  The assistant uses a distinctive way of answering, in relation to language usage and personality traits.
"""

def update_roleplay_answer(client, Answers, df, label, step):
    question = client.get_questions(Answers, roleplay_task_indication)
    answer = client.correct_questions(question, label)
    Answers += "|" + answer
    return Answers

def update_expo_abstract(client, Abstract, df, chunks, step):
    Abstract += chunks[step] + "|"
    return Abstract

def update_roleplay_baseline(client, Answers, df, label, step):
    return (utils.extractBaselineAnswer(df))

def run(embeddings_el, df, updating, update_fn, client, steps = 10):
  incrementText = ""
  lande_measure = []
  i = 0
  while i < steps:
      incrementText = update_fn(client, incrementText, df, updating, i)    
      tensor = torch.Tensor(
         embGet.generate_embeddings(incrementText)).unsqueeze(0)
      la = utils.lande_index(tensor, embeddings_el)
      print(la)
      lande_measure.append(la)
      i += 1
  return lande_measure

def run_conversation(embeddings_el, generics, label, conversation_rounds = 10):
    client = Chat()
    lande_convers = run(embeddings_el, generics, label, 
                        update_roleplay_answer, client, conversation_rounds)
    lande_base = run(embeddings_el, generics, label,
                        update_roleplay_baseline, client, conversation_rounds)
    return lande_convers, lande_base

def run_on_expositive_texts(abstract_chunks, title_embs, n_batches = 8):
    return run(title_embs, None, abstract_chunks, update_expo_abstract, None, n_batches)

#   Abstract = ""
#   i = 0
#   lande_abstract = []
#   while i < n_batches:
#       Abstract += abstract_chunks[i] + "|"
#       #print(f"Answers: -----------: {Answers}\n")
#       t_chunks = torch.Tensor(embGet.generate_embeddings(Abstract)).unsqueeze(0)
#       la = utils.lande_index(t_chunks, title_embs)
#       #print(f"li: {la}\n")
#       lande_abstract.append(la)
#       i += 1
#   return lande_abstract
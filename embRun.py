import torch
import embUtils as utils
from embUtils import Chat
from embDataset import HFDataset
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

def run_conversation_intra(embeddings_el, df, updating, update_fn, client, steps = 10):
    incrementText = ""
    lande_measure = []
    i = 0
    while i < steps:
        incrementText = update_fn(client, incrementText, df, updating, i)    
        tensor = torch.Tensor(
            embGet.generate_embeddings(incrementText)).unsqueeze(0)
        la = utils.lande_intra_index(tensor.numpy()[0])
        print(la)
        lande_measure.append(la)
        i += 1
    return lande_measure

def run_conversation(embeddings_el, generics : HFDataset, label, conversation_rounds = 10):
    client = Chat()
    lande_convers = run(embeddings_el, generics.get_df(), label, 
                        update_roleplay_answer, client, conversation_rounds)
    lande_intra = run_conversation_intra(embeddings_el, generics.get_df(), label,
                        update_roleplay_answer, client, conversation_rounds)
    lande_base = run(embeddings_el, generics.get_df(), label,
                        update_roleplay_baseline, client, conversation_rounds)
    return lande_convers, lande_intra, lande_base

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
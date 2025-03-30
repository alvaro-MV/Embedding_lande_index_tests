import torch
import embUtils as utils
import embGet


roleplay_task_indication = """
 This dataset contians conversations between a user and an agen, calles system.
  The assistant does a roleplay trying to adjust to a real character, either human or non-human, or fictitious.
  The assistant uses a distinctive way of answering, in relation to language usage and personality traits.
"""

def update_roleplay_answer(Answers, df, label, step):
    question = utils.get_questions(Answers, roleplay_task_indication)
    answer = utils.correct_questions(question, label)
    Answers += "|" + answer
    return Answers

def update_expo_abstract(Abstract, df, chunks, step):
    Abstract += chunks[step] + "|"
    return Abstract

def update_roleplay_baseline(Answers, df, label, step):
    return (utils.extractBaselineAnswer(df))

def run(embeddings_el, df, updating, update_fn, steps = 10):
  incrementText = ""
  lande_measure = []
  i = 0
  while i < steps:
      incrementText = update_fn(incrementText, df, updating, steps)    
      tensor = torch.Tensor(
         embGet.generate_embeddings(incrementText)).unsqueeze(0)
      la = utils.lande_index(tensor, embeddings_el)
      lande_measure.append(la)
      i += 1
  return lande_measure

def run_conversation(embeddings_el, generics, label, conversation_rounds = 10):
    run(embeddings_el, generics, label, update_roleplay_answer, conversation_rounds)

def run_on_expositive_texts(abstract_chunks, title_embs, n_batches = 8):
    run(title_embs, None, abstract_chunks, update_expo_abstract, n_batches)

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
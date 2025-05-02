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

def update_mayeutic_answer(client, Answers, df, label, step):
    question = client.get_questions(Answers, roleplay_task_indication)
    answer = client.correct_questions(question, label)
    Answers += "|" + answer
    return Answers

def update_expo_abstract(client, Abstract, df, chunks, step):
    Abstract += chunks[step] + "|"
    return Abstract

def update_join_next_chunk(client, Abstract, df, chunks, step):
    Abstract = " ".join(chunks[:step])
    return Abstract

def update_roleplay_baseline(client, Answers, df, label, step):
    return (utils.extractBaselineAnswer(df))

class Run:
    def __init__(self, update_fn, df= None, client = None, steps = None):
        self.df = df
        self.update_fn = update_fn
        self.client = client
        self.steps = steps
    
    def run(self, embeddings_el, updating, type_index):
        incrementText = ""
        lande_measure = []
        i = 0
        while i < self.steps:
            incrementText = self.update_fn(self.client, incrementText, self.df, updating, i)    
            tensor = torch.Tensor(
                embGet.generate_embeddings(incrementText)).unsqueeze(0)
            if type_index == 'inter':
                lande = utils.lande_index(tensor, embeddings_el)
            elif type_index == 'intra':
                lande = utils.lande_intra_index(tensor.numpy()[0])
            print(lande)
            lande_measure.append(lande)
            i += 1
        return lande_measure, i


def run_conversation(embeddings_el, generics : HFDataset, label, conversation_rounds = 10):
    client = Chat()
    runner_mayeutic = Run(update_mayeutic_answer, None, client, conversation_rounds)
    runner_baseline = Run(update_roleplay_baseline, generics.get_df(), None, conversation_rounds)

    # lande_convers = runner_mayeutic.run(embeddings_el, label, 'inter')
    lande_intra = runner_mayeutic.run(None, label, 'intra')
    lande_base = runner_baseline.run(None, label, 'intra')
    
    return lande_intra, lande_base
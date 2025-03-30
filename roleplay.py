import openai
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
import random
from getpass import getpass
import embDataset
import embGet
import embRun

## Esto realmente iría en main.
# os.environ["OPENAI_API_KEY"] = getpass("api key: ")
# client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
## Esto realmente iría en main.


# Funcion que se le pasa a método
# trasnsform_dataset
def select_train_set(dataset):
	return dataset['train']

roleplay_task_indication = """
 This dataset contians conversations between a user and an agen, calles system.
  The assistant does a roleplay trying to adjust to a real character, either human or non-human, or fictitious.
  The assistant uses a distinctive way of answering, in relation to language usage and personality traits.
"""

roleplay = embDataset.HFDataset("hieunguyenminh/roleplay")
roleplay.load_dataset()
generics = embDataset.HFDataset("generics_kb")
generics.load_dataset()

roleplay.transform_dataset(select_train_set)
generics.transform_dataset(select_train_set)

roleplay.get_df_from_dataset()
generics.get_df_from_dataset()

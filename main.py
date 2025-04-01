import os
import json
import argparse
from roleplay import roleplay
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from getpass import getpass
import torch
import pandas as pd
from expositive import expositive

def convert_tensors_to_floats(item):
    for key, value in item.items():
        print(f"key: {key},   value: {value}\n")
        if isinstance(value, list):
            item[key] = [v.item() if isinstance(v, torch.Tensor) else v for v in value]
        elif isinstance(value, torch.Tensor):
            item[key] = value.item()
    return item



if "OPENAI_API_KEY" not in os.environ:
	os.environ["OPENAI_API_KEY"] = getpass("api key: ")

parser = argparse.ArgumentParser(description = 'Test for probing lande index as a proxy for information flow')
 
parser.add_argument('-t', '--task',  dest ='task', 
                    action ='store', 
                    help ='task to be performed: <expositive> or <conversation>')


args = parser.parse_args()
print(args.task)

if (args.task == 'conversation'):
	result = roleplay()
elif (args.task == 'expositive'):
	result = expositive(0)
elif (args.task == 'expositive_intra'):
	result = expositive(1)
print(result)

paragraphs = list(range(1, len(result) + 1))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(paragraphs, result, marker="o", label="Lande Diversity")

plt.xlabel("Number of Paragraphs")
plt.ylabel("Lande Diversity Index")
plt.title("Embedding Diversity Collapse with Growing Text Input")
plt.legend()
plt.grid(True)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.show()
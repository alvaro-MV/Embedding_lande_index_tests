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

def plot_expositive(result):
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

def plot_conversation(resultados):
    round_values = [5, 10, 20]
    role_batch_size = [3, 8, 15, 20]
    fig, axs = plt.subplots(3, 3, figsize=(8.3, 9))
    row = 0
    for i in range(0, 3):
        for j in range(0, 3):
            axs[i, j].plot(resultados.iloc[row, 2])
            axs[i, j].plot(resultados.iloc[row, 3])
            axs[i, j].set_title(f'lote de {resultados.iloc[row, 0]}')
            row += 1

    for ax in axs.flat:
        ax.set(xlabel='rondas', ylabel='Lande')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

if "OPENAI_API_KEY" not in os.environ:
	os.environ["OPENAI_API_KEY"] = getpass("api key: ")

parser = argparse.ArgumentParser(description = 'Test for probing lande index as a proxy for information flow')
 
parser.add_argument('-t', '--task',  dest ='task', 
                    action ='store', 
                    help ='task to be performed: <expositive> or <conversation>')


args = parser.parse_args()
print(args.task)

if (args.task == 'conversation'):
  resultado = roleplay()
  plot_conversation(resultado)

elif (args.task == 'expositive'):
  resultado = expositive(0)
  plot_expositive(resultado)


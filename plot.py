import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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
    ks = sorted(set(entry['k'] for entry in resultados))
    rounds_set = sorted(set(entry['conversation_rounds'] for entry in resultados))

    # Crear un diccionario para acceder rápidamente a los datos por (k, rounds)
    data_dict = {(d['k'], d['conversation_rounds']): d for d in resultados}

    # Crear la cuadrícula de subplots
    fig, axs = plt.subplots(len(ks), len(rounds_set), figsize=(4 * len(rounds_set), 3 * len(ks)), squeeze=False)

    # Rellenar los subplots
    for i, k in enumerate(ks):
        for j, rounds in enumerate(rounds_set):
            ax = axs[i][j]
            entry = data_dict.get((k, rounds), None)
            if entry:
                x = list(range(1, rounds + 1))
                # conv = [t.item() for t in entry['lande_conversation']]
                intra = [t for t in entry['lande_intra']]
                # base = [t.item() for t in entry['lande_baseline']]
                # ax.plot(x, conv, marker='o', label='Conversation')
                ax.plot(x, intra, marker='o', label='Intra')
                # ax.plot(x, base, marker='s', label='Baseline')
                ax.set_title(f'k={k}, rounds={rounds}')
            else:
                ax.set_visible(False)  # Ocultar subplot si no hay datos
            ax.set(xlabel='Ronda', ylabel='Lande')
    for ax in axs.flat:
        ax.label_outer()

    # Añadir leyenda a uno de los subplots
    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
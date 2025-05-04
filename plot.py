import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import math

def plot_expositive_panels(results, n_per_row: int = 2, figsize=(8, 3)):
    """
    Dibuja un gráfico independiente por texto expositivo,
    organizados en filas con `n_per_row` columnas.

    Parameters
    ----------
    results : List[dict]
        Cada dict debe contener las claves:
        - 'lande_intra'   : lista con los valores índice de Lande
        - 'n_paragraphs'  : número total de pasos
        - 'text_name'     : nombre del texto
    n_per_row : int, optional
        Número de gráficos por fila (default 2).
    figsize : tuple, optional
        Tamaño (width, height) de cada subgráfico en pulgadas.
    """
    n_texts   = len(results)
    n_rows    = math.ceil(n_texts / n_per_row)

    # Figura global
    fig, axes = plt.subplots(
        n_rows, n_per_row,
        figsize=(figsize[0] * n_per_row, figsize[1] * n_rows),
        sharey=False, sharex=False
    )
    axes = axes.flatten()  # facilita el indexado lineal

    for ax_idx, (entry, ax) in enumerate(zip(results, axes)):
        lande = entry['lande_intra']
        name  = entry['text_name']
        xs    = list(range(1, entry['n_paragraphs'] + 1))

        ax.plot(xs, lande, marker='o')
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Number of Paragraphs")
        ax.set_ylabel("Lande Diversity Index")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True)

    # Oculta ejes sobrantes si n_texts no es múltiplo de n_per_row
    for ax in axes[n_texts:]:
        ax.axis('off')

    fig.suptitle("Embedding Diversity Collapse with Growing Text Input",
                 fontsize=14, y=1.02)
    fig.tight_layout()
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
                base = [t.item() for t in entry['lande_baseline']]
                # ax.plot(x, conv, marker='o', label='Conversation')
                ax.plot(x, intra, marker='o', label='Intra')
                ax.plot(x, base, marker='s', label='Baseline')
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
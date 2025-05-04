import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_expositive_panels(results, cols: int = 2,
                           base_figsize=(5, 3),
                           title_font=11,
                           label_font=9,
                           y_padding=0.5):
    """
    Traza un panel ordenado de curvas Lande-intra (una por texto).

    Parámetros
    ----------
    results : list[dict]
        Cada dict con claves:
          • 'lande_intra'   : lista[float]
          • 'n_paragraphs'  : int
          • 'text_name'     : str
    cols : int
        Número fijo de columnas (default = 2).
    base_figsize : tuple
        Tamaño (ancho, alto) de **cada** subgráfico en pulgadas.
    y_padding : float
        Margen extra (arriba/abajo) aplicado al rango global Y.
    """

    n_texts = len(results)
    rows    = math.ceil(n_texts / cols)

    # — rango global Y para uniformidad —
    all_vals = np.concatenate([r['lande_intra'] for r in results])
    y_min, y_max = all_vals.min() - y_padding, all_vals.max() + y_padding

    # — figura —
    fig_w  = base_figsize[0] * cols
    fig_h  = base_figsize[1] * rows
    fig, axes = plt.subplots(rows, cols,
                             figsize=(fig_w, fig_h),
                             sharex=False, sharey=False)
    axes = axes.flatten()

    for idx, (entry, ax) in enumerate(zip(results, axes)):
        xs   = range(1, entry['n_paragraphs'] + 1)
        ys   = entry['lande_intra']

        ax.plot(xs, ys, marker='o', linewidth=1.4)
        ax.set_ylim(y_min, y_max)
        ax.set_title(entry['text_name'], fontsize=title_font, pad=4)

        # Etiquetas X sólo en la última fila
        if idx // cols == rows - 1:
            ax.set_xlabel("Number of Paragraphs", fontsize=label_font)
        else:
            ax.set_xticklabels([])

        # Etiquetas Y sólo en la primera columna
        if idx % cols == 0:
            ax.set_ylabel("Lande Diversity Index", fontsize=label_font)
        else:
            ax.set_yticklabels([])

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)

    # Ejes vacíos (si sobran)
    for ax in axes[n_texts:]:
        ax.axis('off')

    fig.suptitle("Embedding Diversity Collapse with Growing Text Input",
                 fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

import numpy as np
from tueplots.constants.color.rgb import tue_red

def ridgeline_plot(
        X, Y, ax, y_labels, 
        hists=None, overlap=0.5, range_threshold=None,
        linewidth_density=1, linewidth_hist=1,
        alpha_density=1, alpha_hist=1,
        fill_density=False, fill_hist=False, fill_color_density=tue_red, fill_color_hist=tue_red,
        edge_color="black", line_color="black"
    ):
    """
    Create a ridgeline plot using Matplotlib.

    Parameters:
    - X: Matrix containing evaluation points as rows.
    - Y: Matrix containing corresponding values of the density.
    - ax: Matplotlib axis where the plot will be drawn.
    - y_label: Label for the y-axis.
    - fill: Whether to fill the areas between the lines.
    - fill-color: Color to fill between
    - fade: 1 is fully opaque while 0 is fully transparent
    - range_threshold: limit x-axis to only those values where y-axis is above the threshold
    - overlap: Overlap factor controlling the spacing between groups.
    """
    seq = enumerate(zip(X, Y, hists)) if hists else enumerate(zip(X, Y))
    for i, t in seq:
        x = t[0]
        y = t[1]
        h = t[2] if hists else None
        offset = i * (1 - overlap)
        if range_threshold:
            x = x[y > range_threshold]
            y = y[y > range_threshold]
        if h:
            probs, bins = h
            ax.bar(bins[:-1], probs, bottom=offset, width=np.diff(bins), zorder=len(Y) - i + 1, color=fill_color_hist, alpha=alpha_hist, linewidth=linewidth_hist, edgecolor=edge_color)
        if fill_density:
            ax.fill_between(x, np.ones_like(x) * offset, offset + y, alpha=alpha_density, zorder=len(Y) - i + 1, color=fill_color_density)
        ax.plot(x, y + offset, zorder=len(Y) - i + 1, alpha=alpha_density, linewidth=linewidth_density, color=line_color)

    for loc in ["top", "bottom", "left", "right"]:
        ax.spines[loc].set_visible(False)
    ax.set_yticks([i * ( 1 - overlap) for i in range(len(X))])
    ax.set_yticklabels(y_labels)
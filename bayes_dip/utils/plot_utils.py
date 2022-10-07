"""
Utilities for plotting.
"""
import matplotlib
import matplotlib.pyplot as plt

def configure_matplotlib():
    """
    Configure common matplotlib settings that should be shared by plotting script.
    """
    matplotlib.rc('text', usetex=True)
    matplotlib.rc('text.latex', preamble='\\usepackage{amsmath}')

def hex_to_rgb(value, alpha):
    """
    Convert a hex color string to a 4-tuple of float.
    """
    value = value.lstrip('#')
    lv = len(value)
    out = tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    out = [el / 255 for el in out] + [alpha]
    return tuple(out)

def plot_hist(  # pylint: disable=too-many-arguments
        data, label_list, title=None, ax=None, xlim=None, ylim=None, yscale='log',
        remove_ticks=False, color_list=None, alpha_list=None, hist_kwargs=None,
        hist_kwargs_per_data=None, legend_kwargs=None):
    """
    Plot a set of histograms.

    Returns
    -------
    ax : :class:`matplotlib.axes.Axes`
        Matplotlib axes.
    n_list : list of array
        For each element in ``data``, the ``n`` array as returned by :func:`matplotlib.pyplot.hist`.
    bins_list : list of array
        For each element in ``data``, the ``bins`` array as returned by
        :func:`matplotlib.pyplot.hist`.
    """
    # pylint: disable=too-many-locals
    if ax is None:
        _, ax = plt.subplots()
    if color_list is None:
        color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
    assert len(color_list) >= len(data)
    if alpha_list is None:
        alpha_list = [0.3] * len(data)
    assert len(alpha_list) >= len(data)
    hist_kwargs = hist_kwargs or {}
    hist_kwargs.setdefault('histtype', 'stepfilled')
    hist_kwargs.setdefault('bins', 25)
    hist_kwargs.setdefault('linewidth', 0.75)
    hist_kwargs.setdefault('alpha', 0.3)
    hist_kwargs.setdefault('linestyle', 'dashed')
    hist_kwargs.setdefault('density', True)
    hist_kwargs_per_data = hist_kwargs_per_data or {}
    hist_kwargs_per_data.setdefault('label', label_list)
    hist_kwargs_per_data.setdefault('zorder', range(3, 3 + len(data)))
    hist_kwargs_per_data.setdefault('facecolor',
            [hex_to_rgb(color, alpha=alpha) for color, alpha in zip(color_list, alpha_list)])
    hist_kwargs_per_data.setdefault('edgecolor',
            [hex_to_rgb(color, alpha=1) for color in color_list])
    assert all(len(v) >= len(data) for v in hist_kwargs_per_data.values())
    hist_kwargs_per_data_list = [
            dict(zip(hist_kwargs_per_data.keys(), v)) for v in zip(*hist_kwargs_per_data.values())]
    n_list = []
    bins_list = []
    for (el, hist_kwargs_overrides) in zip(data, hist_kwargs_per_data_list):
        hist_kwargs_merged = hist_kwargs.copy()
        hist_kwargs_merged.update(hist_kwargs_overrides)
        n, bins, _ = ax.hist(el.flatten(), **hist_kwargs_merged)
        n_list.append(n)
        bins_list.append(bins)
    ax.set_title(title)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(alpha=0.3)
    ax.legend(**(legend_kwargs or {}))
    ax.set_yscale(yscale)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('density', labelpad=2)
    ax.tick_params(labelbottom=not remove_ticks)
    return ax, n_list, bins_list

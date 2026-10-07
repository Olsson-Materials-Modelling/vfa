"""House style of the demo figures: Okabe-Ito palette, boxed axes with inward ticks, framed legends, panel letters."""
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

OKABE_ITO = {'black': '#000000', 'orange': '#E69F00', 'skyblue': '#56B4E9', 'bluishgreen': '#009E73', 'yellow': '#F0E442', 'blue': '#0072B2', 'vermillion': '#D55E00', 'purple': '#CC79A7', 'grey': '#999999'}
PALETTE = [OKABE_ITO[k] for k in ('blue', 'vermillion', 'bluishgreen', 'orange', 'purple', 'skyblue', 'yellow', 'grey', 'black')]; MARKERS = 'osD^v<>phX'   # colour is never the only encoding
INK, GREY = OKABE_ITO['black'], OKABE_ITO['grey']
ONE_COLUMN, TWO_COLUMN = 3.35, 7.35                                        # journal column widths (in)


def series(keys):
    """Colour and marker of each series, in the order given."""
    return {k: PALETTE[i % len(PALETTE)] for i, k in enumerate(keys)}, {k: MARKERS[i % len(MARKERS)] for i, k in enumerate(keys)}


def apply():
    """Install the rcParams."""
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 10, 'legend.fontsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
                         'lines.linewidth': 1.5, 'lines.markersize': 4.5, 'axes.linewidth': 0.8, 'axes.spines.top': True, 'axes.spines.right': True, 'axes.grid': False,
                         'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True, 'xtick.major.size': 4, 'ytick.major.size': 4, 'xtick.minor.size': 2, 'ytick.minor.size': 2,
                         'legend.frameon': True, 'legend.fancybox': False, 'legend.framealpha': 1.0, 'legend.facecolor': 'white', 'legend.edgecolor': INK, 'pdf.fonttype': 42, 'ps.fonttype': 42})


def style_axes(ax, minor_x=True, minor_y=True):
    """Inward ticks on all four spines; minor ticks on linear quantitative axes."""
    for side in ('top', 'right', 'bottom', 'left'): ax.spines[side].set_visible(True)
    ax.tick_params(which='both', direction='in', top=True, right=True)
    if minor_x and ax.get_xscale() == 'linear': ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    if minor_y and ax.get_yscale() == 'linear': ax.yaxis.set_minor_locator(AutoMinorLocator(5))


def legend(ax, **kw):
    """Framed legend."""
    leg = ax.legend(**kw); leg.get_frame().set_linewidth(0.8); return leg


def panel_label(ax, label, x=-0.16, y=1.03):
    """Bold panel letter outside the upper-left corner."""
    ax.text(x, y, label, transform=ax.transAxes, ha='right', va='bottom', fontsize=12, fontweight='bold', clip_on=False)


def save(fig, out, figures, name):
    """PDF and 300 dpi PNG to the results, 200 dpi PNG to the figures, marked as synthetic data."""
    fig.text(0.995, 0.995, 'synthetic data', ha='right', va='top', fontsize=7, color=GREY)
    fig.savefig(out / f'{name}.pdf', bbox_inches='tight'); fig.savefig(out / f'{name}.png', dpi=300, bbox_inches='tight'); fig.savefig(figures / f'{name}.png', dpi=200, bbox_inches='tight'); plt.close(fig); print('wrote', name)

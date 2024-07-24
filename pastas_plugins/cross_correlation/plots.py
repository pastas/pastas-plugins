import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import _plot_corr, plot_acf, plot_pacf

from pastas_plugins.cross_correlation.cross_correlation import ccf


def plot_corr(
    corr: pd.Series | pd.DataFrame,
    ax: plt.Axes | None = None,
    vlines_kwargs: dict | None = None,
    **kwargs,
):
    """Helper function for the statsmodels _plot_corr function.

    Parameters
    ----------
    corr : pd.Series or pd.DataFrame
        the correlation result to plot
    ax : plt.Axes, optional
        axes to plot on, by default None
    vlines_kwargs : dict, optional
        keyword arguments for the vlines function, by default None

    Returns
    -------
    plt.Axes
        axes with the plot
    """

    if ax is None:
        _, ax = plt.subplots(**kwargs)

    acf_x = corr.values if isinstance(corr, pd.Series) else corr.iloc[:, 0].values
    confint = corr.iloc[:, 1:3].values if isinstance(corr, pd.DataFrame) else None
    lags = corr.index.values
    vlines_kwargs = {} if vlines_kwargs is None else vlines_kwargs
    _plot_corr(
        ax=ax,
        title="",
        acf_x=acf_x,
        confint=confint,
        lags=lags,
        irregular=False,
        use_vlines=True,
        vlines_kwargs=vlines_kwargs,
    )
    return ax


def plot_ccf_overview(x, y, nlags=None, tmin=None, tmax=None, axes=None):
    """Plot an overview of the cross-correlation between two time series.

    Parameters
    ----------
    x : pd.Series
        Time series 1
    y : pd.Series
        Time series 2
    nlags : int, optional
        number of lags to return cross-correlations for, by default None which
        uses number of lags equal to len(x).
    tmin : str or pd.Timestamp, optional
        tmin for both time series, by default None
    tmax : str or pd.Timestamp, optional
        tmax for both time series, by default None
    axes : Axes mosaic, optional
        if provided, use axes from previous plot

    Returns
    -------
    axes : Axes mosaic
        return axes of subplots mosaic
    """
    if tmin is None:
        tmin = np.min([x.index[0], y.index[0]])
    if tmax is None:
        tmax = np.max([x.index[-1], y.index[-1]])

    x = x.loc[tmin:tmax]
    y = y.loc[tmin:tmax]

    if axes is None:
        mosaic = [
            ["x", "x", "norm", "norm"],
            ["y", "y", "norm", "norm"],
            ["x-acf", "y-acf", "ccf", "ccf"],
            ["x-pacf", "y-pacf", "ccf", "ccf"],
        ]

        fig, axes = plt.subplot_mosaic(mosaic, figsize=(16, 8))
        rescale_axes = False
        newaxes = True
        color1 = "C0"
        color2 = "C1"
    else:
        fig = axes["x"].figure
        rescale_axes = True
        newaxes = False
        color1 = "C2"
        color2 = "C3"

    # set names if not provided
    if x.name is None:
        x.name = "x"
    if y.name is None:
        y.name = "y"

    # plot time series
    axes["x"].plot(x.index, x, label=x.name, color=color1)
    axes["x"].legend(loc=(0, 1), frameon=False)
    axes["x"].set_ylabel("x")
    axes["x"].set_xlim(pd.Timestamp(tmin), pd.Timestamp(tmax))
    axes["y"].plot(y.index, y, label=y.name, c=color2)
    axes["y"].legend(loc=(0, 1), frameon=False)
    axes["y"].set_ylabel("y")
    axes["y"].set_xlim(pd.Timestamp(tmin), pd.Timestamp(tmax))

    # plot normalized series
    xnorm = (x - x.mean()) / x.std()
    ynorm = (y - y.mean()) / y.std()
    axes["norm"].plot(
        xnorm.index, xnorm, label=x.name + " (normalized)", alpha=0.7, color=color1
    )
    axes["norm"].plot(
        ynorm.index, ynorm, label=y.name + " (normalized)", alpha=0.7, color=color2
    )
    axes["norm"].legend(loc=(0, 1), frameon=False, ncol=2)
    axes["norm"].set_ylabel("normalized [-]")
    axes["norm"].set_xlim(pd.Timestamp(tmin), pd.Timestamp(tmax))
    handles, _ = axes["norm"].get_legend_handles_labels()

    # plot acf, pacf
    plot_acf(
        xnorm,
        ax=axes["x-acf"],
        color=color1,
        alpha=0.05,
        title="",
        zero=False,
        auto_ylims=True,
        vlines_kwargs={"color": "k"},
    )
    plot_acf(
        ynorm,
        ax=axes["y-acf"],
        color=color2,
        alpha=0.05,
        title="",
        zero=False,
        auto_ylims=True,
        vlines_kwargs={"color": "k"},
    )
    axes["x-acf"].set_xlim(left=0.0)
    axes["x-acf"].set_ylabel("ACF [-]")
    (p1,) = axes["x-acf"].plot([], [], marker="o", ls="none", color=color1)
    if not newaxes:
        leg = axes["x-acf"].get_legend()
        handles = leg.legend_handles
        labels = [t.get_text() for t in leg.get_texts()]
        handles += [p1]
        labels += [x.name]
    else:
        handles = [p1]
        labels = [x.name]

    axes["x-acf"].legend(handles, labels, loc=(0, 1), frameon=False)
    axes["y-acf"].set_xlim(left=0.0)
    axes["y-acf"].get_children()[3].set_facecolor(color2)
    (p2,) = axes["y-acf"].plot([], [], marker="o", ls="none", color=color2)
    if not newaxes:
        leg = axes["y-acf"].get_legend()
        handles = leg.legend_handles
        labels = [t.get_text() for t in leg.get_texts()]
        handles += [p2]
        labels += [y.name]
    else:
        handles = [p2]
        labels = [y.name]
    axes["y-acf"].legend(handles, labels, loc=(0, 1), frameon=False)

    plot_pacf(
        xnorm,
        method="ywm",
        ax=axes["x-pacf"],
        color=color1,
        alpha=0.05,
        title="",
        zero=False,
        auto_ylims=True,
        vlines_kwargs={"color": "k"},
    )
    plot_pacf(
        ynorm,
        method="ywm",
        ax=axes["y-pacf"],
        color=color2,
        alpha=0.05,
        title="",
        zero=False,
        auto_ylims=True,
        vlines_kwargs={"color": "k"},
    )
    axes["x-pacf"].set_xlim(left=0.0)
    axes["x-pacf"].set_ylabel("PACF [-]")
    axes["x-pacf"].set_xlabel("Lags")

    axes["y-pacf"].set_xlim(left=0.0)
    axes["y-pacf"].get_children()[3].set_facecolor("C1")
    axes["y-pacf"].set_xlabel("Lags")

    # ccf
    cc = ccf(x, y, nlags=nlags)
    axes["ccf"].bar(
        cc.index,
        cc,
        width=1.0,
        linewidth=0.5,
        alpha=0.5,
        label=f"CCF ({x.name}|{y.name})",
    )
    axes["ccf"].set_ylabel("CCF [-]")
    axes["ccf"].set_xlabel("Lags")
    axes["ccf"].legend(loc=(0, 1), frameon=False)
    axes["ccf"].set_xlim(left=0.0)

    share_x = [axes["x"], axes["y"], axes["norm"]]
    for i, iax in enumerate(share_x):
        if i < (len(share_x) - 1):
            iax.sharex(share_x[-1])

    # share_x = [axes["x-acf"], axes["x-pacf"], axes["y-acf"], axes["y-pacf"]]
    # for i, iax in enumerate(share_x):
    #     if i < (len(share_x) - 1):
    #         iax.sharex(share_x[-1])

    fig.tight_layout()
    fig.align_ylabels()

    if rescale_axes:
        for iax in axes.values():
            iax.autoscale()

    return axes

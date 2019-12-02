from typing import Tuple, List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_heatmap(df: pd.DataFrame,
                 columns: List[str],
                 method: str = 'pearson',
                 figsize: Tuple[int, int] = (10, 10),
                 title: str = None,
                 output_path: str = None):
    """Plots HeatMap for given columns from DataFrame."""
    correlation = df[columns].corr(method=method)
    mask = np.zeros_like(correlation)
    mask[np.triu_indices_from(mask)] = True
    colormap = sns.diverging_palette(220, 10, as_cmap=True)
    plt.figure(figsize=figsize)
    sns.heatmap(
        correlation,
        cmap=colormap,
        # mask=mask,
        annot=True,
        annot_kws={'size': 10, 'color': 'black'},
        fmt='.2f',
        xticklabels=correlation.columns,
        yticklabels=correlation.columns,
        vmin=np.min(correlation.values),
        linewidths=.5,
        cbar=False
    )
    if title:
        plt.title(title, fontsize=15, fontweight='black')
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.show()


def plot_countplot(df: pd.DataFrame,
                   column: str,
                   hue: str = None,
                   title: str = None,
                   figsize: Tuple[int, int] = (14, 8),
                   output_path: str = None):
    """Plots count plot."""
    plt.figure(figsize=figsize)
    sns.countplot(x=column, data=df, hue=hue)
    if title:
        plt.title(title)
    plt.xticks(rotation=45)
    if output_path:
        plt.savefig(output_path)
    plt.show()


def plot_pairwise_dependency(df: pd.DataFrame,
                             columns: List[str],
                             hue_column: str = None,
                             hue_dict: Dict = None,
                             markers: List[str] = None,
                             title: str = None,
                             output_path: str = None):
    """Plots pairwise dependency plots for given DataFrame and list of columns."""
    with sns.plotting_context("notebook", font_scale=1.4):
        title = title if title else 'Pairwise Plots'
        pp = sns.pairplot(
            df,
            vars=columns,
            kind='scatter',
            hue=hue_column,
            markers=markers,
            palette=hue_dict,
            height=5,
            plot_kws=dict(edgecolor="black", linewidth=0.5, s=50)
        )

        fig = pp.fig
        fig.subplots_adjust(top=0.93)
        fig.suptitle(title, fontsize=30, fontweight='black')
        for ax in fig.axes:
            ax.xaxis.label.set_size(25)
            ax.yaxis.label.set_size(25)
        if output_path:
            plt.savefig(output_path)
        plt.show()


def plot_normalised_barplot(df: pd.DataFrame,
                            x_column: str,
                            hue_column: str,
                            title: str = None,
                            figsize: Tuple[int, int] = (14, 8),
                            palette: Dict = None,
                            output_path: str = None):
    """
    Plots percentage of every category from hue_column in division of categories from x_column.
    :param df: Data Frame with variables to plot
    :param x_column: categorical column on the x-axis
    :param hue_column: categorical column for which we want to calculate percentages of categories
    :param title: the title of the plot
    :param figsize: the size of the figure
    :param palette: the dict for defining the colors
    :param output_path: the path, where we want to save the plot
    """
    x, y, hue = x_column, "percentage", hue_column
    normalized_df = (df[hue]
                     .groupby(df[x])
                     .value_counts(normalize=True)
                     .rename(y)
                     .reset_index()
                     )
    plt.figure(figsize=figsize)
    title = title if title else f'percentage of {hue_column} in {x_column}'
    plt.title(title)
    splot = sns.barplot(x=x, y=y, hue=hue, data=normalized_df, palette=palette, )
    for p in splot.patches:
        splot.annotate(
            format(p.get_height(), '.2f'),
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center',
            va='center',
            xytext=(0, 10),
            textcoords='offset points'
        )
    plt.legend(loc="upper right", bbox_to_anchor=(1, 1))
    plt.xticks(rotation=45)
    if output_path:
        plt.savefig(output_path)
    plt.show()


def make_stylish_df(df, fmt: str = "{:,.0f}", axis: int = 1):
    return (
        df.style.background_gradient(cmap='Spectral', low=0.5, high=0.75, axis=axis)
            .set_properties(**{'width': '8em'})
            .highlight_null('red').format(fmt)
    )

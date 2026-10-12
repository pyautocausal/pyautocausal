# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 16:17:07 2023

@author: Carlos Guevara
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

# get_ipython().run_line_magic('matplotlib', 'qt') # To aopen separate window
# get_ipython().run_line_magic('matplotlib', 'inline') # In line graph


def _mark_unavailable_intervals(results, ax):
    if (~np.isfinite(results['c'] * results['att_se'])).any():
        ax.text(.02, .98, 'Confidence intervals unavailable for some estimates',
                transform=ax.transAxes, va='top', fontsize='small')

def gplot(ssresults, ax, ylim=None, xlab=None, ylab=None, title="Group", xgap=1,
           legend=True, ref_line=0, theming=True):
    if ylab is None:
        ylab = 'ATT'
    
    # Use the correct column name - it might be 'year' or 'event_time'
    x_col = 'event_time' if 'event_time' in ssresults.columns else 'year'
    
    ssresults = ssresults[ssresults[x_col].notnull()].copy()
    ssresults.loc[:, x_col] = ssresults[x_col].astype(int)
    
    pre_points = ssresults.loc[ssresults['post'] == 0]
    post_points = ssresults.loc[ssresults['post'] == 1]
    
    ax.errorbar(pre_points[x_col], pre_points['att'], yerr=pre_points['c']*pre_points['att_se'],
                 fmt='o', markersize=5, color='#B7791F', ecolor='#B7791F', capsize=4, label='Pre-treatment')
    
    ax.errorbar(post_points[x_col], post_points['att'], yerr=post_points['c']*post_points['att_se'],
                 fmt='o', markersize=5, color='#087F8C', ecolor='#087F8C', capsize=4, label='Post-treatment')
    
    ax.set_ylim(ylim)
    ax.set_title(title)
    ax.set_xlabel(xlab)    
    ax.set_ylabel(ylab)    
    _mark_unavailable_intervals(ssresults, ax)

    handles, labels = ax.get_legend_handles_labels()    
    
    if ref_line is not None:
        ax.axhline(ref_line, linestyle='dashed', color='#64748B', linewidth=1)
    if theming:
        ax.set_facecolor('white')
        ax.set_title(title, color="#0F172A", fontsize=12, pad=12)
        ax.spines['bottom'].set_color('#CBD5E1')
        ax.spines['left'].set_color('#CBD5E1')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(colors='#475569', labelsize=10)
        ax.set_axisbelow(True)
        ax.grid(True, axis='y', color='#E2E8F0', linewidth=.7)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if not pre_points.empty and not post_points.empty:
            ax.legend(handles[0:2], labels[0:2], loc='lower center',fontsize='small', frameon=False, ncol=2, bbox_to_anchor=(0.5,-0.27))
        elif not pre_points.empty:
            ax.legend(handles[:1], labels[:1], loc='lower center',fontsize='small', frameon=False, ncol=2, bbox_to_anchor=(0.5,-0.27))
        elif not post_points.empty:
            ax.legend(handles[1:2], labels[1:2], loc='lower center',fontsize='small', frameon=False, ncol=2, bbox_to_anchor=(0.5,-0.27))
    if not legend:
        ax.legend().set_visible(False)
        
    return ax


def splot(ssresults, ax, ylim=None, xlab=None, ylab=None, title="Group",
          legend=True, ref_line=0, theming=True):
    
    if xlab is None:
        xlab = 'Group'
    if ylab is None:
        ylab = 'ATT'

    # Use the correct column name - it might be 'year' or 'event_time'
    x_col = 'event_time' if 'event_time' in ssresults.columns else 'year'
    ssresults[x_col] = ssresults[x_col].copy().astype(str)
    
    pre_points = ssresults.loc[ssresults['post'] == 0]
    post_points = ssresults.loc[ssresults['post'] == 1]
    
    ax.errorbar(pre_points[x_col], pre_points['att'], yerr=pre_points['c']*pre_points['att_se'],
                 fmt='o', markersize=5, color='#B7791F', ecolor='#B7791F', capsize=4, label='Pre-treatment')
    
    ax.errorbar(post_points[x_col], post_points['att'], yerr=post_points['c']*post_points['att_se'],
                 fmt='o', markersize=5, color='#087F8C', ecolor='#087F8C', capsize=4, label='Post-treatment')
    
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    _mark_unavailable_intervals(ssresults, ax)

    handles, labels = ax.get_legend_handles_labels()    
    
    if ylim is not None:
        ax.set_ylim(ylim)
    
    if ref_line is not None:
        ax.axhline(ref_line, linestyle='dashed', color='#64748B', linewidth=1)
    
    if theming:
        ax.set_facecolor('white')
        ax.set_title(title, color="#0F172A", fontsize=12, pad=12)
        ax.spines['bottom'].set_color('#CBD5E1')
        ax.spines['left'].set_color('#CBD5E1')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(colors='#475569', labelsize=10)
        ax.set_axisbelow(True)
        ax.grid(True, axis='y', color='#E2E8F0', linewidth=.7)
        if not pre_points.empty and not post_points.empty:
            ax.legend(handles[0:2], labels[0:2], loc='lower center',fontsize='small', frameon=False, ncol=2, bbox_to_anchor=(0.5,-0.27))
        elif not pre_points.empty:
            ax.legend(handles[:1], labels[:1], loc='lower center',fontsize='small', frameon=False, ncol=2, bbox_to_anchor=(0.5,-0.27))
        elif not post_points.empty:
            ax.legend(handles[1:2], labels[1:2], loc='lower center',fontsize='small', frameon=False, ncol=2, bbox_to_anchor=(0.5,-0.27))
            
    if not legend:
        ax.legend().set_visible(False)
    
    return ax

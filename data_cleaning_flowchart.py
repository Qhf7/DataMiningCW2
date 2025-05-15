#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate Data Cleaning Flowchart
This script uses matplotlib to create a simplified data cleaning flowchart with multiple columns
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import numpy as np
import matplotlib as mpl

# Set font properties for academic style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.unicode_minus'] = False

# Create canvas with landscape orientation for multiple columns
fig, ax = plt.subplots(figsize=(20, 16), facecolor='white')
fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

# Academic color palette
main_color = '#1A5276'  # Dark blue for main boxes
text_color = '#17202A'  # Dark color for text
accent_color = '#2874A6'  # Medium blue for accents
arrow_color = '#2E86C1'  # Color for arrows
bg_color = '#F8F9F9'    # Very light gray for backgrounds

# Define process steps - keeping only the main steps
steps = [
    "Data Import and Initial Exploration",
    "Handle Missing Values",
    "Convert Data Types",
    "Handle Outliers",
    "Remove Duplicates",
    "Clean Text Data",
    "Data Standardization and Normalization",
    "Feature Engineering",
    "Save Cleaned Data"
]

# Multi-column layout parameters - simplified
num_columns = 3  # Arrange steps in 3 columns
column_width = 6  # Width of each column
box_height = 1.4  # Made slightly taller for better visibility
box_width = 5.5  # Width of main step boxes
x_margin = 1.0  # Margin between columns
y_margin = 1.5  # Increased margin between steps in a column

# Draw background
background = patches.Rectangle(
    (0, 0),
    20, 16,
    facecolor=bg_color,
    edgecolor='none',
    alpha=0.5,
    zorder=0
)
ax.add_patch(background)

# Calculate positions for each step
positions = []
max_steps_per_column = (len(steps) + num_columns - 1) // num_columns  # Ceiling division
for i, step in enumerate(steps):
    # Determine column and position within column
    column = i // max_steps_per_column
    pos_in_column = i % max_steps_per_column
    
    # Calculate x and y coordinates with more even spacing
    x = 1 + column * (column_width + x_margin)
    y = 13 - pos_in_column * (box_height + y_margin)  # Start from top, move down
    
    # Store position information
    positions.append({
        'step': step, 
        'index': i,
        'x': x,
        'y': y,
        'column': column,
        'pos_in_column': pos_in_column
    })

# Add grid lines for academic paper look
for y in np.arange(1, 15, 1):
    ax.axhline(y=y, color='#E5E7E9', linestyle='-', alpha=0.3, zorder=0)

# Draw main steps only
for pos in positions:
    step = pos['step']
    i = pos['index']
    x = pos['x']
    y = pos['y']
    
    # Create main step box with improved appearance
    main_box = patches.FancyBboxPatch(
        (x, y),
        box_width, box_height,
        boxstyle=patches.BoxStyle("Round", pad=0.4, rounding_size=0.2),
        facecolor=main_color,
        edgecolor=accent_color,
        linewidth=1.5,
        zorder=2
    )
    ax.add_patch(main_box)
    
    # Add main step title with larger font
    ax.text(x + box_width/2, y + box_height/2, 
            f"Step {i+1}: {step}", 
            ha='center', va='center', 
            color='white', 
            fontsize=11,
            fontweight='bold',
            family='serif')

# Draw connecting arrows between steps
for i, pos in enumerate(positions):
    if i < len(positions) - 1:
        next_pos = positions[i+1]
        
        # If the next step is in the same column
        if pos['column'] == next_pos['column']:
            # Vertical arrow within column
            x_start = pos['x'] + box_width/2
            y_start = pos['y'] - 0.2
            
            x_end = next_pos['x'] + box_width/2
            y_end = next_pos['y'] + box_height + 0.2
            
            # Draw vertical arrow
            arrow = patches.FancyArrowPatch(
                (x_start, y_start),
                (x_end, y_end),
                arrowstyle='-|>',
                connectionstyle='arc3,rad=0.0',
                color=arrow_color,
                linewidth=1.5,
                mutation_scale=15,
                zorder=1
            )
            ax.add_patch(arrow)
            
            # Add arrow label
            mid_y = (y_start + y_end) / 2
            ax.text(
                x_start + 0.3, mid_y,
                "→",
                ha='left', va='center',
                fontsize=12,
                color=arrow_color,
                family='serif'
            )
        else:
            # Horizontal arrow between columns
            x_start = pos['x'] + box_width
            y_start = pos['y'] + box_height/2
            
            x_end = next_pos['x']
            y_end = next_pos['y'] + box_height/2
            
            # Draw curved arrow
            arrow = patches.FancyArrowPatch(
                (x_start, y_start),
                (x_end, y_end),
                arrowstyle='-|>',
                connectionstyle=f'arc3,rad=-0.3',
                color=arrow_color,
                linewidth=1.5,
                mutation_scale=15,
                zorder=1
            )
            ax.add_patch(arrow)
            
            # Add arrow label
            mid_x = (x_start + x_end) / 2
            mid_y = (y_start + y_end) / 2 - 0.5
            ax.text(
                mid_x, mid_y,
                "Next Step",
                ha='center', va='center',
                fontsize=8,
                style='italic',
                color=arrow_color,
                family='serif',
                bbox=dict(boxstyle="round", facecolor='white', alpha=0.9, pad=0.2)
            )

# Add title and subtitle
ax.text(
    10, 15.2,
    'DATA CLEANING WORKFLOW',
    ha='center', va='center',
    fontsize=20, fontweight='bold',
    family='serif',
    color=text_color
)

ax.text(
    10, 14.6,
    'A Systematic Process for Preparing Data for Analysis',
    ha='center', va='center',
    fontsize=12,
    style='italic',
    family='serif',
    color=text_color
)

# Add footer
ax.text(
    10, 0.3,
    "© Data Cleaning Methodology for Emotion Analysis Dataset",
    ha='center', va='center',
    fontsize=8,
    family='serif',
    style='italic',
    color='#555555'
)

# Set chart bounds
ax.set_xlim(0, 20)
ax.set_ylim(0, 16)
ax.axis('off')

# Save chart
plt.savefig('data_cleaning_flowchart_simplified.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Simplified data cleaning flowchart has been saved as data_cleaning_flowchart_simplified.png")
plt.close() 
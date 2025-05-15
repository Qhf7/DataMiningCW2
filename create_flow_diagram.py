#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.patheffects as path_effects

# Set font and style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans']
plt.rcParams['figure.figsize'] = (14, 18)
plt.rcParams['figure.dpi'] = 150

def draw_box(ax, x, y, width, height, title, items=None, color='#3498db', alpha=0.2, fontsize=10, 
             titlecolor='black', boxstyle="round,pad=0.5", title_offset=0):
    """Draw a box with a title and optional items"""
    # Draw the box
    rect = Rectangle((x, y), width, height, facecolor=color, alpha=alpha, 
                     edgecolor='gray', linewidth=1.5, zorder=1)
    ax.add_patch(rect)
    
    # Add the title
    title_y = y + height + title_offset
    text = ax.text(x + width/2, title_y, title, ha='center', va='bottom', 
                  fontsize=fontsize, fontweight='bold', color=titlecolor,
                  bbox=dict(facecolor='white', alpha=0.8, boxstyle=boxstyle, 
                            edgecolor='lightgray', pad=0.5))
    text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])
    
    # Add items if provided
    if items:
        item_height = height / (len(items) + 1)
        for i, item in enumerate(items):
            item_y = y + height - (i + 1) * item_height
            item_text = ax.text(x + width/2, item_y, item, ha='center', va='center', 
                              fontsize=fontsize-1, wrap=True)
            item_text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])
    
    return rect

def draw_arrow(ax, start, end, color='gray', style='simple', linewidth=1.5, connectionstyle="arc3,rad=0.1"):
    """Draw an arrow from start to end"""
    arrow = FancyArrowPatch(
        start, end, 
        connectionstyle=connectionstyle,
        arrowstyle=style, 
        color=color, 
        linewidth=linewidth,
        zorder=0
    )
    ax.add_patch(arrow)
    return arrow

def main():
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 18))
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1600)
    ax.axis('off')
    
    # Background
    ax.set_facecolor('#f8f9fa')
    
    # Title
    ax.text(500, 1550, 'Text Emotion Multi-label Classification Model', 
            ha='center', va='center', fontsize=20, fontweight='bold')
    
    # Main module boxes
    box_width, box_height = 800, 180
    box_x = 100
    
    # Text Input
    text_input_box = draw_box(ax, box_x, 1300, box_width, 80, 'Text Input', 
                            color='#85C1E9', fontsize=14)
    
    # Text Preprocessing Module
    preprocess_box = draw_box(ax, box_x, 1100, box_width, box_height, 'Text Preprocessing Module', 
                            color='#85C1E9', fontsize=14)
    
    # Draw preprocessing sub-boxes
    preproc_items = [
        'Data Cleaning (HTML/URL removal, Normalization)',
        'Tokenization (NLTK tools)',
        'Stopwords Filtering'
    ]
    sub_width, sub_height = 600, 40
    sub_x = box_x + 100
    for i, item in enumerate(preproc_items):
        sub_y = 1100 + box_height - (i + 1) * 50 - 20
        draw_box(ax, sub_x, sub_y, sub_width, sub_height, item, 
               color='#AED6F1', alpha=0.5, fontsize=11, title_offset=-20)
    
    # Parallel Feature Extraction
    feature_box = draw_box(ax, box_x, 750, box_width, box_height, 'Parallel Feature Extraction', 
                         color='#ABEBC6', fontsize=14)
    
    # BERT branch
    bert_box = draw_box(ax, box_x + 50, 850, 350, 160, 'BERT Deep Semantic Branch', 
                      color='#D4EFDF', fontsize=12)
    
    bert_items = [
        'RoBERTa Pre-trained Model',
        'Dynamic Word Vector Generation',
        'Contextual Embedding (256 dimensions)'
    ]
    for i, item in enumerate(bert_items):
        sub_y = 850 + 160 - (i + 1) * 40 - 20
        draw_box(ax, box_x + 80, sub_y, 290, 30, item, 
               color='#E8F8F5', alpha=0.5, fontsize=10, title_offset=-15)
    
    # Traditional Features branch
    trad_box = draw_box(ax, box_x + 420, 850, 350, 160, 'Traditional Feature Branch', 
                       color='#D4EFDF', fontsize=12)
    
    trad_items = [
        'TF-IDF Vectorization',
        'Emotion Lexicon Matching (Manual Features)',
        'Statistical Features (Sentence Length, Emotion Word Frequency)'
    ]
    for i, item in enumerate(trad_items):
        sub_y = 850 + 160 - (i + 1) * 40 - 20
        draw_box(ax, box_x + 450, sub_y, 290, 30, item, 
               color='#E8F8F5', alpha=0.5, fontsize=10, title_offset=-15)
    
    # Feature Fusion Layer
    fusion_box = draw_box(ax, box_x, 550, box_width, box_height, 'Feature Fusion Layer', 
                        color='#F9E79F', fontsize=14)
    
    # Multi-head Attention
    attn_box = draw_box(ax, box_x + 50, 630, 350, 130, 'Multi-head Attention Mechanism', 
                       color='#FCF3CF', fontsize=12)
    
    attn_items = [
        'Query: BERT Embedding',
        'Key: Traditional Features',
        'Value: Weighted Fusion'
    ]
    for i, item in enumerate(attn_items):
        sub_y = 630 + 130 - (i + 1) * 35 - 15
        draw_box(ax, box_x + 80, sub_y, 290, 25, item, 
               color='#FDEBD0', alpha=0.5, fontsize=10, title_offset=-15)
    
    # Feature Concatenation
    concat_box = draw_box(ax, box_x + 420, 630, 350, 130, 'Feature Concatenation', 
                        color='#FCF3CF', fontsize=12)
    
    # Multi-label Classification Layer
    classify_box = draw_box(ax, box_x, 350, box_width, box_height, 'Multi-label Classification Layer', 
                          color='#D2B4DE', fontsize=14)
    
    classify_items = [
        'Bidirectional LSTM (Capturing Sequential Dependencies)',
        'Attention Mechanism (Focus on Key Emotional Words)',
        '8 Sigmoid Output Nodes (Corresponding to 8 Emotion Classes)'
    ]
    for i, item in enumerate(classify_items):
        sub_y = 350 + box_height - (i + 1) * 50 - 20
        draw_box(ax, box_x + 100, sub_y, 600, 40, item, 
               color='#E8DAEF', alpha=0.5, fontsize=11, title_offset=-20)
    
    # Prediction Results Output
    output_box = draw_box(ax, box_x, 150, box_width, 120, 'Prediction Results Output', 
                        color='#F5B7B1', fontsize=14)
    
    output_items = [
        'Sadness: 0.92',
        'Loneliness: 0.85',
        'Suicidal Intent: 0.61',
        '... (Probabilities for other 5 emotion classes)'
    ]
    for i, item in enumerate(output_items):
        sub_y = 150 + 120 - (i + 1) * 25 - 10
        draw_box(ax, box_x + 100, sub_y, 600, 20, item, 
               color='#FADBD8', alpha=0.5, fontsize=10, title_offset=-15)
    
    # Arrows connecting the boxes
    # Text Input to Preprocessing
    draw_arrow(ax, (500, 1300), (500, 1280), 
             style='simple', color='#566573', linewidth=2, connectionstyle="arc3,rad=0")
    
    # Preprocessing to Feature Extraction
    draw_arrow(ax, (500, 1100), (500, 930), 
             style='simple', color='#566573', linewidth=2, connectionstyle="arc3,rad=0")
    
    # Feature Extraction to Feature Fusion
    draw_arrow(ax, (500, 750), (500, 730), 
             style='simple', color='#566573', linewidth=2, connectionstyle="arc3,rad=0")
    
    # Feature Fusion to Classification
    draw_arrow(ax, (500, 550), (500, 530), 
             style='simple', color='#566573', linewidth=2, connectionstyle="arc3,rad=0")
    
    # Classification to Output
    draw_arrow(ax, (500, 350), (500, 270), 
             style='simple', color='#566573', linewidth=2, connectionstyle="arc3,rad=0")
    
    # Save and show
    plt.tight_layout()
    plt.savefig('text_emotion_model_diagram.png', dpi=300, bbox_inches='tight')
    print("Model diagram saved as 'text_emotion_model_diagram.png'")
    plt.close()

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
'''
This sceipt will generate N random colors.
This is mostly helpful in case where we have
n items which needs to be uniquely identified using colors
Like in 
- image segmentation, segmentation color for each class
- charts
- object detection bounding box colors
'''
import seaborn as sns

class_names = ['class1', 'class2', 'class3']
number_of_colors = len(class_names)
# generate a list that contains one color for each class
colors = sns.color_palette(None, number_of_colors)

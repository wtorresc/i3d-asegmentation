
import pandas as pd
import pdb

"""
Functions for extracting corresponding video names from annotations
"""

def pair(x, y):
    """
    Cantor pairing function
    http://en.wikipedia.org/wiki/Pairing_function#Inverting_the_Cantor_pairing_function
    """
    return ((x + y) * (x + y + 1) / 2) + y

def depair(z):
    """
    Pairing function inverse
    """
    import math
    
    w = math.floor((math.sqrt(8 * z + 1) - 1)/2)
    t = (w**2 + w) / 2
    y = int(z - t)
    x = int(w - y)
    # assert z != pair(x, y, safe=False):
    return x, y

def get_video_and_frame(st):
    """
    Input: a cantor-paired (video, frame) in strung-float format 
    Output : video (string), frame (int)
    """
    csv = pd.read_csv('NewAnnotateds.csv')
    cvatjobid, frame = depair(int(float(st)))
    try:
        video_name = csv[csv['cvatjobid'] == cvatjobid]['videoname'].iloc[0]
    except:
        pdb.set_trace()
    return video_name, frame

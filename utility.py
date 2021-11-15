""" this script record useful functions """
import numpy as np

def cart2pol(point):
    """
    convert cartesian coordinate system to polar
    """
    x = point[0]
    y = point[1]
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return 360 - (phi / np.pi *180 + 180)


def points_sort_by_clockwise(points_input):
    """ this function sort 2d points by clockwise order"""
    """ the image center is subtracted to move the coordinate system """
    ## convert to numpy array
    points_input = np.array(points_input)
    width, height = np.mean(points_input[:,0]), np.mean(points_input[:,1])
    ## flip y axis
    points_input[:,1] = height*2 - points_input[:,1]
    ## centering
    points_input_centered = points_input - np.array([width , height ])
    ## sort by angle to the x negative axis clockwise
    points_input_sorted = sorted(points_input_centered, key = cart2pol)
    ## convert to numpy array
    points_input_sorted = np.array(points_input_sorted)
    ## un centering
    points_input_sorted = points_input_sorted + np.array([width , height ])
    ## flip y axis
    points_input_sorted[:,1] = height*2 - points_input_sorted[:,1]
    return points_input_sorted.astype(int)
    
def points_maximum_inscribed_rectangle(points_input):
    """ output points from 1.top left -->2.top right-->3.bottom right-->4.bottom left """
    points_input_x_start = np.max([points_input[0][0], points_input[3][0]]) 
    points_input_x_end = np.min([points_input[1][0], points_input[2][0]]) 
    points_input_y_start = np.min([points_input[0][1], points_input[1][1]]) 
    points_input_y_end = np.max([points_input[2][1], points_input[3][1]]) 
    #points_input_bl = 
    points_output = [[points_input_x_start, points_input_y_start], [points_input_x_end, points_input_y_start], [points_input_x_end, points_input_y_end], [points_input_x_start, points_input_y_end]]
    return np.array(points_output)
    

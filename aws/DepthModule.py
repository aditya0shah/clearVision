import cv2 as cv
import numpy as np


class DepthModule:
    MAX_DEPTH = 1 # TO DO

    def __init__(self, depth_map, mask):
        self.depth_map = depth_map
        self.mask = mask

        self.depth_map = self.depth_map * (self.mask > 0).astype(np.float32)
        self.split_depth_map()
    
    def split_depth_map(self):
        width = self.depth_map.shape[1]
        column_width = width // 3
        
        # Split into 3 columns
        self.left_column = self.depth_map[:, :column_width]
        self.center_column = self.depth_map[:, column_width:2*column_width]
        self.right_column = self.depth_map[:, 2*column_width:]

        return [self.left_column, self.center_column, self.right_column]
        
    def get_average(self, column, mask):
        return np.mean(column[mask])
    
    def get_buzz_values(self):
        return [1 - self.get_average(self.left_column, (self.left_column > 0) & (self.left_column < self.MAX_DEPTH))/self.MAX_DEPTH, 
                1 - self.get_average(self.center_column, (self.center_column > 0) & (self.center_column < self.MAX_DEPTH))/self.MAX_DEPTH, 
                1 - self.get_average(self.right_column, (self.right_column > 0) & (self.right_column < self.MAX_DEPTH))/self.MAX_DEPTH]
        
    def get_depth_map(self):
        return self.depth_map

    def get_depth_map_shape(self):
        return self.depth_map.shape

    def get_depth_map_size(self):
        return self.depth_map.size
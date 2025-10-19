import cv2 as cv
import numpy as np


class DepthModule:
    MAX_DEPTH = 1 # TO DO

    def __init__(self, depth_map, mask):
        self.depth_map = depth_map
        self.mask = mask
        
        for i in range(self.depth_map.shape[0]):
            for j in range(self.depth_map.shape[1]):
                if self.mask[i, j] == 0:
                    self.depth_map[i, j] = 0
        
        self.split_depth_map()
    
    def split_depth_map(self):

        width = self.depth_map.shape[1]
        column_width = width // 3
        
        # Split into 3 columns
        self.left_column = self.depth_map[:, :column_width]
        self.center_column = self.depth_map[:, column_width:2*column_width]
        self.right_column = self.depth_map[:, 2*column_width:]

        return [self.left_column, self.center_column, self.right_column]
        
    def get_average(self, column):
        count = 0
        sum = 0
        for i in range(column.shape[0]):
            for j in range(column.shape[1]):
                if column[i, j] > 0 and column[i, j] < self.MAX_DEPTH:
                    count += 1
                    sum += column[i, j]
        
        return sum / count
    
    def get_buzz_valus(self):
        return [self.get_average(self.left_column)/self.MAX_DEPTH, 
                self.get_average(self.center_column)/self.MAX_DEPTH, 
                self.get_average(self.right_column)/self.MAX_DEPTH]
        

        
    def get_depth_map(self):
        return self.depth_map

    def get_depth_map_shape(self):
        return self.depth_map.shape

    def get_depth_map_size(self):
        return self.depth_map.size
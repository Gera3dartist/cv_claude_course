from typing import Any
from scipy.spatial import Delaunay, Voronoi
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2 as cv
from enum import StrEnum


class PointSamplingType(StrEnum):
    PURE_RANDOM = 'pure_random'
    CONSTRAINED_RANDOM = 'constrained_random'
    


class AbstractTransformation:
    def __init__(self, width: int, height: int, density: int = 100):
        self.width = width
        self.height = height
        self.min_distance = min(self.width, self.height) // 30
        self.density = density
        
    
    def generate_points(self, method: str = PointSamplingType.CONSTRAINED_RANDOM, density: int = 100) -> np.ndarray:
        if method == PointSamplingType.CONSTRAINED_RANDOM:
            return self._constrained_random_points(density)
        else:
            raise ValueError(f"Not implemented: {method}")

    def _pure_random_points(self, num_points: int) -> np.ndarray:
        rows = np.random.randint(0, self.height, size=num_points)
        cols = np.random.randint(0, self.width, size=num_points)
        return np.column_stack((cols, rows))
    
    def _generate_random_point(self) -> np.ndarray:
        return np.array(
            [random.randint(0, self.width), random.randint(0, self.height)]
        )

    def _is_valid_point(self, existing_points: np.ndarray, candidate: np.ndarray) -> bool:
        if len(existing_points) == 0:
            return True
        distances = np.sqrt(np.sum((existing_points - candidate)**2, axis=1))
        return np.all(distances >= self.min_distance)

    
    def _constrained_random_points(self, num_points: int, retries: int = 10) -> np.ndarray:
        result = []
        result.append(self._generate_random_point())
        for _ in range(1, num_points):
            candidate = None
            attempts = retries
            while attempts > 0:
                candidate = self._generate_random_point()
                if self._is_valid_point(np.array(result), candidate):
                    result.append(candidate)
                    break
                else:
                    attempts -= 1
            if attempts == 0:
                break
        return np.array(result)

    def _create_triangulation(self, points: np.ndarray) -> np.ndarray:
        triangles = Delaunay(points)
        return triangles.simplices
    
    def triangular_abstraction(self, frame):
        abstract = np.zeros_like(frame)
        points = self.generate_points(density=self.density)
        points = points.astype(int)
        
        triangles = self._create_triangulation(points)
        for triangle in triangles:
            # find centroid
            poly = np.array([points[triangle[0]], points[triangle[1]], points[triangle[2]]])
            centroid = np.mean(poly, axis=0).astype(int)
            # sample color
            color = frame[centroid[1], centroid[0]]
            # fill polygons
            cv.fillPoly(abstract, [poly], color.tolist())
        return abstract
    

    def fractal_mirror_transform(self, frame: np.ndarray) -> np.ndarray:
        """Apply fractal-like mirror transformations"""
        result = frame.copy()
        
        # Create quadrants with different transformations
        h_mid, w_mid = self.height // 2, self.width // 2
        
        # Top-left: original
        # Top-right: horizontal flip
        result[:h_mid, w_mid:] = cv.flip(result[:h_mid, :w_mid], 1)
        
        # Bottom-left: vertical flip  
        result[h_mid:, :w_mid] = cv.flip(result[:h_mid, :w_mid], 0)
        
        # Bottom-right: both flips
        result[h_mid:, w_mid:] = cv.flip(result[:h_mid, :w_mid], -1)
        
        return result


    
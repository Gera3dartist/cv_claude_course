import cv2 as cv
import numpy as np


class DistanceMeasure:
    img: np.ndarray
    ref_points: np.ndarray
    ref_object_size: float

    # def __init__(self):
        # self.img = None
        # self.ref_points: np.ndarray | None = None
        # self.ref_object_size = None

    def load_image(self,  path: str) -> np.ndarray:
        self.img = cv.imread(path)
        return self.img

    def set_reference(self, point1: np.ndarray, point2: np.ndarray, reference_distance_mm: float) -> None:
        self.ref_points = np.ndarray([point1, point2])
        self.ref_object_size = reference_distance_mm

    def calculate_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """
        :point1 - starting point
        :point2 - end point
        returns - distance in mm
        real_obj_dist / real_ref_dist = pixel_obj_dist / pixel_ref_dist

        real_obj_dist = (pixel_obj_dist *  real_ref_dist) / pixel_ref_dist

        """
        pixel_obj_dist = np.linalg.norm(point2 - point1)
        pixel_ref_dist = np.linalg.norm(self.ref_points[1] - self.ref_points[0]) 
        real_obj_dist = float((pixel_obj_dist * self.ref_object_size) / pixel_ref_dist)
        return real_obj_dist

    def show_image(self, img: np.ndarray) -> None:
        _img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        cv.imshow('img', _img)
        cv.waitKey(500)

    def mouse_callback(self, event, x, y, flags, param) -> None:
        pass

    def calculate_are(self, points: list[tuple]) -> float:
        pass

def main():
    pass

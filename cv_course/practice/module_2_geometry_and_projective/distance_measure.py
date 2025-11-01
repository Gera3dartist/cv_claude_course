from enum import StrEnum
from functools import partial
import cv2 as cv
import numpy as np
from cv2.typing import Point

class OperationMode(StrEnum):
    REFERENCE = 'REFERENCE'
    MEASUREMENT = 'MEASUREMENT'
    VIEW = 'VIEW'


class DistanceMeasure:
    img: np.ndarray

    def __init__(self):
        self.user_input = []
        self.pointer = 0
        self.reference_set: bool = False
        self.ref_object_size: int = 0
        self.reference_points = []
        self.mode = OperationMode.VIEW

    def load_image(self,  path: str) -> np.ndarray:
        self.img = cv.imread(path)
        return self.img
    
    def add_points(self, x: int, y: int, container) -> None:
        container.append(np.array([x,y]))



    def set_mode(self, mode: OperationMode) -> None:
        self.mode = mode
    
    def handle_reference_mode(self, mask: np.ndarray):
        if self.reference_set:
            return
            
        if len(self.reference_points) < 2:
            print('peak 2 points')
            return
        reference_distance_mm = 0
        while reference_distance_mm == 0:
            inp = input('input reference size, positive number: ')
            reference_distance_mm = int(inp) if inp.isnumeric() else 0

        self.ref_object_size = reference_distance_mm
        self.reference_set = True
    
        self._put_text_to_point(mask, self.reference_points[1], text=f'{self.ref_object_size} mm')
        self.set_mode(OperationMode.VIEW)

    def handle_measurement_mode(self, mask: np.ndarray):
        
        if len(self.user_input) < 2:
            print('peak 2 points')
            return
        (point1, point2, *_) = self.user_input
        distance = self.calculate_distance(point1, point2)
        text = f'Measured: {distance} mm'
        print(text)
        self._put_text_to_point(mask, point2, text=text)
        self.user_input = []
        self.set_mode(OperationMode.VIEW)

    def calculate_distance(self, point1: np.ndarray, point2: np.ndarray, precision: int = 2) -> float:
        """
        :point1 - starting point
        :point2 - end point
        returns - distance in mm
        real_obj_dist / real_ref_dist = pixel_obj_dist / pixel_ref_dist

        real_obj_dist = (pixel_obj_dist *  real_ref_dist) / pixel_ref_dist

        """
        pixel_obj_dist = np.linalg.norm(point2 - point1)
        pixel_ref_dist = np.linalg.norm(self.reference_points[1] - self.reference_points[0]) 
        real_obj_dist = float((pixel_obj_dist * self.ref_object_size) / pixel_ref_dist)
        return round(real_obj_dist, precision)

    def populate_mask(self) -> np.ndarray:
        return np.zeros(self.img.shape[:2], dtype=np.uint8)
    
    def reset(self, mask: np.ndarray):
        mask.fill(1)
        self.user_input = []
        self.ref_object_size = 0
        self.reference_points = []
        self.reference_set = False
        self.user_input_set = False
        self.mode = OperationMode.VIEW


    def run(self):
        mask = self.populate_mask()
        print(f'shape of the mask: {mask.shape}')
        cv.namedWindow('image')
        callback = partial(self.mouse_callback, img=mask)
        cv.setMouseCallback('image', callback)
        # self.show_controls(mask)
        mask_3_channel = np.zeros_like(self.img)

        while True:
            self.show_controls(mask)
            mask_3_channel[:, :, 1] = mask
            result = cv.addWeighted(self.img, 1.0, mask_3_channel, 1.0, gamma=0)
            # result = cv.bitwise_or(self.img, self.img, mask)
            # result = mask
            cv.imshow('image', result)
         
            key = cv.waitKey(1) & 0xFF
            if key == 113:  # q - quit
                break
            elif key == 114:  # r - reset
                self.reset(mask)
            elif key == 115:  # s - set reference
                self.set_mode(OperationMode.REFERENCE)
            elif key == 109:  # m - measure
                if not self.reference_set:
                    print('set reference first')
                    continue
                self.set_mode(OperationMode.MEASUREMENT)
            elif key == 118:  # v - view info
                print(f'Ref points: {self.reference_points}')
                print(f'User Input: {self.user_input}')
                print(f'Current Mode: {self.mode}')
            elif self.mode == OperationMode.REFERENCE and not self.reference_set:
                self.handle_reference_mode(mask)
            elif self.mode == OperationMode.MEASUREMENT:
                self.handle_measurement_mode(mask)
            else:
                continue


                    
            # if key == 99:  # c - clear
            #     _img = self.img.copy()
        cv.destroyAllWindows()
    
    # visualizers
    def show_controls(self, img: np.ndarray):
        text = f'q - quit; r - reset; s - set reference, m - measure, current mode: {self.mode.value}'
        (height, width) = img.shape[:]
        org = (int(0.06 * width), int(0.95 * height),)
        self._display_text(img, text, org, color=(255, 255, 255))


    def _display_text(
            self, 
            img,
            text: str, 
            org: tuple[int, int],
            color: tuple[int, int, int] = (0, 255, 0),
            thickness: int = 4,
            font: int = cv.FONT_HERSHEY_SIMPLEX,
            font_scale: int = 2

    ):
        cv.putText(img, text, org, font, font_scale, color, thickness, cv.LINE_AA)
        
        
    def show_image(self, img: np.ndarray | None = None) -> None:
        _img = img or self.img
        # _img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        cv.imshow('img', _img)
        cv.waitKey(0)
        cv.destroyAllWindows()
    

    def mouse_callback(self, event, x, y, flags, param, img: np.ndarray | None = None) -> None:
        if self.mode not in (OperationMode.REFERENCE, OperationMode.MEASUREMENT):
            return

        img = img if img is not None else self.img
        # img = kwargs.get('img') or

        # print(f"{'img' in kwargs}")
        print('Inside a callback')
        if event == cv.EVENT_LBUTTONDOWN:
            container = self.user_input if self.mode == OperationMode.MEASUREMENT else self.reference_points
            self.add_points(x, y, container)
            draw_cross(img, x, y, color=(255, 255, 255))

            if len(container) == 2:
                self.display_line(img, container[0], container[1])
            elif len(self.user_input) % 2 == 0:
                # we can do measurements
                pass


            print('clicked mouse button!')

    def display_line(self, img: np.ndarray, point1: Point, point2: Point,  text: str | None = None,
                     text_offset: int = 20, 
                     line_color: tuple[int, int, int] =  (255, 255, 255), 
                     line_thinkness: int = 4):
        cv.line(img, point1, point2, line_color, line_thinkness)
        text_x = (point1[0] - point1[0]) + text_offset
        text_y = (point1[1] - point1[1])
        if text:
            self._display_text(img, text, (text_x, text_y))

    def display_refernce_text(
            self, 
            img: np.ndarray, 
            text_color: tuple[int, int, int] =  (255, 255, 255), 
            text_thikness: int = 4
        ):
        point2 = self.reference_points[1] 
        org = (point2[0] + int(0.01 * point2[0]), point2[1] - int(0.3 * point2[1]))
        text = f'{self.ref_object_size} mm'
        self._display_text(img, text, org, text_color, text_thikness)

    def _put_text_to_point(
            self, 
            img: np.ndarray,
            point: Point,
            text: str,
            text_color: tuple[int, int, int] =  (255, 255, 255), 
            text_thikness: int = 4
        ):
        org = (point[0] + int(0.01 * point[0]), point[1] - int(0.1 * point[1]))
        self._display_text(img, text, org, text_color, text_thikness)

def draw_cross(img: np.ndarray, x: int, y: int, 
                   line_size: int = 100, 
                   color: tuple[int, int, int] = (0,255,0),
                   thickness: int = 5
        ) -> None:
        print('Drawing a cross')
        center = np.array([x, y])
        hor_start = (center[0] - line_size // 2, center[1])
        hor_end = (center[0] + line_size // 2, center[1])
        ver_start = (center[0], center[1] - line_size // 2)
        ver_end = (center[0], center[1] + line_size // 2)
        # hor
        cv.line(img, hor_start, hor_end, color, thickness)
        # ver
        cv.line(img, ver_start, ver_end, color, thickness)

def run(_img: np.ndarray):
    img = np.zeros_like(_img)
    (h, w) = img.shape[:2]

    draw_cross(img,  w//2, h//2,)
    cv.namedWindow('image')
    
    while True:
        cv.imshow('image', img)
        key = cv.waitKey(1) & 0xFF
        if key == 113:  # q - quit
            break

        # if key == 99:  # c - clear
        #     _img = self.img.copy()
    cv.destroyAllWindows()

def main():
    processor = DistanceMeasure()
    processor.load_image('/Users/agerasymchuk/private_repo/cv_claude_course/cv_course/images/distance/IMG_3701.jpg')

    # cv.namedWindow('Distance Measure')
    # cv.setMouseCallback('Distance Measure', processor.mouse_callback)
    # processor.show_image()
    processor.run()
    # run(processor.img)


if __name__ == '__main__':
    main()
    

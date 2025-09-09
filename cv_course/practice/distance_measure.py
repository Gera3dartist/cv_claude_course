from functools import partial
import cv2 as cv
import numpy as np


class DistanceMeasure:
    img: np.ndarray
    ref_points: np.ndarray
    ref_object_size: float


    def __init__(self):
        self.user_input = []
        self.pointer = 0

    def load_image(self,  path: str) -> np.ndarray:
        self.img = cv.imread(path)
        return self.img
    
    def add_points(self, x: int, y: int) -> None:
        self.user_input.append(np.array([x,y]))


    def show_points(self):
        print(f'First point: {self.user_input[0]}')
        print(f'Second point: {self.user_input[1]}')


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


    def run(self):
        mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
        print(f'shape of the mask: {mask.shape}')
        cv.namedWindow('image')
        callback = partial(self.mouse_callback, img=mask)
        cv.setMouseCallback('image', callback)
        self.show_controls(mask)
        mask_3_channel = np.zeros_like(self.img)
        mask_3_channel[:, :, 1] = mask
        while True:
            mask_3_channel[:, :, 1] = mask
            result = cv.addWeighted(self.img, 1.0, mask_3_channel, 1.0, gamma=0)
            # result = cv.bitwise_or(self.img, self.img, mask)
            # result = mask
            cv.imshow('image', result)
            key = cv.waitKey(1) & 0xFF
            if key == 113:  # q - quit
                break
            if key == 118:  # v - visualise
                self.show_points()
            # if key == 99:  # c - clear
            #     _img = self.img.copy()
        cv.destroyAllWindows()
    
    # visualizers
    def show_controls(self, img: np.ndarray):
        text = 'q - quit; r - reset, start over'
        (height, width) = img.shape[:]
        org = (int(0.06 * width), int(0.95 * height),)
        self._display_text(img, text, org)


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

        img = img if img is not None else self.img
        # img = kwargs.get('img') or

        # print(f"{'img' in kwargs}")
        print('Inside a callback')
        if event == cv.EVENT_LBUTTONDOWN:
            self.add_points(x, y)
            draw_cross(img, x, y, color=(255, 255, 255))

            if len(self.user_input) == 2:
                cv.line(img, self.user_input[0], self.user_input[1], (255, 255, 255), 4)

            #     distance = self.calculate_distance(*self.user_input)

            print('clicked mouse button!')



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
    

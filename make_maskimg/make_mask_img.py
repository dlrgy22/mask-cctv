import cv2
import dlib
image = cv2.imread('nomask.jpeg')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor()

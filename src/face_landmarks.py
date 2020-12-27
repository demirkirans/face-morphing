import dlib
import cv2
import os

def showImage(image):
    cv2.imshow('Resized Window', image)  #Display image
    cv2.waitKey(0)  #Wait for keyboard button press
    cv2.destroyAllWindows()  #Exit window and destroy all windows using



def get_landmark_points(image):

    
    DATA_DIR = os.environ.get('PREDICTOR_DATA_DIR')

    
    #The detector object is used to detect the faces given in an image. It works generally better than 
    #OpenCV's default face detector
    detector = dlib.get_frontal_face_detector()

    #To predict the landark points given a face image, a shape predictor with a ready-to-use model
    #is created. The model can be found under "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    predictor = dlib.shape_predictor(os.path.join(DATA_DIR, 'shape_predictor_68_face_landmarks.dat'))

    #The predictor only works on grayscale images
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Use detector to find a list of rectangles containing the faces in the image. The rectangles are represented by
    #their xy coordinates
    rectangles = detector(gray)
    
    #points is a dlib structure that stores all facial landmarks points
    points = predictor(gray, rectangles[0])
    
    return points

def mark_landmark(image, points):    
    green_color = (0, 255, 0)
    radius = 1
    circle_thickness = 4
    
  
    
    for j in range(68):
        x = points.part(j).x # get x coordinate of point
        y = points.part(j).y # get y coordinate of point
        cv2.circle(image, (x, y), radius, green_color, circle_thickness)
        



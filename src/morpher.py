"""

Morph from source to destination face 

Usage:
    morpher.py (--src=<src_path> --dest=<dest_path>) [--fps=<frames_per_second>]


Options:
    -h, --help              Show this screen.
    --src=<src_imgpath>     Filepath to source image (.jpg, .jpeg, .png)
    --dest=<dest_imgpath>   Filepath to destination image (.jpg, .jpeg, .png)
    --fps=<fps>             Number frames per second for the video [default: 24]

"""

import cv2
import numpy as np
import os
import moviepy.editor as mpy
from docopt import docopt

import delaunay
import face_landmarks 


def verify_arguments(args):

    if not os.path.isfile(args["--src"]):
        print(" \"{src}\"  file does not exists. Check again".format(src = args["--src"]))
        exit(1)
    if not os.path.isfile(args["--dest"]):
        print(" \"{dest}\"  file does not exists. Check again".format(dest = args["--dest"]))
        exit(1)

def showImage(image):
    cv2.imshow('Resized Window', image)  #Display image
    cv2.waitKey(0)  #Wait for keyboard button press
    cv2.destroyAllWindows()  #Exit window and destroy all windows using


def make_homogeneous(triangle):
    homogeneous = np.array([ triangle[::2], triangle[1::2], [1, 1, 1] ]) #C
    
    # [x1,y1, x2, y2, x3, y3]
    # becomes
    # [x1,  x2,  x3]
    # [y1,  y2,  y3]
    # [1,   1,   1 ]
    
    return homogeneous

def calc_transform(triangle1, triangle2):
    source = make_homogeneous(triangle1).T
    
    #source
    # P point
    # [x1, y1, 1]
    # [x2, y2, 1]
    # [x3, y3, 1]
    
    target = triangle2
    
    #target
    # Q point
    # [u1, v1, u2, v2, u3, v3]
    
    Mtx = np.array([np.concatenate((source[0], np.zeros(3))), \
                    np.concatenate((np.zeros(3), source[0])), \
                    np.concatenate((source[1], np.zeros(3))), \
                    np.concatenate((np.zeros(3), source[1])), \
                    np.concatenate((source[2], np.zeros(3))), \
                    np.concatenate((np.zeros(3), source[2]))]) 
    
    #Mtx
    # [ x1, y1, 1,  0,  0,  0]
    # [ 0,  0,  0,  x1, y1, 1]
    # [ x2, y2, 1,  0,  0,  0]
    # [ 0,  0,  0,  x2, y2, 1]
    # [ x3, y3, 1,  0,  0,  0]
    # [ 0,  0,  0,  x3, y3, 1]
    
    coefs = np.matmul(np.linalg.pinv(Mtx), target) 
    
    #We got three points are three correspondences points. Mtx-1*q gives us coefficients needed for transformation
    
    #pseudo invese why? --> Mtx is rectangular so its not invertible. Instead Mtx^T * Mtx to make it square matrix
    
    Transform = np.array([coefs[:3], coefs[3:], [0, 0, 1]]) 
    
    #Transform
    # [a11, a12, a13]
    # [a21, a22, a23]
    # [0,   0,   1 ]
    #affine transformation
    
    
    return Transform

def vectorised_Bilinear(coordinates, target_img, size):
    #coordinates
    #coordinates[0]: [x0, x1, x2, x3, x4, ...]
    #coordinates[1]: [y0, y1, y2, y3, y4, ...]
    
    coordinates[0] = np.clip(coordinates[0], 0, size[0]-1)
    coordinates[1] = np.clip(coordinates[1], 0, size[1]-1)
    #keep values between 0 and 400. Clip ohter values
    
    
    lower = np.floor(coordinates).astype(np.uint32) 
    upper = np.ceil(coordinates).astype(np.uint32) 
    
    error = coordinates - lower
    resindual = 1 -error
    
    
    top_left = np.multiply(np.multiply(resindual[0], resindual[1]).reshape(coordinates.shape[1], 1), target_img[lower[0], lower[1], :])
    top_right = np.multiply(np.multiply(resindual[0], error[1]).reshape(coordinates.shape[1], 1), target_img[lower[0], upper[1], :])
    bot_left = np.multiply(np.multiply(error[0], resindual[1]).reshape(coordinates.shape[1], 1), target_img[upper[0], lower[1], :])
    bot_right = np.multiply(np.multiply(error[0], error[1]).reshape(coordinates.shape[1], 1), target_img[lower[0], lower[1], :])

    return np.uint8(np.round(top_left + top_right + bot_left + bot_right))

def image_morph(image1, image2, triangles1, triangles2, transforms, t):

    inter_image_1 = np.zeros(image1.shape).astype(np.uint8)
    inter_image_2 = np.zeros(image2.shape).astype(np.uint8)
    for i in range(len(transforms)):
        
        homo_inter_tri = (1 - t)*make_homogeneous(triangles1[i]) + t*make_homogeneous(triangles2[i]) 
        #corresponding triangle in inter image
        
        
        #homo_inter_tri
        # [a1, a2, a3]
        # [b1, b2, b3]
        # [1,  1,  1]
        
        polygon_mask = np.zeros(image1.shape[:2], dtype=np.uint8) #400 * 400 shape
        #polygon_mask is 400*400 empty array
        
        cv2.fillPoly(polygon_mask, [np.int32(np.round(homo_inter_tri[1::-1,:].T))], color=255) 
          
        #homo_inter_tri[1::-1,:].T
        #[b1, a1]
        #[b2, a2]
        #[b3, a3]
        
        
        seg = np.where(polygon_mask == 255) 
        
        #seg pixel locations of inter triangle area
        #seg[0] = [x0, x1, x2, x3, x4, x5, ...]
        #seg[1] = [y0, y1, y2, y3, y4, y5, ...]

        mask_points = np.vstack((seg[0], seg[1], np.ones(len(seg[0]))))
        
        #mask_points
        #[[x0, x1, x2, x3, x4, ...]
        # [y0, y1, y2, y3, y4, ... ]
        # [1,  1,  1,  1,  1,  ....]]

        inter_tri = homo_inter_tri[:2].flatten(order="F") 
        
        #inter_tri
        # [a1, b1, a2, b2, a3, b3]
        # it is a one dimensional array

        #compute mapping function
        inter_to_img1 = calc_transform(inter_tri, triangles1[i])
        inter_to_img2 = calc_transform(inter_tri, triangles2[i])

        mapped_to_img1 = np.matmul(inter_to_img1, mask_points)[:-1] 
        mapped_to_img2 = np.matmul(inter_to_img2, mask_points)[:-1]
        
        #mapped_to_image
        #[u0, u1,  u2, u3, u4, ...]
        #[v0, v1,  v2, v3, v4, ...]
        

        inter_image_1[seg[0], seg[1], :] = vectorised_Bilinear(mapped_to_img1, image1, inter_image_1.shape) 
        inter_image_2[seg[0], seg[1], :] = vectorised_Bilinear(mapped_to_img2, image2, inter_image_2.shape) 


    result = (1-t)*inter_image_1 + t*inter_image_2 
    
    return result.astype(np.uint8)

def get_morph_frames(image_1, img1_triangles, image_2, img2_triangles):

    img1_triangles = img1_triangles[:, [1,0,3,2,5,4]]
    img2_triangles = img2_triangles[:, [1,0,3,2,5,4]]

    Transforms = np.zeros((len(img1_triangles), 3, 3))

    for i in range(len(img1_triangles)):
        source = img1_triangles[i]
        target = img2_triangles[i]
        Transforms[i] = calc_transform(source, target) #A
        #compute transform matrix
    morphs = []

    for t in np.arange(0, 1.0001, 0.02): #B
        print("processing:\t", t*100, "%")
        morphs.append(image_morph(image_1, image_2, img1_triangles, img2_triangles, Transforms, t)[:, :, ::-1])

    return morphs

def face_morphing_video(image_1, image_2, fps=24):

    #get average of heights and widths of images, then resize them

    height = (image_1.shape[0] + image_2.shape[0]) // 2
    height = int(round(height, -2))

    width = (image_1.shape[1] + image_2.shape[1]) // 2
    width = int(round(width, -2))

    image_1 = cv2.resize(image_1, (int(height), int(width)))
    image_2 = cv2.resize(image_2, (int(height), int(width)))

    #get face landmark points of images
    point_image_1 = face_landmarks.get_landmark_points(image_1)
    point_image_2 = face_landmarks.get_landmark_points(image_2)


    img1_triangles, image_1_id_list= delaunay.get_triangles_edges_included(image_1, point_image_1)
    img2_triangles = delaunay.get_matched_triangles_for_second_image(point_image_2, img1_triangles, image_1_id_list)
    
    #we got images and their triangles
    
    morph_video = get_morph_frames(image_1, img1_triangles, image_2, img2_triangles)
    
    morphs_np_array = np.reshape(morph_video, (51, height,width,3))
    
    image_list = []

    for morphs in morphs_np_array:
        image_list.append(morphs)
    
    clip = mpy.ImageSequenceClip(image_list, fps = fps)
    clip.write_videofile('morph_video.gif', codec='gif', audio=True) #codec='libx264'


def main():
    args = docopt(__doc__)

    verify_arguments(args)

    #extract command line arguments
    image1_path = args["--src"]
    image2_path = args["--dest"]
    fps = args["--fps"]

    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    face_morphing_video(image1, image2, int(fps))

if __name__ == "__main__":
    main()


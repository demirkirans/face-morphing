import numpy as np
import cv2

def get_triangles(image, face_landmark_points):
    
    
    #The function creates an empty Delaunay subdivision where 2D points can be added 
    #Subdiv2D( Rect(top_left_x, top_left_y, width, height) )
    rect = (0, 0, image.shape[1], image.shape[0])
    subdiv = cv2.Subdiv2D( rect )

    for i in range(68):
        #each landmark point should be inserted into Subdiv2D object as a tuple
        x = face_landmark_points.part(i).x
        y = face_landmark_points.part(i).y
        #x = points_list[1].part(i).x # get x coordinate of point
        #y = points_list[1].part(i).y # get y coordinate of point


        #add the points to another list to check again

        subdiv.insert((x,y)) 

    img_triangles = subdiv.getTriangleList()
    
    return img_triangles


def get_triangles_edges_included(image, face_landmark_points):
    
    ID_list = []
    
    #The function creates an empty Delaunay subdivision where 2D points can be added 
    #Subdiv2D( Rect(top_left_x, top_left_y, width, height) )
    rect = (0, 0, image.shape[1], image.shape[0])
    subdiv = cv2.Subdiv2D( rect )

    for i in range(68):
        #each landmark point should be inserted into Subdiv2D object as a tuple
        x = face_landmark_points.part(i).x
        y = face_landmark_points.part(i).y
        #x = points_list[1].part(i).x # get x coordinate of point
        #y = points_list[1].part(i).y # get y coordinate of point

        #print("x point: " + str(x) + " y point: " + str(y));

        #add the points to another list to check again
        ID_list.append((x,y))

        subdiv.insert((x,y)) 

    #add points on edges

    top_left_point = (0,0)
    ID_list.append(top_left_point)
    subdiv.insert(top_left_point) #top left

    bottom_left_point = (0, image.shape[0]-1)
    ID_list.append(bottom_left_point)
    subdiv.insert(bottom_left_point) #bottom left

    medium_left_point = (0, image.shape[0]//2)
    ID_list.append(medium_left_point)
    subdiv.insert(medium_left_point) #medium left 

    top_right_point = (image.shape[1]-1, 0)
    ID_list.append(top_right_point)
    subdiv.insert(top_right_point) #top right

    bottom_right_point = (image.shape[1]-1, image.shape[0]-1)
    ID_list.append(bottom_right_point)
    subdiv.insert(bottom_right_point) #bottom right

    medium_right_point = (image.shape[1]-1, image.shape[0]//2)
    ID_list.append(medium_right_point)
    subdiv.insert(medium_right_point) #medium right

    medium_top_point = (image.shape[1]//2, 0)
    ID_list.append(medium_top_point)
    subdiv.insert(medium_top_point) #medium top

    medium_bottom_point = (image.shape[1]//2, image.shape[0]-1)
    ID_list.append(medium_bottom_point)
    subdiv.insert(medium_bottom_point) #medium bottom

    img_triangles = subdiv.getTriangleList()
    
    return (img_triangles, ID_list)



def find_index(id_list, point):
    #find the index for given point
    #in the list of points
    #id_list: [ (x1,y1), (x2,y2) , (x3,y3), ...]
    #point: (x, y)
    #return: index number of point
    index = 0
    for item in id_list:
        if point == item:
            return index
        index += 1
    return -1

def get_matched_triangles_for_second_image(points_for_second_image, image1_triangles, id_list):
    
    img2_triangles_list = [] #We will store the triangle points here
    
    for triangle in image1_triangles:
        pt = []
        pt.append((triangle[0], triangle[1])) #x and y coordinate for first point
        pt.append((triangle[2], triangle[3])) #x and y coordinate for second point
        pt.append((triangle[4], triangle[5])) #x and y coordinate for third point

        ids = [] # store 3 id for every loop 
        id1 = find_index(id_list, pt[0])
        ids.append(id1)
        id2 = find_index(id_list, pt[1])
        ids.append(id2)
        id3 = find_index(id_list, pt[2])
        ids.append(id3)

        #ids: [id1, id2, id3]

        for id_number in ids:
            #if the point is on edge, handle it in different way
            if id_number >= 68: # 0-67 are face points, others are edge points
                img2_triangles_list.append(id_list[id_number])
            else: #face landmark point
                p_x = points_for_second_image.part(id_number).x
                p_y = points_for_second_image.part(id_number).y
                img2_triangles_list.append((p_x, p_y))
            #We added {x1, y1, x2, y2, x3, y3} to the img2_triangles

    img2_triangles = np.reshape(img2_triangles_list, (142, 6))
    
    return img2_triangles

def draw_triangles_on_image(image, triangles):
    
    for triangle in triangles:
        pt = []
        pt.append((triangle[0], triangle[1]))
        pt.append((triangle[2], triangle[3]))
        pt.append((triangle[4], triangle[5])) 
        #if rectContainPoint(rect, pt[0]) and rectContainPoint(rect, pt[1]) and rectContainPoint(rect, pt[2]):
        cv2.line(image, pt[0], pt[1], (0,255,0), 1)
        cv2.line(image, pt[0], pt[2], (0,255,0), 1)
        cv2.line(image, pt[1], pt[2], (0,255,0), 1)
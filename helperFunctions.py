import numpy as np

# takes a rectangle object identified by dlib and convert it to a 
# tuple of points corresponding to the points of the rectangle
def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return (x, y, w, h)

#converts a shape object as defined by dlib into a numpy array
def shape_to_np(shape, dtype = "int"):
    coords = np.zeros((68, 2), dtype = dtype)

    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords

#takes in two 128D vectors as defined by dlib and subtracts them
#the smaller the difference the closer the faces
def compareFaces(knownFace, faceToCompare, tolerance):
    knownFace = np.array(knownFace)
    faceToCompare = np.array(faceToCompare)
    diff = np.linalg.norm(knownFace - faceToCompare, axis = 0)
    #print("Differernce in faces: " + diff)
    #if faces are the same return true
    return diff <= tolerance

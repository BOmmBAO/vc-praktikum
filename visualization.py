import numpy as np
import matplotlib.pyplot as plt

from mesh2img import *
from comparison import load_imgth


def visualMeshAndImg(points,img,para_img):

    alpha = para_img[0]
    beta = para_img[1]
    gamma = para_img[2]
    scale = para_img[3]
    size = para_img[4]
    offy = para_img[5]
    offx = para_img[6]

    obj_center = (np.max(points, axis=1) + np.min(points, axis=1))/2
    obj_center = obj_center.reshape(3,1)
    obj_temp = (np.max(points, axis=1) - np.min(points, axis=1))/2
    obj_max = np.around(np.sqrt(np.sum(obj_temp**2)))

    img_y,img_x = np.nonzero(img)
    img_points = np.zeros((3,len(img_x)))
    img_points[1,:] = (img_x - 100)*scale*obj_max/100 - offx
    img_points[0,:] = (img_y - 100)*scale*obj_max/100 - offy
    img_points[2,:] = np.zeros(len(img_x))
    m_rotBeta = np.array([[np.cos(np.pi+beta),0,np.sin(np.pi+beta)],
                          [0,1,0],
                          [-np.sin(np.pi+beta),0,np.cos(np.pi+beta)]])

    m_rotAlpha = np.array([[np.cos(alpha),-np.sin(alpha),0],
                           [np.sin(alpha),np.cos(alpha),0],
                           [0,0,1]])
    img_points_rot = np.dot(m_rotAlpha,np.dot(m_rotBeta,img_points))

    img_points_rot[0,:] += obj_center[0]
    img_points_rot[1,:] += obj_center[1]
    img_points_rot[2,:] += obj_center[2]

    linespace = np.linspace(0,100)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(linespace,0,0,marker='*')
    ax.scatter(0,linespace,0,marker='*')
    ax.scatter(0,0,linespace,marker='*')
            
    ax.scatter(points[0,:],points[1,:],points[2,:],c='#1f77b4',marker='o')
#    ax.scatter(img_points[0,:],img_points[1,:],img_points[2,:],c='#1f77b4',marker='o')
    ax.scatter(img_points_rot[0,:],img_points_rot[1,:],img_points_rot[2,:],c='#ff7f0e',marker='X')
        
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

    return 0

if __name__ == '__main__':
    points = load_stl2points()
    print(points.shape)
    img = load_imgth(size=200)
    print(np.sum(img))
    para_img = np.array([7.11922,4.90357,0,1.03822,200,1.51101,-53.2197])

    visualMeshAndImg(points,img,para_img)
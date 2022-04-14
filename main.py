
import cv2
import numpy as np
from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
import matplotlib as plt
from optimization import *
from mesh2img import *
from comparison import *
from visualization import *


def show_mesh():
    my_mesh = mesh.Mesh.from_file('./data/tube_phantom.stl')
    figure = pyplot.figure()
    axes = mplot3d.Axes3D(figure)
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(my_mesh.vectors))
    scale = my_mesh.points.flatten()
    axes.auto_scale_xyz(scale, scale, scale)
    pyplot.xlabel('xlabel')
    pyplot.ylabel('ylabel')
    pyplot.show()


def show_img(threshold):
    img = cv2.imread('./data/xray_phantom.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(gray.shape)
    _, gray_th = cv2.threshold(gray, threshold, 255, 0)
    gray_th = cv.resize(gray_th,(300,300))
    cv2.imshow('test', gray_th)
    cv2.waitKey(0)


if __name__ == '__main__':
    #show_img(200)
    #show_mesh()
    points = load_stl2points()
    imgth = load_imgth(size=200)
    cf, cb, _,_,_ = op_v5(points,imgth)
    showOptimization(imgth,cf,cb)
    #para_img = np.array([7.11922,4.90357,0,1.03822,200,1.51101,-53.2197])
    #visualMeshAndImg(points,imgth,para_img)
from math import gamma
import cv2 as cv
from cv2 import INTER_NEAREST
import numpy as np

from mesh2img import *
from comparison import *

def showall():
    canvas_all = np.zeros((600,600))
    points = load_stl2points()
    for i in range(6):
        for j in range(6):
            img_temp = points2canvas(points,i*np.pi/6,j*np.pi/6,0,scale=1.2,size=100)
            canvas_all[j*100:j*100+100,i*100:i*100+100] = img_temp
    cv.imshow('canvasall',canvas_all)
    cv.waitKey(0)


'''
v3: find out rough offset; firstly search with pi/6, then search with pi/12

'''

def op_v3(points,img):
    img_e = averageCenter(img)
    p_group = []
    for i in range(6):
        for j in range(6):
            canv_temp = points2canvas(points,i*np.pi/6,j*np.pi/6,0)
            canv_e = averageCenter(canv_temp)
            o = np.array(img_e - canv_e)
            canv0_n = points2canvas(points,i*np.pi/6,j*np.pi/6,offsety=o[0],offsetx=o[1])
            dloss = dice_loss(canv0_n,img)
            p_group.append([dloss,i,j,o[0],o[1]])
    d_group = np.array(p_group)
    d_argmin = np.argmin(d_group[:,0])
    dmin = d_group[d_argmin,:]
#    print(dmin)
    p2_group = []
    for i in range(-3,3):
        for j in range(-3,3):
            canv0_n = points2canvas(points,dmin[1]*np.pi/6+i*np.pi/12,dmin[2]*np.pi/6+i*np.pi/12,offsety=dmin[3],offsetx=dmin[4])
            dloss = dice_loss(canv0_n,img)
            print(dloss,i,j)
            p2_group.append([dloss,i,j])
    d2_group = np.array(p2_group)
    d2_argmin = np.argmin(d2_group[:,0])
    d2min = d2_group[d2_argmin,:]
    canv_mid = points2canvas(points,dmin[1]*np.pi/6,dmin[2]*np.pi/6,offsety=dmin[3],offsetx=dmin[4])
    canv_final = points2canvas(points,dmin[1]*np.pi/6+d2min[1]*np.pi/12,dmin[2]*np.pi/6+d2min[1]*np.pi/12,offsety=dmin[3],offsetx=dmin[4])
    cv.imshow('orig',img)
    cv.imshow('mid',canv_mid)
    cv.imshow('final',canv_final)
    cv.waitKey()
    
'''
v4: improve sphere sample

'''

def op_v4(points,img):
    img_ac = averageCenter(img)
#    print(img_ac)
    sample_number = 500
    u = np.random.rand(sample_number)
    v = np.random.rand(sample_number)
    alpha_rand = np.pi*u
    phi_rand = np.arccos(2*v-1)
    beta_rand = phi_rand
    p_group = []
    for i in range(sample_number):
        canv_temp = points2canvas(points,alpha=alpha_rand[i],beta=beta_rand[i],size=200)
        ga, sc, dl = compareab(img,canv_temp,img_ac)
        p_group.append([dl,ga,sc])
    d_group = np.array(p_group)
    d_argmin = np.argmin(d_group[:,0])
#    print(d_argmin)
#    print(d_group[d_argmin,:])
    canv_mid = points2canvas(points,alpha=alpha_rand[d_argmin],beta=beta_rand[d_argmin],
                             gamma=np.pi*d_group[d_argmin,1]/6,size=200)
    canv_mid2 = points2canvas(points,alpha=(alpha_rand[d_argmin]+np.pi),beta=(beta_rand[d_argmin]+np.pi),
                              gamma=(np.pi-np.pi*d_group[d_argmin,1]/6),size=200)

    cv.imshow('orig',img)
    cv.imshow('mid',canv_mid)
    cv.imshow('mid2',canv_mid2)
    cv.waitKey()

'''
v5: more detail search using projected image 

'''

def op_v5(points,img,backcheck=1):
    img_ac = averageCenter(img)

    #first round
    sample_number = 200
    u = np.random.rand(sample_number)
    v = np.random.rand(sample_number)
    alpha_rand = 2*np.pi*u
    phi_rand = np.arccos(2*v-1)
    beta_rand = phi_rand
    p_group = []
    for i in range(sample_number):
        canv_temp = points2canvas(points,alpha=alpha_rand[i],beta=beta_rand[i],size=200)
        ga, sc, dl = compareab(img,canv_temp,img_ac)
        p_group.append([dl,ga,sc])
    d_group = np.array(p_group)
    d_argmin = np.argmin(d_group[:,0])

    #second round
    sample_number = 20
    u = np.random.rand(sample_number)*0.2 + u[d_argmin]
    v = np.random.rand(sample_number)*0.2 + v[d_argmin]
    alpha_rand = 2*np.pi*u
    phi_rand = np.arccos(2*v-1)
    beta_rand = phi_rand
    p_group2 = []
    for i in range(sample_number):
        canv_temp = points2canvas(points,alpha=alpha_rand[i],beta=beta_rand[i],size=200)
        ga, sc, dl = compareab(img,canv_temp,img_ac)
        p_group2.append([dl,ga,sc])
    d_group2 = np.array(p_group2)
    d_argmin2 = np.argmin(d_group2[:,0])
      
    #third round
    p_group3 = []
    canv_0 = points2canvas(points,alpha=alpha_rand[d_argmin2],beta=beta_rand[d_argmin2],
                           gamma=np.pi*d_group2[d_argmin2,1]/6,size=200)
    scale_rate = np.sqrt(np.sum(img)/np.sum(canv_0))
    for i in range(9):
        scale_rate_itr = scale_rate+0.01*(i-4)
        canv_1 = points2canvas(points,alpha=alpha_rand[d_argmin2],beta=beta_rand[d_argmin2],
                               gamma=np.pi*d_group2[d_argmin2,1]/6,scale=scale_rate_itr,size=200)
        canv_ac = averageCenter(canv_1)
        o = np.array(img_ac - canv_ac)
        for m in range(9):
            for n in range(9):
                o0 = o[0] + m - 4
                o1 = o[1] + n - 4
                canv_2 = points2canvas(points,alpha_rand[d_argmin2],beta_rand[d_argmin2],
                                       np.pi*d_group2[d_argmin2,1]/6,scale=scale_rate_itr,size=200,
                                       offsety=o0,offsetx=o1)
                dl = dice_loss(img,canv_2)
                p_group3.append([dl,scale_rate_itr,o0,o1])
    d_group3 = np.array(p_group3)
    d_argmin3 = np.argmin(d_group3[:,0])
    print(d_group3[d_argmin3])
    para_final = np.array([alpha_rand[d_argmin2],beta_rand[d_argmin2],
                           np.pi*d_group2[d_argmin2,1]/6,d_group3[d_argmin3,1],200,
                           d_group3[d_argmin3,2],d_group3[d_argmin3,3]])
    canv_final = points2canvas(points,alpha_rand[d_argmin2],beta_rand[d_argmin2],
                               np.pi*d_group2[d_argmin2,1]/6,scale=d_group3[d_argmin3,1],size=200,
                               offsety=d_group3[d_argmin3,2],offsetx=d_group3[d_argmin3,3])

    # backcheck
    if backcheck==1:
        p_group4 = []
        canv_0 = points2canvas(points,(np.pi+alpha_rand[d_argmin2]),(np.pi+beta_rand[d_argmin2]),
                               (np.pi+np.pi*d_group2[d_argmin2,1]/6),size=200)
        scale_rate = np.sqrt(np.sum(img)/np.sum(canv_0))
        for i in range(9):
            scale_rate_itr = scale_rate+0.01*(i-4)
            canv_1 = points2canvas(points,(np.pi+alpha_rand[d_argmin2]),(np.pi+beta_rand[d_argmin2]),
                                   (np.pi+np.pi*d_group2[d_argmin2,1]/6),scale=scale_rate_itr,size=200)
            canv_ac = averageCenter(canv_1)
            o = np.array(img_ac - canv_ac)
            for m in range(9):
                for n in range(9):
                    o0 = o[0] + m - 4
                    o1 = o[1] + n - 4
                    canv_2 = points2canvas(points,alpha_rand[d_argmin2],beta_rand[d_argmin2],
                                           np.pi*d_group2[d_argmin2,1]/6,scale=scale_rate_itr,size=200,
                                           offsety=o0,offsetx=o1)
                    dl = dice_loss(img,canv_2)
                    p_group4.append([dl,scale_rate_itr,o0,o1])
        d_group4 = np.array(p_group4)
        d_argmin4 = np.argmin(d_group4[:,0])
        print(d_group4[d_argmin4])
        para_backcheck = np.array([(np.pi+alpha_rand[d_argmin2]),(np.pi+beta_rand[d_argmin2]),
                                   (np.pi-np.pi*d_group2[d_argmin2,1]/6),d_group4[d_argmin4,1],200,
                                   d_group4[d_argmin4,2],d_group4[d_argmin4,3]])
        canv_backcheck = points2canvas(points,(np.pi+alpha_rand[d_argmin2]),(np.pi+beta_rand[d_argmin2]),
                                       (np.pi-np.pi*d_group2[d_argmin2,1]/6),scale=d_group4[d_argmin4,1],size=200,
                                       offsety=d_group4[d_argmin4,2],offsetx=d_group4[d_argmin4,3])
    else:
        canv_backcheck = canv_final
        para_backcheck = para_final
    
    backcheck_better = d_group4[d_argmin4]<d_group3[d_argmin3]
    return canv_final, canv_backcheck, backcheck_better, para_final, para_backcheck

def showOptimization(img,c_final,c_bc):
    cv.imshow('orig',img)
    cv.imshow('mid',c_final)
    cv.imshow('mid2',c_bc)
    cv.waitKey()


if __name__ == '__main__':
    imgth = load_imgth(size=200)
    points = load_stl2points()
    cf, cb, _,_,_ = op_v5(points,imgth)

    showOptimization(imgth,cf,cb)

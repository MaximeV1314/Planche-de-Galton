import random as rd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import glob
import matplotlib.cm as cm
import time

mpl.use('Agg')

start_time = time.time()

N = 80             #nombre de particules
N_e = 3             #nombre d'étage de la plache de Galton
N_p = 3             #paramètre de voisinnage entre chaque particule
D = 0.8             #espace entre les étages
d = (N_e + 1) * D   #dimension du fig
coef = 1.5

#coordonées du premier clou (haut du triangle)
x0 = 0
y0 = 2 * D

m = 1   #masse

#paramètres WCA inter-particule
r0_p = D/10
r_cut_p = 2.5 * r0_p
E0_p = 1000

#paramètres WCA clou
r0_c = D/7
r_cut_c = 2.5 * r0_c
E0_c = 1000

#paramètres WCA compartiment
r0_o = D/16
r_cut_o = 2.5 * r0_o
E0_o = 1000

g = 9.81    #accélération de pesanteur
K = 3       #coefficient de frottement

dt = 0.001
tf = 90

pas_img = 20
digit = 4

def strcuture():
    """structure"""
    barr = []    #barres
    clous = []   #clous

    n = 20      #nombre de particules composant les compartiments

    barr.append(np.array([np.ones((n,)) * x0 - (int(N_e/2) + 1 ) * 2 * D/coef, np.linspace(y0 - N_e * D, y0 - (N_e+2) * D, n) ]))                                        #mur gauche

    for i in range(N_e):
        barr.append(np.array([np.linspace(x0 - (int(N_e/2) - i) * 2 * D/coef, x0 - (int(N_e/2) - i + 1) * 2 * D/coef, int(n/2)), (y0 - (N_e + 2) * D) * np.ones((int(n/2),)) ]))  #sol
        barr.append(np.array([np.ones((n,)) * x0 - (int(N_e/2) - i) * 2 * D / coef, np.linspace(y0 - N_e * D, y0 - (N_e+2) * D, n) ]))                                     #barre i
        clous.append(np.array([np.linspace(x0 - i * D/coef, x0 + i * D/coef, i + 1), np.ones((i+1,)) * (y0 - i * D)]))                                                #étage i

    barr.append(np.array([np.linspace(x0 - (int(N_e/2) - N_e+1) * 2 * D/coef, x0 - (int(N_e/2) - N_e) * 2 * D/coef, int(n/2)), (y0 - (N_e + 2) * D) * np.ones((int(n/2),)) ]))    #dernier sol (droite)
    barr.append(np.array([np.ones((n,)) * x0 - (int(N_e/2) - N_e ) * 2 * D/coef, np.linspace(y0 - N_e * D, y0 - (N_e+2) * D, n) ]))                                      #dernier mur (droite)

    barr_u = []
    clou_u = []

    N_1 = len(barr)
    barr_u = [ [] for _ in range(N_e + 1) ]

    for i in range(0, 2 * (N_e + 1), 2):
        barr_u[int(i/2)] = np.concatenate(barr[i : i+3], axis = 1)

    for i in range(2):
        pos2 = []

        for vec in clous:
            pos2 = np.concatenate((pos2, vec[i]))
        clou_u.append(pos2)

    return barr_u, np.array(clou_u), barr


def pos_ini(t):
    """initialisation des positions"""
    vec = []

    for i in range(N):
        vec.append(np.zeros((4, t.shape[0])))
        x1 = rd.uniform(x0 - d/55, x0 + d/55)
        vec[i][:, 0] = np.array([x1, y0 + 3 * (i + 1), 0, 0])

    return vec

def mini(L):
    """renvoie une liste dans la valeur minimale de la liste intial"""

    k = 0
    c = L[0]
    for i in range(1, len(L)):
        if L[i] < c :
            k = i
            c = L[i]

    return k

zero_arr = np.zeros((6,))
def a(vec_n, vec_c, vec_a, k, c):
    """accélération potentiel WCA entre les paricules et les obstacles"""

    Fx_p, Fy_p, Fx_c, Fy_c, Fx_o, Fy_o = zero_arr   #initialisation des forces

    Y = vec_n[1]
    X = vec_n[0]
    if  Y < y0 + D:

        if  Y < y0 - (N_e - 1) * D and len(vec_c) > 1:
            #force intra-particulaire dans un compartiment
            X1 = vec_c[:, 0]
            Y1 = vec_c[:, 1]

            OM_p = (( X- X1)**2 + ( Y - Y1)**2)**(1/2)
            j = mini(OM_p)
            OM_p = OM_p[np.arange(len(OM_p))!=j]

            THETA_p = np.arctan2(Y1 - Y, X1 -  X)
            THETA_p = THETA_p[np.arange(len(THETA_p))!=j]

            F_p = 6 * E0_p/r0_p * (2*(OM_p/r0_p)**(-13) - (OM_p/r0_p)**(-7)) * (OM_p<r_cut_p)
            Fx_p = np.sum(F_p * np.cos(THETA_p))
            Fy_p = np.sum(F_p * np.sin(THETA_p))

        if  vec_n[1] > y0 - (N_e - 1) * D :
            #force intra-particulaire hors compartiment
            X1 = vec_a[:, 0]
            Y1 = vec_a[:, 1]

            OM_p = (( X- X1)**2 + ( Y - Y1)**2)**(1/2)
            OM_p = OM_p[np.arange(len(OM_p))!= N_p]

            THETA_p = np.arctan2(Y1 - Y, X1 -  X)
            THETA_p = THETA_p[np.arange(len(THETA_p))!= N_p]

            F_p = 6 * E0_p/r0_p * (2*(OM_p/r0_p)**(-13) - (OM_p/r0_p)**(-7)) * (OM_p<r_cut_p)
            Fx_p = np.sum(F_p * np.cos(THETA_p))
            Fy_p = np.sum(F_p * np.sin(THETA_p))

        if Y > y0 - (N_e+3) * D :
            #force des clous
            OM_c = (( X - structure2[0])**2 + ( Y - structure2[1])**2)**(1/2)
            THETA_c = np.arctan2(structure2[1] -  Y, structure2[0] - X)

            F_c = 6 * E0_c/r0_c * (2*(OM_c/r0_c)**(-13) - (OM_c/r0_c)**(-7)) * (OM_c<r_cut_c)
            Fx_c = np.sum(F_c * np.cos(THETA_c))
            Fy_c = np.sum(F_c * np.sin(THETA_c))

        if  Y < y0 - (N_e - 1) * D :
            #force des obstacles
            OM_o = (( X - structure1[c][0])**2 + ( Y - structure1[c][1])**2)**(1/2)
            THETA_o = np.arctan2(structure1[c][1] -  Y, structure1[c][0] - X)

            F_o = 6 * E0_o/r0_o * (2*(OM_o/r0_o)**(-13) - (OM_o/r0_o)**(-7)) * (OM_o<r_cut_o)
            Fx_o = np.sum(F_o * np.cos(THETA_o))
            Fy_o = np.sum(F_o * np.sin(THETA_o))

    return np.array([vec_n[2], vec_n[3], Fx_p + Fx_c + Fx_o - K * vec_n[2], Fy_p + Fy_c + Fy_o - g - K * vec_n[3]])

def rk4(derivee, step, fin):
    """Fonction Runge-Kutta ordre 4. Prend en entré la fonction différentielle,
    le pas, le temps de l'expérience et le vecteur initial.
    Renvoie un tableau de toutes les positions et un tableau de temps."""

    t = np.arange(0,fin,step)

    vec = np.array(pos_ini(t))
    print("Initial position sucessed")
    l_t = t.shape[0] - 1

    comp_vid_p = [ [] for _ in range(N_e + 1) ]     #initialisation liste comportant les positions des particules de chaque compartiment
    comp_vid_num = np.zeros((N,), dtype = 'int64')  #initialisation liste comportant les numéros des particules de chaque compartiment

    Yc = y0 - (N_e - 1) * D                         #ordonnée haute des compartiment

    for i in range(l_t):

        comp_full_p = list(comp_vid_p)
        comp_full_num = list(comp_vid_num)

        comp_vid_p = [ [] for _ in range(N_e + 1) ]
        comp_vid_num = np.zeros((N,), dtype = 'int64')

        for k in range(N):

            V = vec[k, :, i]
            num_comp = comp_full_num[k]
            Vc = np.array(comp_full_p[num_comp])
            Va = np.array(vec[k-N_p : k+N_p, :, i])

            d1 = derivee(V, Vc, Va, k, num_comp)
            d2 = derivee(V + d1 * step / 2, Vc, Va, k, num_comp)
            d3 = derivee(V + d2 * step / 2, Vc, Va, k, num_comp)
            d4 = derivee(V + d3 * step, Vc, Va, k, num_comp)
            vec[k,:, i + 1] = V + step / 6 * (d1 + 2 * d2 + 2 * d3 + d4)

            if vec[k, 1, i + 1] <  Yc:

                for j in range(N_e + 1):
                    Vxl = vec[k, 0, i + 1]

                    if (Vxl > x0 - (int(N_e/2) - j + 1) * 2 * D/coef) and (Vxl < x0 - (int(N_e/2) - j) * 2 * D/coef) :

                        comp_vid_num[k] = j                    #numéro compartiment de la particule k
                        comp_vid_p[j].append(vec[k,:, i + 1]) #position particule k dans compartiment j
                        break

    return vec, t

def name(i,digit):

    i = str(i)

    while len(i)<digit:
        i = '0'+i

    i = 'img/'+i+'.png'

    return(i)


structure1, structure2, barres = strcuture()
vec, t = rk4(a, dt, tf)

print("RK4 sucessed")

extension="img/*.png"
for f in glob.glob(extension):
  os.remove(f)

for i in range(0, t.shape[0] - 1, pas_img):

    fig = plt.figure(figsize = (8, 8))

    ax = plt.gca()

    #ax.set_aspect('equal', adjustable='box')
    ax.set_xlim( -d, d)
    ax.set_ylim( -d-1, d)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    fig.patch.set_facecolor('black')

    circle1 = plt.Circle((0, 0), 100, color='black', ec = "black", lw = 10, zorder = -1)    #cercle
    ax.add_patch(circle1)

    for k in range(len(barres)):
        plt.plot(barres[k][0], barres[k][1], "-", color = 'white')

    ax.scatter(structure2[0], structure2[1], s = 200, c = "white")
    ax.scatter(vec[:, 0, i], vec[:, 1, i], s = 90, c = "red", ec = "white")

    name_pic = name(int(i/pas_img), digit)
    plt.savefig(name_pic, bbox_inches='tight', dpi=300)

    ax.clear()
    plt.close(fig)

    print(i/t.shape[0])

print("img sucessed")
print("--- %s seconds ---" % (time.time() - start_time))

# ffmpeg -i img/%05d.png -r 30 -pix_fmt yuv420p hexagon.mp4
# ffmpeg -r 10 -i img/%04d.png -vcodec libx264 -y -an test.mp4 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" (if 'width not divisible by two' error)

#!/usr/bin/python
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if (len(sys.argv) < 2):
    sys.exit('Usage: %s FICHER.out' % sys.argv[0])


f = open(sys.argv[1],'r')
content = f.read()

# On split 
info = content.split('Non-trainable') 

#On récupère les valeurs 
info.append(info[1].split('confusion matrix'))
info.pop(1)

#On récupère les valeurs de l'accuracy
text_acc = info[1][0].split('accuracy: ')         
                     
acc = []

for i in range(1,len(text_acc)):
    acc.append( float(text_acc[i][:6]))

plt.plot(acc)
plt.show()

#On affiche les matrices de confusion

for i in range(3):

    #On récupère les valeurs des matrices
    mat = info[1][i+1].split(' ')
    #On retire tous les caractères qu'on veux pas
    for j in range(len(mat)):
        mat[j] = mat[j].replace('\n','')
        mat[j] = mat[j].replace('[','')
        mat[j] = mat[j].replace(']','')
        mat[j] = mat[j].replace('#','')
    mat = list(filter(None,mat))
    mat_conf = np.zeros((23,23))
    mat = mat[:598]
    index = 0
    for x in range(23):
        for y in range(23):
            mat_conf[x][y] = float(mat[index])
            index = index +1
    img = plt.matshow(mat_conf)
    plt.colorbar(img)
    plt.show()
f.close()

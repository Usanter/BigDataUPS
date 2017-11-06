#!/usr/bin/python
# coding: utf-8


###################################
#####       Author            #####
###################################
#####    Veronique DEFONTE    #####
#####    Valentin GABEUR      #####
#####    Edouard VILLAIN      #####
#####    Thomas ROLLAND       #####
#####    Arnaud JOUANNEAU     #####
###################################


##################################
#####       Licenses        ######
##################################
##### Non commercial usage  ######
### Research and Teaching only ###
##################################





try:
    import ijson
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.misc
    from PIL import Image
except:
    print("Import Error : Please install required module")




def extract_img_from_json(json, path, nb_img):

    rec=False
    count=0
    rgbCount=0
    max_val=0
    
    with open(json) as file:
        parser = ijson.parse(file)
        for prefix, event, value in parser:
                
                if count>=nb_img : break
    
                if prefix=="features.item.Patch" and event=="start_array":
                    rec=True
                    a=np.array([])
                    rgbCount=0
                    
                if rgbCount%4==0: 
                    rgb=False 
                else: rgb=True
                
                if prefix=="features.item.Patch.item" and rec==True and rgb==True:
                    a=np.append(a, value)
                    
                if prefix=="features.item.Patch.item":
                    rgbCount+=1
                    
                if prefix=="features.item.Patch" and event=="end_array":
                    rec=False
                    a = np.resize(a,(24,24,3))
                    
                    pic_max=np.amax(a)
                    if pic_max>max_val:
                        max_val=pic_max
                        print(pic_max)
                    
                    a = np.divide(a, 2010)
                    #plt.figure()
                    #plt.imshow(a)
                    count+=1
                    scipy.misc.imsave(path + str(count) + '.jpg', a)
                    #print(str(count) + ' images saved')
         
            
def concatenate_img(path, filename, nb_img, img_size, start, lines, columns):

    list_im = []
    for i in range(nb_img):
        i+= start
        list_im.append(path + str(i) + '.jpg')
    
    total_width = columns*img_size
    max_height = lines*img_size
    
    new_im = Image.new('RGB', (total_width, max_height))
    
    x_offset = 0
    y_offset = 0
    count = 0
    for elem in list_im:
        im=Image.open(elem)
        new_im.paste(im, (x_offset,y_offset))
        count+=1
        if count%columns==0:
            y_offset += img_size
            x_offset = 0
        else:
            x_offset += img_size
    
    new_im.save(filename)
            



if __name__ == '__main__':        
    json = 'features_set_0.json' #large
    #json = '../part_features_set.json' #small

    path='./images/'

    extract_img_from_json(json, path, nb_img=37*37)

    concatenate_img(path, './results/r1.jpg', 37*37, 24, 1, 37, 37)
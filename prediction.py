# coding: utf-8

import keras
import h5py as h5
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D, Reshape, Dropout
import keras.layers.normalization
from keras.callbacks import Callback
import pywt
import os

PATH_SUBMIT_1_m1 = '/projets/bigdata4space/model_1_edouard_submit_1.h5'
PATH_SUBMIT_2_m1 = '/projets/bigdata4space/model_1_edouard_submit_2.h5'
PATH_SUBMIT_3_m1 = '/projets/bigdata4space/model_1_edouard_submit_3.h5'

GENERATOR_NAME_m1 = 'prediction_generator_model_edouard'

PATH_WEIGHT_1 = '/projets/bigdata4space/model_1_edouard_weight.h5'
PATH_MODEL_1 = '/projets/bigdata4space/model_1.h5'

PATH_SUBMIT_1_m2 = ''
PATH_SUBMIT_2_m2 = ''
PATH_SUBMIT_3_m2 = ''

GENERATOR_NAME_m1 = ''

PATH_SUBMIT_1_m2 = ''
PATH_SUBMIT_2_m2 = ''
PATH_SUBMIT_3_m2 = ''

GENERATOR_NAME_m1 = ''

PATH_PREDICT_WITH_GT_1 = '/projets/bigdata4space/HACKATHON/hackathon/data/pred_teachers/pred_from_full/pred_eighties_from_full_1.h5'
PATH_PREDICT_WITH_GT_2 = '/projets/bigdata4space/HACKATHON/hackathon/data/pred_teachers/pred_from_full/pred_eighties_from_full_2.h5'
PATH_PREDICT_WITH_GT_3 = '/projets/bigdata4space/HACKATHON/hackathon/data/pred_teachers/pred_from_full/pred_eighties_from_full_3.h5'

PATH_PREDICT_WITHOUT_GT_1 = '/projets/bigdata4space/HACKATHON/hackathon/data/pred_students/pred_from_full/pred_eighties_from_full_1_without_gt.h5'
PATH_PREDICT_WITHOUT_GT_2 = '/projets/bigdata4space/HACKATHON/hackathon/data/pred_students/pred_from_full/pred_eighties_from_full_2_without_gt.h5'
PATH_PREDICT_WITHOUT_GT_3 = '/projets/bigdata4space/HACKATHON/hackathon/data/pred_students/pred_from_full/pred_eighties_from_full_3_without_gt.h5'





def prediction_generator_model_edouard(h5_path, batch_size, idxs):
    f = h5.File(h5_path, 'r')
    batch_count = get_batch_count(idxs, batch_size)

    for b in range(batch_count):
        batch_idxs = idxs[b*batch_size:(b+1)*batch_size]
        batch_idxs = sorted(batch_idxs)
        X = []
        batch = f['S2'][batch_idxs, :,:,:]
        for bb in range(len(batch)):
            patch = batch[bb,:,:,:]

            # dwt2 for each chan
            cA0, (cD0, cV0, cH0) = pywt.dwt2(batch[bb, :,:,0], 'haar')
            cA1, (cD1, cV1, cH1) = pywt.dwt2(batch[bb, :,:,1], 'haar')
            cA2, (cD2, cV2, cH2) = pywt.dwt2(batch[bb, :,:,2], 'haar')
            cA3, (cD3, cV3, cH3) = pywt.dwt2(batch[bb, :,:,3], 'haar')

            dwt2 = np.zeros((16,16,4))
            dwt2[:,:,0] = np.concatenate((np.concatenate((cA0, cD0), axis=1), np.concatenate((cV0, cH0), axis=1)), axis=0)
            dwt2[:,:,1] = np.concatenate((np.concatenate((cA1, cD1), axis=1), np.concatenate((cV1, cH1), axis=1)), axis=0)
            dwt2[:,:,2] = np.concatenate((np.concatenate((cA2, cD2), axis=1), np.concatenate((cV2, cH2), axis=1)), axis=0)
            dwt2[:,:,3] = np.concatenate((np.concatenate((cA3, cD3), axis=1), np.concatenate((cV3, cH3), axis=1)), axis=0)

            fft2 = np.zeros((16,16,4))
            fft2[:,:,0] = np.real(np.fft.fft2(batch[bb, :,:,0]))
            fft2[:,:,1] = np.real(np.fft.fft2(batch[bb, :,:,1]))
            fft2[:,:,2] = np.real(np.fft.fft2(batch[bb, :,:,2]))
            fft2[:,:,3] = np.real(np.fft.fft2(batch[bb, :,:,3]))

            tmp1 = np.zeros((16,16,12))
            tmp1[:,:,0] = patch[:,:,0]
            tmp1[:,:,1] = patch[:,:,1]
            tmp1[:,:,2] = patch[:,:,2]
            tmp1[:,:,3] = patch[:,:,3]
            tmp1[:,:,4] = dwt2[:,:,0]
            tmp1[:,:,5] = dwt2[:,:,1]
            tmp1[:,:,6] = dwt2[:,:,2]
            tmp1[:,:,7] = dwt2[:,:,3]
            tmp1[:,:,8] = fft2[:,:,0]
            tmp1[:,:,9] = fft2[:,:,1]
            tmp1[:,:,10] = fft2[:,:,2]
            tmp1[:,:,11] = fft2[:,:,3]

            X.append(tmp1)


        yield np.array(X)


def build_h5_pred_file(pred, h5_output_path):
    if os.path.exists(h5_output_path):
        os.remove(h5_output_path)
    f = h5.File(h5_output_path, 'w')
    top_landcover_submit = f.create_dataset("TOP_LANDCOVER", (len(pred), 1), maxshape=(None, 1))
    top_landcover_submit[:, 0] = pred
    f.close()

    return 1



def gt_generator(h5_path, batch_size, idxs):
    f = h5.File(h5_path, 'r')

    batch_count = get_batch_count(idxs, batch_size)

    for b in range(batch_count):
        batch_idxs = idxs[b*batch_size:(b+1)*batch_size]
        batch_idxs = sorted(batch_idxs)
        Y = f['TOP_LANDCOVER'][batch_idxs, :]
        yield keras.utils.np_utils.to_categorical(np.array(Y), 23)



def prediction_function(model, PATH_SUBMIT_1, PATH_SUBMIT_2, PATH_SUBMIT_3, generator_prediction_name):
    for i in range(3):

        print('#### Prediction '+str(i+1)+' ####')
        if i == 0:
            PATH_PREDICT_WITH_GT = PATH_PREDICT_WITH_GT_1
            PATH_PREDICT_WITHOUT_GT = PATH_PREDICT_WITHOUT_GT_1
            PATH_SUBMIT = PATH_SUBMIT_1
        if i == 1:
            PATH_PREDICT_WITH_GT = PATH_PREDICT_WITH_GT_2
            PATH_PREDICT_WITHOUT_GT = PATH_PREDICT_WITHOUT_GT_2
            PATH_SUBMIT = PATH_SUBMIT_2
        if i == 2:
            PATH_PREDICT_WITH_GT = PATH_PREDICT_WITH_GT_3
            PATH_PREDICT_WITHOUT_GT = PATH_PREDICT_WITHOUT_GT_3
            PATH_SUBMIT = PATH_SUBMIT_3


        pred_idx = get_idxs(PATH_PREDICT_WITHOUT_GT)
        print(len(pred_idx))
        ## select the generator
        if generator_prediction_name == 'prediction_generator_model_edouard':
            pred_gen = prediction_generator_model_edouard(PATH_PREDICT_WITHOUT_GT, BATCH_SIZE, pred_idx)
        else:
            print('generator name does not exist')
        prediction = model.predict_generator(pred_gen, steps=get_batch_count(pred_idx, BATCH_SIZE), verbose=0)
        print(len(prediction))
        build_h5_pred_file(np.argmax(prediction, axis = 1), PATH_SUBMIT)


        gt_gen = gt_generator(PATH_PREDICT_WITH_GT, BATCH_SIZE, pred_idx)
        gt = []
        for elem in gt_gen:
            gt.append(elem)
        gt = np.vstack(gt)


        nb_ok = 0
        conf_mat = np.zeros((23,23))
        for i in range(len(gt)):
            conf_mat[int(np.argmax(gt[i]))][int(np.argmax(prediction[i]))] += 1
            if np.argmax(gt[i]) == np.argmax(prediction[i]):
                nb_ok += 1

        print("nombre de prediction : "+str(len(gt)))
        print("nombre de bonne prédiction : " + str(nb_ok))
        print("taux de reconnaissance : " + str(nb_ok/len(gt)))
        print("confusion matrix")
        print(conf_mat)



if __name__ == "__main__":
    print('\tModel 1 prediction : ')
    model_1 = keras.models.load_model(PATH_MODEL_1)
    model_1 = model.load_weights(PATH_WEIGHT_1)
    prediction_function(model_1, PATH_SUBMIT_1_m1, PATH_SUBMIT_2_m1, PATH_SUBMIT_3_m1, 'prediction_generator_model_edouard')

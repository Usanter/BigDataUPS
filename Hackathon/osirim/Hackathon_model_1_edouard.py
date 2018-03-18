# coding: utf-8

import keras
import h5py as h5
import numpy as np

PATH_DATA = '/projets/bigdata4space/HACKATHON/hackathon/data/train/full.h5'
PATH_SUBMIT_1 = '/projets/bigdata4space/model_1_edouard_submit_1.h5'
PATH_SUBMIT_2 = '/projets/bigdata4space/model_1_edouard_submit_2.h5'
PATH_SUBMIT_3 = '/projets/bigdata4space/model_1_edouard_submit_3.h5'
PATH_PREDICT_WITH_GT_1 = '/projets/bigdata4space/HACKATHON/hackathon/data/pred_teachers/pred_from_full/pred_eighties_from_full_1.h5'
PATH_PREDICT_WITH_GT_2 = '/projets/bigdata4space/HACKATHON/hackathon/data/pred_teachers/pred_from_full/pred_eighties_from_full_2.h5'
PATH_PREDICT_WITH_GT_3 = '/projets/bigdata4space/HACKATHON/hackathon/data/pred_teachers/pred_from_full/pred_eighties_from_full_3.h5'
PATH_PREDICT_WITHOUT_GT_1 = '/projets/bigdata4space/HACKATHON/hackathon/data/pred_students/pred_from_full/pred_eighties_from_full_1_without_gt.h5'
PATH_PREDICT_WITHOUT_GT_2 = '/projets/bigdata4space/HACKATHON/hackathon/data/pred_students/pred_from_full/pred_eighties_from_full_2_without_gt.h5'
PATH_PREDICT_WITHOUT_GT_3 = '/projets/bigdata4space/HACKATHON/hackathon/data/pred_students/pred_from_full/pred_eighties_from_full_3_without_gt.h5'
PATH_WEIGHT = '/projets/bigdata4space/model_1_edouard_weight.h5'

BATCH_SIZE = 32
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D, Reshape, Dropout
import keras.layers.normalization
from keras.callbacks import Callback
import pywt
#import cv2
import os


def get_idxs(h5_path):
    f = h5.File(h5_path)
    return range(len(f['S2']))

def shuffle_idx(sample_idxs):
    return list(np.random.permutation(sample_idxs))

def split_train_val(sample_idxs, proportion):
    n_samples = len(sample_idxs)
    return sample_idxs[:int((1.-proportion)*n_samples)], sample_idxs[int((1.-proportion)*n_samples):]

def get_batch_count(idxs, batch_size):
    batch_count = int(len(idxs)//batch_size)
    remained_samples = len(idxs)%batch_size
    if remained_samples > 0:
        batch_count += 1

    return batch_count


idxs = get_idxs(PATH_DATA)
shuffled_idxs = shuffle_idx(idxs)
train_idxs, val_idxs = split_train_val(shuffled_idxs, 0.2)



def generator_histogram(h5_path, batch_size, idxs):
    f = h5.File(h5_path, 'r')
    while True :
        idxs = shuffle_idx(idxs)
        batch_count = get_batch_count(idxs, batch_size)
        for b in range(batch_count):
            batch_idxs = idxs[b*batch_size:(b+1)*batch_size]
            batch_idxs = sorted(batch_idxs)
            X = []
            Y = f['TOP_LANDCOVER'][batch_idxs, :]
            batch = f['S2'][batch_idxs, :,:,:]
            for bb in range(len(batch)):
                patch = batch[bb,:,:,:]

                # dwt2 for each chan
                cA0, (cD0, cV0, cH0) = pywt.dwt2(batch[bb,:,:,0], 'haar')
                cA1, (cD1, cV1, cH1) = pywt.dwt2(batch[bb,:,:,1], 'haar')
                cA2, (cD2, cV2, cH2) = pywt.dwt2(batch[bb,:,:,2], 'haar')
                cA3, (cD3, cV3, cH3) = pywt.dwt2(batch[bb,:,:,3], 'haar')

                dwt2 = np.zeros((16,16,4))
                dwt2[:,:,0] = np.concatenate((np.concatenate((cA0, cD0), axis=1), np.concatenate((cV0, cH0), axis=1)), axis=0)
                dwt2[:,:,1] = np.concatenate((np.concatenate((cA1, cD1), axis=1), np.concatenate((cV1, cH1), axis=1)), axis=0)
                dwt2[:,:,2] = np.concatenate((np.concatenate((cA2, cD2), axis=1), np.concatenate((cV2, cH2), axis=1)), axis=0)
                dwt2[:,:,3] = np.concatenate((np.concatenate((cA3, cD3), axis=1), np.concatenate((cV3, cH3), axis=1)), axis=0)

                fft2 = np.zeros((16,16,4))
                fft2[:,:,0] = np.real(np.fft.fft2(batch[bb,:,:,0]))
                fft2[:,:,1] = np.real(np.fft.fft2(batch[bb,:,:,1]))
                fft2[:,:,2] = np.real(np.fft.fft2(batch[bb,:,:,2]))
                fft2[:,:,3] = np.real(np.fft.fft2(batch[bb,:,:,3]))


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

            yield np.array(X), keras.utils.np_utils.to_categorical(np.array(Y), 23)


train_gen = generator_histogram(PATH_DATA, BATCH_SIZE, train_idxs)
train_batch_count = get_batch_count(train_idxs, BATCH_SIZE)

val_gen = generator_histogram(PATH_DATA, BATCH_SIZE, val_idxs)
val_batch_count = get_batch_count(val_idxs, BATCH_SIZE)
print(train_batch_count, val_batch_count)
print(np.shape(train_gen.__next__()[0]))

input_shape = (16,16,12)
model = Sequential()
model.add(BatchNormalization(input_shape=input_shape))
model.add(Conv2D(16, kernel_size=(3, 3),activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(23, activation='softmax'))


model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

model.summary()


history = model.fit_generator(train_gen, steps_per_epoch=train_batch_count/BATCH_SIZE, epochs=3, verbose=2,
                              validation_data=val_gen, validation_steps=val_batch_count/BATCH_SIZE)
model.save_weights(PATH_WEIGHT)


score = model.evaluate_generator(val_gen, steps=val_batch_count/BATCH_SIZE)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


def prediction_generator(h5_path, batch_size, idxs):
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
    if i == 0:
        PATH_PREDICT_WITH_GT = PATH_PREDICT_WITH_GT_3
        PATH_PREDICT_WITHOUT_GT = PATH_PREDICT_WITHOUT_GT_3
        PATH_SUBMIT = PATH_SUBMIT_3

    pred_idx = get_idxs(PATH_PREDICT_WITHOUT_GT)
    print(len(pred_idx))
    pred_gen = prediction_generator(PATH_PREDICT_WITHOUT_GT, BATCH_SIZE, pred_idx)
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

############ Test

import os
import tensorflow as tf
from keras.backend import tensorflow_backend

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

from utils import define_model, prepare_dataset, crop_prediction
from utils.evaluate import evaluate
from keras.layers import ReLU
from tqdm import tqdm
import numpy as np
from skimage.transform import resize
import cv2
import pickle


def predict(ACTIVATION='ReLU', dropout=0.1, batch_size=32, repeat=4, minimum_kernel=32,
            epochs=200, iteration=3, crop_size=128, stride_size=3, DATASET='DRIVE'):
    prepare_dataset.prepareDataset(DATASET)
    test_data = [prepare_dataset.getTestData(0, DATASET),
                 prepare_dataset.getTestData(1, DATASET),
                 prepare_dataset.getTestData(2, DATASET)]

    IMAGE_SIZE = {
        'DRIVE': (565, 584),
        'CHASEDB1': (999, 960),
        'STARE': (700, 605),
        'MYTEST': (512, 512)
    }.get(DATASET, (512, 512))

    gt_list_out = {}
    pred_list_out = {}
    for out_id in range(iteration + 1):
        os.makedirs(f"./output/{DATASET}/crop_size_{crop_size}/out{out_id + 1}/", exist_ok=True)
        gt_list_out[f"out{out_id + 1}"] = []
        pred_list_out[f"out{out_id + 1}"] = []

    activation = globals()[ACTIVATION]
    model = define_model.get_unet(minimum_kernel=minimum_kernel, do=dropout, activation=activation, iteration=iteration)
    model_name = f"Final_Emer_Iteration_{iteration}_cropsize_{crop_size}_epochs_{epochs}"
    print("Model : %s" % model_name)
    load_path = f"./trained_model/weights.hdf5"
    model.load_weights(load_path, by_name=False)

    imgs = test_data[0]
    segs = test_data[1] if test_data[1] is not None else [None] * len(imgs)
    masks = test_data[2] if test_data[2] is not None else [None] * len(imgs)

    for i in tqdm(range(len(imgs))):
        img = imgs[i]
        seg = segs[i]
        mask = masks[i]

        patches_pred, new_height, new_width, adjustImg = crop_prediction.get_test_patches(img, crop_size, stride_size)
        preds = model.predict(patches_pred)

        out_id = 0
        for pred in preds:
            pred_patches = crop_prediction.pred_to_patches(pred, crop_size, stride_size)
            pred_imgs = crop_prediction.recompone_overlap(pred_patches, crop_size, stride_size, new_height, new_width)
            pred_imgs = pred_imgs[:, 0:prepare_dataset.DESIRED_DATA_SHAPE[0], 0:prepare_dataset.DESIRED_DATA_SHAPE[0], :]
            probResult = pred_imgs[0, :, :, 0]
            pred_ = probResult

            # Save raw prediction
            out_dir = f"./output/{DATASET}/crop_size_{crop_size}/out{out_id + 1}/"
            with open(os.path.join(out_dir, f"{i + 1:02}.pickle"), 'wb') as handle:
                pickle.dump(pred_, handle, protocol=pickle.HIGHEST_PROTOCOL)

            pred_ = resize(pred_, IMAGE_SIZE[::-1])

            gt_ = None
            if seg is not None:
                seg_ = resize(seg, IMAGE_SIZE[::-1])
                gt_ = (seg_ > 0.5).astype(int)

            if mask is not None:
                mask_ = resize(mask, IMAGE_SIZE[::-1])
            else:
                mask_ = None

            gt_flat = []
            pred_flat = []
            if gt_ is not None:
                for p in range(pred_.shape[0]):
                    for q in range(pred_.shape[1]):
                        if mask_ is None or mask_[p, q] > 0.5:
                            gt_flat.append(gt_[p, q])
                            pred_flat.append(pred_[p, q])
                gt_list_out[f"out{out_id + 1}"] += gt_flat
                pred_list_out[f"out{out_id + 1}"] += pred_flat

            # Save image
            pred_norm = 255. * (pred_ - np.min(pred_)) / (np.max(pred_) - np.min(pred_))
            cv2.imwrite(os.path.join(out_dir, f"{i + 1:02}.png"), pred_norm.astype(np.uint8))
            out_id += 1

    for out_id in range(iteration + 1)[-1:]:
        if gt_list_out[f"out{out_id + 1}"]:
            print(f"\n\nout{out_id + 1}")
            evaluate(gt_list_out[f"out{out_id + 1}"], pred_list_out[f"out{out_id + 1}"], DATASET)
        else:
            print(f"\n\nout{out_id + 1} — no ground truth provided, skipping evaluation")


if __name__ == "__main__":
    predict(batch_size=32, epochs=200, iteration=3, stride_size=3, DATASET='MYTEST')

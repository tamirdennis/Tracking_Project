import os
import numpy as np
from matplotlib.pyplot import imread
from IPython.display import Image
from skimage.transform import resize
import matplotlib.pyplot as plt
import random
from itertools import combinations, product, cycle
from keras.layers import Dense, Dropout, Activation, Flatten, ReLU, BatchNormalization, MaxPooling2D, Conv2D, Lambda, Input
from keras import regularizers, Model
from keras.models import Sequential
from keras import backend as K
import keras
from keras.utils import plot_model
from keras.callbacks import TensorBoard
import cv2
import imageio
import imgaug as ia
from imgaug import augmenters as iaa


INPUT_SHAPE = [80, 35, 3]
BEST_MODEL_PATH = "models/best_model.hdf5"


def process_image(img):
    return resize(img, (INPUT_SHAPE[0], INPUT_SHAPE[1]), anti_aliasing=True)


def get_augmented_image_list(images):
    """
    Generating a list of augmented images with three different augmentations: noise, contrast, and dropout.
    :param images: list of images
    :return: list of images with length of 3 * len(images) of augmented images from images.
    """
    result = []
    noise = iaa.AdditiveGaussianNoise(scale=0.05 * 255, per_channel=True)
    noise_images = noise.augment_images(images)

    contrast = iaa.LinearContrast(alpha=1.5)
    contrast_images = contrast.augment_images(images)

    dropout = iaa.CoarseDropout(p=0.2, size_percent=0.1)
    dropout_images = dropout.augment_images(images)

    result.extend(dropout_images)
    result.extend(noise_images)
    result.extend(contrast_images)
    return result


def get_labels_proc_images_dict(path):
    '''
    path => Path of train directory or test directory
    '''
    images_dict = {}

    for image_path in os.listdir(path):
        id = int(image_path.split("_")[0])
        img = imread(path + "/" + image_path)
        if id in images_dict.keys():
            images_dict[id].append(img)
        else:
            images_dict[id] = [img]
    for img_list in images_dict.values():
        augmented_images = get_augmented_image_list(img_list)
        img_list.extend(augmented_images)
        for i, img in enumerate(img_list):
            img_list[i] = process_image(img)
        # for visualizing all the image list with the augmentations
        # ia.imshow(np.hstack(img_list))
        random.shuffle(img_list)
    return images_dict


def get_all_possible_equal_pairs_indexes(imgs_dict_labeled):
    """
    will return a list of ((label, image), (label, other_image)) tuples (duplets of same label)
    :param imgs_dict_labeled: dictionary of label as key and images list as value
    :return: list of same label duplets
    """
    result = []
    labels = imgs_dict_labeled.keys()
    for label in labels:
        pair_comb = list(combinations(list(range(len(imgs_dict_labeled[label]))), 2))
        sample_indexes = [((label, pair[0]), (label, pair[1])) for pair in pair_comb]
        result.extend(sample_indexes)
    return result


def get_all_possible_unequal_pairs_indexes(imgs_dict_labeled):
    """
    will return a list of ((label, image), (other_label, other_image)) tuples (duplets of different labels)
    :param imgs_dict_labeled: dictionary of label as key and images list as value
    :return: list of different labels duplets
    """
    result = []
    labels = imgs_dict_labeled.keys()
    for label_pair in list(combinations(labels, 2)):
        first_label_indexes = list(range(len(imgs_dict_labeled[label_pair[0]])))
        second_label_indexes = list(range(len(imgs_dict_labeled[label_pair[1]])))
        img_indexes_pairs = list(product(first_label_indexes, second_label_indexes))
        sample_indexes = [((label_pair[0], pair[0]), (label_pair[1], pair[1])) for pair in img_indexes_pairs]
        result.extend(sample_indexes)
    return result


def get_pair_indexes_from_dict(img_dict):
    """
    Generating a list of duplets organized as one equal duplet(same label) and one unequal after each list was shuffled.
    :param img_dict: dictionary of label as key and images list as value
    :return: list of duplets.
    """
    equals_indexes = get_all_possible_equal_pairs_indexes(img_dict)
    random.shuffle(equals_indexes)
    unequals_indexes = get_all_possible_unequal_pairs_indexes(img_dict)
    random.shuffle(unequals_indexes)
    pair_indexes_data = []
    for i in range(len(equals_indexes)):
        pair_indexes_data.append(equals_indexes[i])
        pair_indexes_data.append(unequals_indexes[i])
    return pair_indexes_data


def build_layers(config, use_BN: bool):
    """

    :param config: config list of layers like the format: [32,32, "MaxPool", 64,128,"MaxPool"] and so on.
    :param use_BN: if should add a BatchNormalization layer after each convolution or not.
    :return:
    """
    layers = []
    for layer in config:
        if layer == "MaxPool":
            layers += [MaxPooling2D(pool_size=2, strides=2)]
        else:
            if len(layers) == 0:
                conv = Conv2D(kernel_size=(3, 3), filters=layer, padding='same',
                              input_shape=INPUT_SHAPE, kernel_initializer='glorot_uniform')
            else:
                conv = Conv2D(kernel_size=(3, 3), filters=layer, padding='same')

            layers.append(conv)
            if use_BN:
                layers.append(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
            layers.append(ReLU())
    return layers


def get_first_blocks_list(num_filters):
    return [num_filters, num_filters, "MaxPool"]


def get_last_blocks_list(num_filters):
    return [num_filters, num_filters, num_filters, "MaxPool"]


def my_plot_model(model):
  im = plot_model(model, to_file='model_{}.png'.format(model.name), show_shapes=True, show_layer_names=True, rankdir='TB')
  return Image(retina=True, filename='model_{}.png'.format(model.name))


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1


def get_siamese_model(input_shape):
    """
        Model architecture
    """

    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    # Convolutional Neural Network

    model = Sequential()
    config_blocks_list = get_last_blocks_list(256)
    blocks_layers = build_layers(config_blocks_list, use_BN=True)
    dense_reg = regularizers.l2(0.001)
    classification_layers = [Flatten(),
                             Dense(units=512,
                                   kernel_initializer='glorot_normal',
                                   bias_initializer='zeros'
                                   ),
                             ReLU(),
                             Dense(units=512,
                                   kernel_initializer='glorot_normal',
                                   bias_initializer='zeros',
                                   name='features'),
                             ReLU()]
    for layer in (blocks_layers + classification_layers):
        model.add(layer)
    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)

    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1, activation='sigmoid', bias_initializer='glorot_normal')(L1_distance)
    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)
    # tried to use contrastive loss but it was locked on accuracy 0.5:

    # distance = Lambda(euclidean_distance,
    #                   output_shape=eucl_dist_output_shape)([encoded_l, encoded_r])
    # Connect the inputs with the outputs
    # siamese_net = Model(inputs=[left_input, right_input], outputs=distance)

    # return the model
    return siamese_net

# tried to use contrastive loss but it was locked on accuracy 0.5:
# def contrastive_loss(y_true, y_pred):
#     '''Contrastive loss from Hadsell-et-al.'06
#     http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
#     '''
#     margin = 1
#     square_pred = K.square(y_pred)
#     margin_square = K.square(K.maximum(margin - y_pred, 0))
#     return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def generate(batch_size, img_pool_dict, pairs_indexes_iterator, s="train"):
    """
    generate batch_sized batch from pairs_indexes_iterator with there labels of 1 if same label person or 0 otherwise.

    """
    h = INPUT_SHAPE[0]
    w = INPUT_SHAPE[1]
    dim = 3
    pairs = [np.zeros((batch_size, h, w, dim)) for _ in range(2)]
    targets = np.zeros((batch_size,))
    i = 0
    for pair in pairs_indexes_iterator:
        img0 = img_pool_dict[pair[0][0]][pair[0][1]]
        img1 = img_pool_dict[pair[1][0]][pair[1][1]]
        if pair[0][0] == pair[1][0]:
            targets[i] = 1
        pairs[0][i, :, :, :] = img0
        pairs[1][i, :, :, :] = img1
        i += 1
        if i == batch_size:
            i = 0
            yield (pairs, targets)


# def accuracy(y_true, y_pred):
#     '''Compute classification accuracy with a fixed threshold on distances.
#     '''
#     return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


if __name__ == '__main__':
    imgs_dict_labeled_train = get_labels_proc_images_dict('data/pku_raid/train')
    imgs_dict_labeled_test = get_labels_proc_images_dict('data/pku_raid/test')
    data_train_pair_indexes = get_pair_indexes_from_dict(imgs_dict_labeled_train)
    data_test_pair_indexes = get_pair_indexes_from_dict(imgs_dict_labeled_test)
    iterator_pairs_indexes_train = cycle(data_train_pair_indexes)
    iterator_pairs_indexes_test = cycle(data_test_pair_indexes)

    batch_size = 32
    model = get_siamese_model(INPUT_SHAPE)
    steps_per_epoch = 40
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-9)
    log_dir = 'log_dir'
    tensorboard = TensorBoard(log_dir=log_dir)
    # tried to use contrastive loss but it was locked on accuracy 0.5:
    # model.compile(loss=contrastive_loss,
    #               optimizer=opt,
    #               metrics=[accuracy])
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    my_plot_model(model)

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                  factor=0.9,
                                                  patience=10,
                                                  min_lr=1e-6)
    checkpoint = keras.callbacks.ModelCheckpoint(BEST_MODEL_PATH, monitor='val_loss', verbose=0, save_best_only=True,
                                    save_weights_only=False, mode='auto', period=1)

    model.fit_generator(
            generate(batch_size, imgs_dict_labeled_train, iterator_pairs_indexes_train, s='train'),
            steps_per_epoch=steps_per_epoch, verbose=1, callbacks=[tensorboard, reduce_lr, checkpoint], validation_steps=30, shuffle=False, epochs=1000,
            validation_data=generate(batch_size, imgs_dict_labeled_test, iterator_pairs_indexes_test, s='test')
    )

import os
import warnings

import librosa
import tensorflow as tf

from musicnn import configuration as config

# disabling (most) warnings caused by change from tensorflow 1.x to 2.x
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
tf.compat.v1.logging.set_verbosity("ERROR")


def define_model(x, is_training, model, num_classes):
    if model == "MTT_musicnn":
        return build_musicnn(
            x, is_training, num_classes, num_filt_midend=64, num_units_backend=200
        )

    elif model == "MTT_vgg":
        return vgg(x, is_training, num_classes, 128)

    elif model == "MSD_musicnn":
        return build_musicnn(
            x, is_training, num_classes, num_filt_midend=64, num_units_backend=200
        )

    elif model == "MSD_musicnn_big":
        return build_musicnn(
            x, is_training, num_classes, num_filt_midend=512, num_units_backend=500
        )

    elif model == "MSD_vgg":
        return vgg(x, is_training, num_classes, 128)

    else:
        raise ValueError("Model not implemented!")


def build_musicnn(
    x,
    is_training,
    num_classes,
    num_filt_frontend=1.6,
    num_filt_midend=64,
    num_units_backend=200,
):
    ### front-end ### musically motivated CNN
    frontend_features_list = frontend(
        x,
        is_training,
        config.N_MELS,
        num_filt=num_filt_frontend,
        type="7774timbraltemporal",
    )
    # concatnate features coming from the front-end
    frontend_features = tf.concat(frontend_features_list, 2)

    ### mid-end ### dense layers
    midend_features_list = midend(frontend_features, is_training, num_filt_midend)
    # dense connection: concatnate features coming from different layers of the front- and mid-end
    midend_features = tf.concat(midend_features_list, 2)

    ### back-end ### temporal pooling
    logits, penultimate, mean_pool, max_pool = backend(
        midend_features,
        is_training,
        num_classes,
        num_units_backend,
        type="globalpool_dense",
    )

    # [extract features] temporal and timbral features from the front-end
    timbral = tf.concat([frontend_features_list[0], frontend_features_list[1]], 2)
    temporal = tf.concat(
        [
            frontend_features_list[2],
            frontend_features_list[3],
            frontend_features_list[4],
        ],
        2,
    )
    # [extract features] mid-end features
    cnn1, cnn2, cnn3 = (
        midend_features_list[1],
        midend_features_list[2],
        midend_features_list[3],
    )
    mean_pool = tf.squeeze(mean_pool, [2])
    max_pool = tf.squeeze(max_pool, [2])

    return logits, timbral, temporal, cnn1, cnn2, cnn3, mean_pool, max_pool, penultimate


def frontend(x, is_training, yInput, num_filt, type):
    expand_input = tf.expand_dims(x, 3)
    normalized_input = tf.compat.v1.layers.batch_normalization(
        expand_input, training=is_training
    )

    if "timbral" in type:
        # padding only time domain for an efficient 'same' implementation
        # (since we pool throughout all frequency afterwards)
        input_pad_7 = tf.pad(
            normalized_input, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT"
        )

        if "74" in type:
            f74 = timbral_block(
                inputs=input_pad_7,
                filters=int(num_filt * 128),
                kernel_size=[7, int(0.4 * yInput)],
                is_training=is_training,
            )

        if "77" in type:
            f77 = timbral_block(
                inputs=input_pad_7,
                filters=int(num_filt * 128),
                kernel_size=[7, int(0.7 * yInput)],
                is_training=is_training,
            )

    if "temporal" in type:
        s1 = tempo_block(
            inputs=normalized_input,
            filters=int(num_filt * 32),
            kernel_size=[128, 1],
            is_training=is_training,
        )

        s2 = tempo_block(
            inputs=normalized_input,
            filters=int(num_filt * 32),
            kernel_size=[64, 1],
            is_training=is_training,
        )

        s3 = tempo_block(
            inputs=normalized_input,
            filters=int(num_filt * 32),
            kernel_size=[32, 1],
            is_training=is_training,
        )

    # choose the feature maps we want to use for the experiment
    if type == "7774timbraltemporal":
        return [f74, f77, s1, s2, s3]


def timbral_block(
    inputs, filters, kernel_size, is_training, padding="valid", activation=tf.nn.relu
):
    conv = tf.compat.v1.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        padding=padding,
        activation=activation,
    )
    bn_conv = tf.compat.v1.layers.batch_normalization(conv, training=is_training)
    pool = tf.compat.v1.layers.max_pooling2d(
        inputs=bn_conv, pool_size=[1, bn_conv.shape[2]], strides=[1, bn_conv.shape[2]]
    )
    return tf.squeeze(pool, [2])


def tempo_block(
    inputs, filters, kernel_size, is_training, padding="same", activation=tf.nn.relu
):
    conv = tf.compat.v1.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        padding=padding,
        activation=activation,
    )
    bn_conv = tf.compat.v1.layers.batch_normalization(conv, training=is_training)
    pool = tf.compat.v1.layers.max_pooling2d(
        inputs=bn_conv, pool_size=[1, bn_conv.shape[2]], strides=[1, bn_conv.shape[2]]
    )
    return tf.squeeze(pool, [2])


def midend(front_end_output, is_training, num_filt):
    front_end_output = tf.expand_dims(front_end_output, 3)

    # conv layer 1 - adapting dimensions
    front_end_pad = tf.pad(
        front_end_output, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT"
    )
    conv1 = tf.compat.v1.layers.conv2d(
        inputs=front_end_pad,
        filters=num_filt,
        kernel_size=[7, front_end_pad.shape[2]],
        padding="valid",
        activation=tf.nn.relu,
    )
    bn_conv1 = tf.compat.v1.layers.batch_normalization(conv1, training=is_training)
    bn_conv1_t = tf.transpose(bn_conv1, [0, 1, 3, 2])

    # conv layer 2 - residual connection
    bn_conv1_pad = tf.pad(bn_conv1_t, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT")
    conv2 = tf.compat.v1.layers.conv2d(
        inputs=bn_conv1_pad,
        filters=num_filt,
        kernel_size=[7, bn_conv1_pad.shape[2]],
        padding="valid",
        activation=tf.nn.relu,
    )
    bn_conv2 = tf.compat.v1.layers.batch_normalization(conv2, training=is_training)
    conv2 = tf.transpose(bn_conv2, [0, 1, 3, 2])
    res_conv2 = tf.add(conv2, bn_conv1_t)

    # conv layer 3 - residual connection
    bn_conv2_pad = tf.pad(res_conv2, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT")
    conv3 = tf.compat.v1.layers.conv2d(
        inputs=bn_conv2_pad,
        filters=num_filt,
        kernel_size=[7, bn_conv2_pad.shape[2]],
        padding="valid",
        activation=tf.nn.relu,
    )
    bn_conv3 = tf.compat.v1.layers.batch_normalization(conv3, training=is_training)
    conv3 = tf.transpose(bn_conv3, [0, 1, 3, 2])
    res_conv3 = tf.add(conv3, res_conv2)

    return [front_end_output, bn_conv1_t, res_conv2, res_conv3]


def backend(feature_map, is_training, num_classes, output_units, type):
    # temporal pooling
    max_pool = tf.reduce_max(feature_map, axis=1)
    mean_pool, _ = tf.nn.moments(feature_map, axes=[1])
    tmp_pool = tf.concat([max_pool, mean_pool], 2)

    # penultimate dense layer
    flat_pool = tf.compat.v1.layers.flatten(tmp_pool)
    flat_pool = tf.compat.v1.layers.batch_normalization(flat_pool, training=is_training)
    flat_pool_dropout = tf.compat.v1.layers.dropout(
        flat_pool, rate=0.5, training=is_training
    )
    dense = tf.compat.v1.layers.dense(
        inputs=flat_pool_dropout, units=output_units, activation=tf.nn.relu
    )
    bn_dense = tf.compat.v1.layers.batch_normalization(dense, training=is_training)
    dense_dropout = tf.compat.v1.layers.dropout(
        bn_dense, rate=0.5, training=is_training
    )

    # output dense layer
    logits = tf.compat.v1.layers.dense(
        inputs=dense_dropout, activation=None, units=num_classes
    )

    return logits, bn_dense, mean_pool, max_pool


def vgg(x, is_training, num_classes, num_filters=32):
    input_layer = tf.expand_dims(x, 3)
    bn_input = tf.compat.v1.layers.batch_normalization(
        input_layer, training=is_training
    )

    conv1 = tf.compat.v1.layers.conv2d(
        inputs=bn_input,
        filters=num_filters,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        name="1CNN",
    )
    bn_conv1 = tf.compat.v1.layers.batch_normalization(conv1, training=is_training)
    pool1 = tf.compat.v1.layers.max_pooling2d(
        inputs=bn_conv1, pool_size=[4, 1], strides=[2, 2]
    )

    do_pool1 = tf.compat.v1.layers.dropout(pool1, rate=0.25, training=is_training)
    conv2 = tf.compat.v1.layers.conv2d(
        inputs=do_pool1,
        filters=num_filters,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        name="2CNN",
    )
    bn_conv2 = tf.compat.v1.layers.batch_normalization(conv2, training=is_training)
    pool2 = tf.compat.v1.layers.max_pooling2d(
        inputs=bn_conv2, pool_size=[2, 2], strides=[2, 2]
    )

    do_pool2 = tf.compat.v1.layers.dropout(pool2, rate=0.25, training=is_training)
    conv3 = tf.compat.v1.layers.conv2d(
        inputs=do_pool2,
        filters=num_filters,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        name="3CNN",
    )
    bn_conv3 = tf.compat.v1.layers.batch_normalization(conv3, training=is_training)
    pool3 = tf.compat.v1.layers.max_pooling2d(
        inputs=bn_conv3, pool_size=[2, 2], strides=[2, 2]
    )

    do_pool3 = tf.compat.v1.layers.dropout(pool3, rate=0.25, training=is_training)
    conv4 = tf.compat.v1.layers.conv2d(
        inputs=do_pool3,
        filters=num_filters,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        name="4CNN",
    )
    bn_conv4 = tf.compat.v1.layers.batch_normalization(conv4, training=is_training)
    pool4 = tf.compat.v1.layers.max_pooling2d(
        inputs=bn_conv4, pool_size=[2, 2], strides=[2, 2]
    )

    do_pool4 = tf.compat.v1.layers.dropout(pool4, rate=0.25, training=is_training)
    conv5 = tf.compat.v1.layers.conv2d(
        inputs=do_pool4,
        filters=num_filters,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        name="5CNN",
    )
    bn_conv5 = tf.compat.v1.layers.batch_normalization(conv5, training=is_training)
    pool5 = tf.compat.v1.layers.max_pooling2d(
        inputs=bn_conv5, pool_size=[4, 4], strides=[4, 4]
    )

    flat_pool5 = tf.compat.v1.layers.flatten(pool5)
    do_pool5 = tf.compat.v1.layers.dropout(flat_pool5, rate=0.5, training=is_training)
    output = tf.compat.v1.layers.dense(
        inputs=do_pool5, activation=None, units=num_classes
    )
    return output, pool1, pool2, pool3, pool4, pool5


def load_model(
    model: str, input_length: float = 3.0
) -> tuple[tf.compat.v1.Session, list, list, int]:
    """Define and load the tensorflow model.

    PARAMETERS
    ----------
    model : str
        Name of the model to load.
    input_length : float, default 3.0
        Length of the input spectrogram patches in seconds.

    RETURNS
    -------
    session : tf.compat.v1.Session
        TensorFlow session with the loaded model.
    feed_dict_values : list
        List of placeholder values for the model.
    extract_vector : list
        List containing the model's output tensors.
    n_frames : int
        Number of frames in the input spectrogram patches.
    """
    # select model
    if model not in config.MODELS:
        raise ValueError(
            f"Model '{model}' is not recognised. Available models are: {config.MODELS}"
        )
    if model.startswith("MTT_"):
        labels = config.MTT_LABELS
    else:
        labels = config.MSD_LABELS
    num_classes = len(labels)

    if "vgg" in model and input_length != 3:
        raise ValueError(
            "Set input_length=3, the VGG models cannot handle different input lengths."
        )

    # convert seconds to frames
    n_frames = (
        librosa.time_to_frames(
            input_length, sr=config.SR, n_fft=config.FFT_SIZE, hop_length=config.FFT_HOP
        )
        + 1
    )

    # tensorflow: define the model
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.name_scope("model"):
        x = tf.compat.v1.placeholder(tf.float32, [None, n_frames, config.N_MELS])
        is_training = tf.compat.v1.placeholder(tf.bool)
        extract_vector = list(define_model(x, is_training, model, num_classes))
        extract_vector[0] = tf.nn.sigmoid(extract_vector[0])  # the tags

    # tensorflow: loading model
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver()
    try:
        saver.restore(sess, os.path.dirname(__file__) + "\\" + model + "\\")
    except tf.errors.DataLossError as error:
        if model == "MSD_musicnn_big":
            raise ValueError(
                "'MSD_musicnn_big' model is only available if you install from source."
            ) from error
        raise ValueError(f"Model '{model}' cannot be loaded.") from error
    feed_list = [x, is_training]

    return sess, feed_list, extract_vector, n_frames

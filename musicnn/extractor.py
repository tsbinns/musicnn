import warnings

import numpy as np
import librosa
import tensorflow as tf

from musicnn import models
from musicnn import configuration as config

# disabling (most) warnings caused by change from tensorflow 1.x to 2.x
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
tf.compat.v1.logging.set_verbosity("ERROR")

# disable eager mode for tf.v1 compatibility with tf.v2
tf.compat.v1.disable_eager_execution()


def batch_data(
    audio: str | np.ndarray,
    n_frames: int,
    input_overlap: float | bool,
    sr: float | None = None,
    resample_kwargs: dict = {},
) -> tuple[np.ndarray, np.ndarray]:
    """Compute and split the full audio spectrogram into batches.

    PARAMETERS
    ----------
    audio: str | np.ndarray, shape (channels, times)
        A path to the audio file or a numpy array containing the audio data.
    n_frames: int
        Length (in frames) of the input spectrogram batches.
    input_overlap: float | bool
        Amount of overlap (in seconds) of the input spectrogram batches. `False` for no
        overlap.
    sr: float | None, default None
        Sampling rate of `audio`, if it is an array.
    resample_kwargs: dict, default {}
        Keyword arguments for `librosa.resample` if `audio` is a numpy array. Only used
        if `sr` does not match the expected 16 kHz.

    RETURNS
    -------
    batched_spectrogram: np.ndarray, shape (batches, times, frequencies)
        Audio spectrogram in batches.

    full_spectrogram: np.ndarray, shape (times, frequencies)
        Full audio spectrogram.
    """
    # convert overlap from seconds to frames
    if isinstance(input_overlap, bool) and input_overlap:
        raise ValueError("If `overlap` is a bool, it must be False.")
    if not input_overlap:
        overlap = n_frames
    else:
        overlap = librosa.time_to_frames(
            overlap,
            sr=config.SR,
            n_fft=config.FFT_SIZE,
            hop_length=config.FFT_HOP,
        )

    # Prepare data
    if isinstance(audio, np.ndarray):
        if sr is None:
            raise ValueError("If `audio` is a numpy array, `sr` must be provided.")
        if audio.ndim != 2:
            raise ValueError("`audio` must be a 2D numpy array of (channels, times).")
        if sr != config.SR:  # standardise sampling rate
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=config.SR, **resample_kwargs
            )
    else:
        if not isinstance(audio, str):
            raise ValueError("`audio` must be a string or a numpy array.")
        audio, sr = librosa.load(audio, sr=config.SR)

    # compute the log-mel spectrogram with librosa
    audio_rep = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        hop_length=config.FFT_HOP,
        n_fft=config.FFT_SIZE,
        n_mels=config.N_MELS,
    ).T
    audio_rep = audio_rep.astype(np.float16)
    audio_rep = np.log10(10000 * audio_rep + 1)

    # batch it for an efficient computing
    first = True
    last_frame = audio_rep.shape[0] - n_frames + 1
    # +1 is to include the last frame that range would not include
    for time_stamp in range(0, last_frame, overlap):
        patch = np.expand_dims(audio_rep[time_stamp : time_stamp + n_frames, :], axis=0)
        if first:
            batch = patch
            first = False
        else:
            batch = np.concatenate((batch, patch), axis=0)

    return batch, audio_rep


def extractor(
    audio: str | np.ndarray,
    model: str = "MTT_musicnn",
    input_length: float = 3.0,
    input_overlap: float | bool = False,
    extract_features: bool = True,
):
    """Extract the taggram and features from audio data.

    PARAMETERS
    ----------
    audio: str | np.ndarray, shape (channels, times)
        A path to the audio file or a numpy array containing the audio data.
    model: str, default 'MTT_musicnn'
        Name of the model to use for extraction.
    input_length: float, default 3.0
        Length (in seconds) of the input spectrogram batches.
    input_overlap: float | bool, default False
        Amount of overlap (in seconds) of the input spectrogram batches. If `False`, no
        overlap is applied.
    extract_features: bool, default True
        Whether to extract the intermediate features of the model.

    RETURNS
    -------
    taggram: np.ndarray, shape (batches, tags)
        The taggram containing the temporal evolution of tags.
    labels: list
        List of tags in `taggram`.
    features: dict
        Dictionary containing the intermediate features of the model. Only returned if
        `extract_features=True`.
    """
    # load model
    session, feed_list, extract_vector, n_frames = models.load_model(
        model=model, input_length=input_length
    )

    # batching data
    print("Computing spectrogram (w/ librosa) and tags (w/ tensorflow)...", end=" ")
    batches, _ = batch_data(audio=audio, n_frames=n_frames, input_overlap=input_overlap)

    # tensorflow: extract features and tags
    return extract_features_tags(
        model_name=model,
        batches=batches,
        session=session,
        feed_list=feed_list,
        extract_vector=extract_vector,
        extract_features=extract_features,
        close_session=True,
    )


def extract_features_tags(
    model_name: str,
    batches: np.ndarray,
    session: tf.compat.v1.Session,
    feed_list: list,
    extract_vector: list,
    extract_features: bool = True,
    close_session: bool = False,
) -> tuple[np.ndarray, list] | tuple[np.ndarray, list, dict]:
    """Extract features and tags from the model.

    PARAMETERS
    ----------
    model_name: str
        Name of the model in `session` being used for extraction.
    batches: np.ndarray, shape (batches, times, frequencies)
        Batches of the audio spectrogram to be processed, as returned from
        `extractor.batch_data`.
    session: tf.compat.v1.Session
        TensorFlow session containing the model, as returned from `models.load_model`.
    feed_list: list
        List of information to pass when running the session, as returned from
        `models.load_model`.
    extract_vector: list
        List of information to extract from the model, as returned from
        `models.load_model`.
    extract_features: bool, default True
        Whether to extract the intermediate features of the model.
    close_session: bool, default False
        Whether to close the TensorFlow session after extraction.

    RETURNS
    -------
    taggram: np.ndarray, shape (batches, tags)
        The taggram containing the temporal evolution of tags.
    labels: list
        List of tags in `taggram`.
    features: dict
        Dictionary containing the intermediate features of the model. Only returned if
        `extract_features=True`.
    """
    # ... first batch!
    if not extract_features:
        extract_vector = extract_vector[0]  # only take the tags

    tf_out = session.run(
        extract_vector,
        feed_dict={
            feed_list[0]: batches[: config.BATCH_SIZE],  # x
            feed_list[1]: False,  # is_training
        },
    )

    if extract_features:
        if "vgg" in model_name:
            predicted_tags, pool1_, pool2_, pool3_, pool4_, pool5_ = tf_out
            features = dict()
            features["pool1"] = np.squeeze(pool1_)
            features["pool2"] = np.squeeze(pool2_)
            features["pool3"] = np.squeeze(pool3_)
            features["pool4"] = np.squeeze(pool4_)
            features["pool5"] = np.squeeze(pool5_)
        else:
            (
                predicted_tags,
                timbral_,
                temporal_,
                cnn1_,
                cnn2_,
                cnn3_,
                mean_pool_,
                max_pool_,
                penultimate_,
            ) = tf_out
            features = dict()
            features["timbral"] = np.squeeze(timbral_)
            features["temporal"] = np.squeeze(temporal_)
            features["cnn1"] = np.squeeze(cnn1_)
            features["cnn2"] = np.squeeze(cnn2_)
            features["cnn3"] = np.squeeze(cnn3_)
            features["mean_pool"] = mean_pool_
            features["max_pool"] = max_pool_
            features["penultimate"] = penultimate_
    else:
        predicted_tags = tf_out[0]

    taggram = np.array(predicted_tags)

    # ... rest of the batches!
    for id_pointer in range(config.BATCH_SIZE, batches.shape[0], config.BATCH_SIZE):
        tf_out = session.run(
            extract_vector,
            feed_dict={
                feed_list[0]: batches[id_pointer : id_pointer + config.BATCH_SIZE],  # x
                feed_list[1]: False,  # is_training
            },
        )

        if extract_features:
            if "vgg" in model_name:
                predicted_tags, pool1_, pool2_, pool3_, pool4_, pool5_ = tf_out
                features["pool1"] = np.concatenate(
                    (features["pool1"], np.squeeze(pool1_)), axis=0
                )
                features["pool2"] = np.concatenate(
                    (features["pool2"], np.squeeze(pool2_)), axis=0
                )
                features["pool3"] = np.concatenate(
                    (features["pool3"], np.squeeze(pool3_)), axis=0
                )
                features["pool4"] = np.concatenate(
                    (features["pool4"], np.squeeze(pool4_)), axis=0
                )
                features["pool5"] = np.concatenate(
                    (features["pool5"], np.squeeze(pool5_)), axis=0
                )
            else:
                (
                    predicted_tags,
                    timbral_,
                    temporal_,
                    cnn1_,
                    cnn2_,
                    cnn3_,
                    mean_pool_,
                    max_pool_,
                    penultimate_,
                ) = tf_out
                features["timbral"] = np.concatenate(
                    (features["timbral"], np.squeeze(timbral_)), axis=0
                )
                features["temporal"] = np.concatenate(
                    (features["temporal"], np.squeeze(temporal_)), axis=0
                )
                features["cnn1"] = np.concatenate(
                    (features["cnn1"], np.squeeze(cnn1_)), axis=0
                )
                features["cnn2"] = np.concatenate(
                    (features["cnn2"], np.squeeze(cnn2_)), axis=0
                )
                features["cnn3"] = np.concatenate(
                    (features["cnn3"], np.squeeze(cnn3_)), axis=0
                )
                features["mean_pool"] = np.concatenate(
                    (features["mean_pool"], mean_pool_), axis=0
                )
                features["max_pool"] = np.concatenate(
                    (features["max_pool"], max_pool_), axis=0
                )
                features["penultimate"] = np.concatenate(
                    (features["penultimate"], penultimate_), axis=0
                )
        else:
            predicted_tags = tf_out[0]

        taggram = np.concatenate((taggram, np.array(predicted_tags)), axis=0)

    if model_name.startswith("MTT_"):
        labels = config.MTT_LABELS
    else:
        labels = config.MSD_LABELS

    if close_session:
        session.close()
    print("done!")

    if extract_features:
        return taggram, labels, features
    return taggram, labels

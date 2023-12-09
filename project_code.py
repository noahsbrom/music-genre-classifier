import numpy as np
import subprocess
import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from pydub import AudioSegment
from pydub.playback import play
from PIL import Image

def main():
    np.random.seed(4)

    # generate regular spectrograms
    data_dir = "Data/genres_original"
    wav_files, labels = organize_data(data_dir)
    spec_dir = "spectrograms/"
    generate_spectrograms(spec_dir, wav_files)

    # loudness variant spectrograms
    loudness_wav_files, labels_loud = augment_data_loud(wav_files, labels)
    spec_dir_loud = "spectrograms_loudness/"
    generate_spectrograms(spec_dir_loud, loudness_wav_files)

    # pitch variant spectrograms
    pitch_wav_files, labels_pitch = augment_data_pitch(wav_files, labels)
    spec_dir_pitch = "spectrograms_pitch/"
    generate_spectrograms(spec_dir_pitch, pitch_wav_files)

    # noise variant spectrograms
    spec_dir_noise = "spectrograms_noisy/"
    generate_specs_noise(spec_dir_noise, wav_files)
    labels_noise = np.array([[i]*2 for i in labels]).flatten()

    # run trials
    run_trial(spec_dir, labels)
    run_trial(spec_dir_loud, labels_loud)
    run_trial(spec_dir_pitch, labels_pitch)
    run_trial(spec_dir_noise, labels_noise)


def generate_specs_noise(directory, wav_files):
    sox_path="/opt/homebrew/bin/sox"
    
    for i in range(len(wav_files)):
        output_file = directory + f"spec{i*2}.png"
        command = [sox_path, wav_files[i], "-n", "spectrogram", "-x", "128", "-y", "128", "-r", "-o", output_file]
        subprocess.run(command)

        curr_spec = Image.open(output_file).convert('RGB')
        curr_spec_arr = np.array(curr_spec)
        noisy_spec = np.copy(curr_spec_arr)

        p = 0.05
        for j in range(noisy_spec.shape[0]):
            for k in range(noisy_spec.shape[1]):
                if np.random.uniform(0, 1) < p:
                    noisy_spec[j,k] = 0

        noisy_spec_image = Image.fromarray(noisy_spec)
        new_index = (i*2) + 1
        new_filename = directory + f"spec{new_index}.png"
        noisy_spec_image.save(new_filename)
    
def augment_data_loud(wav_files, labels):
    loudness_dir = "Data/loudness_wavs/"
    loudness_wav_files = []
    for i in range(len(wav_files)):
        file = wav_files[i]

        audio = AudioSegment.from_file(file, format="wav")
        audio_plus = audio + 10
        audio_minus = audio - 10

        wav_plus = loudness_dir + f"audio{i}_plus.wav"
        wav_minus = loudness_dir + f"audio{i}_minus.wav"
        wav_reg = loudness_dir + f"audio{i}_reg.wav"
        audio_plus.export(wav_plus, format="wav")
        audio_minus.export(wav_minus, format="wav")
        audio.export(wav_reg, format="wav")

        loudness_wav_files.append(wav_reg)
        loudness_wav_files.append(wav_minus)
        loudness_wav_files.append(wav_plus)
    loudness_wav_files = np.array(loudness_wav_files)
    labels3 = np.array([[i]*3 for i in labels]).flatten()

    return loudness_wav_files, labels3
    

def augment_data_pitch(wav_files, labels):
    pitch_dir = "Data/pitch_wavs/"
    pitch_wav_files = []
    for i in range(len(wav_files)):
        file = wav_files[i]

        audio = AudioSegment.from_file(file, format="wav")
        pitch_up = int(audio.frame_rate * 2)
        pitch_down = int(audio.frame_rate * 0.5)

        audio_up = audio.set_frame_rate(pitch_up)
        audio_down = audio.set_frame_rate(pitch_down)


        wav_up = pitch_dir + f"audio{i}_up.wav"
        wav_down = pitch_dir + f"audio{i}_down.wav"
        wav_reg = pitch_dir + f"audio{i}_reg.wav"
        audio_up.export(wav_up, format="wav")
        audio_down.export(wav_down, format="wav")
        audio.export(wav_reg, format="wav")

        pitch_wav_files.append(wav_reg)
        pitch_wav_files.append(wav_down)
        pitch_wav_files.append(wav_up)
    pitch_wav_files = np.array(pitch_wav_files)
    labels3 = np.array([[i]*3 for i in labels]).flatten()

    return pitch_wav_files, labels3


def run_trial(directory, labels):
    x_train, y_train, x_test, y_test = split_data(directory, labels)

    cnn_model = train(x_train, y_train, x_test, y_test)
    
    score = cnn_model.evaluate(generate_images(x_test), y_test, verbose=0)
    print("CNN accuracy:", score[1])


""" return 80/20 train/test split of the data """
def split_data(directory, labels):
    n = len(labels)

    specs = [directory + f"spec{i}.png" for i in range(n)]
    labeled_specs= np.column_stack((specs, labels))
    np.random.shuffle(labeled_specs)

    train = labeled_specs[:int(n*0.8)]
    x_train = train[:,0]
    y_train = np.array([int(i) for i in train[:,1]])

    test = labeled_specs[int(n*0.8):]
    x_test = test[:,0]
    y_test = np.array([int(j) for j in test[:,1]])

    return x_train, y_train, x_test, y_test

""" given wav_files, convert to spectrograms and put in directory """
def generate_spectrograms(directory, wav_files):
    sox_path="/opt/homebrew/bin/sox"
    
    for i in range(len(wav_files)):
        output_file = directory + f"spec{i}.png"
        command = [sox_path, wav_files[i], "-n", "spectrogram", "-x", "128", "-y", "128", "-r", "-o", output_file]
        subprocess.run(command)


""" return wav files and corresponding labels as 1000 element arrays """
def organize_data(directory):
    labels = np.array([[i]*100 for i in range(10)]).flatten()
    wav_files = []

    # loop through genres and store each genre's wav files in array
    for file in os.listdir(directory):
        genre_dir= os.path.join(directory, file)
        if os.path.isdir(genre_dir):
            raw_wavs = os.listdir(genre_dir)
            relative_wavs = [genre_dir + "/" + wav for wav in raw_wavs]
            wav_files.append(relative_wavs)
    
    wav_files = np.array([file for genre in wav_files for file in genre])   
    return wav_files, labels


""" return CNN model """
def get_cnn():
    model = models.Sequential()

    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model

""" train CNN model """
def train(x_train, y_train, x_test, y_test):
    cnn_model = get_cnn()
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    cnn_model.compile(optimizer=Adam(learning_rate=0.001),loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    cnn_model.fit(generate_images(x_train), y_train, epochs=100, batch_size=32, validation_data=(generate_images(x_test),y_test), callbacks=[early_stopping])
    return cnn_model


""" convert image paths to valid format for CNN training """
def generate_images(paths):
    images = []
    for path in paths:
        image = img_to_array(load_img(path, color_mode='grayscale'))
        image = image.reshape(128, 128, 1)
        images.append(image)
    return np.array(images)

main()


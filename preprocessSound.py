import os
from types import TracebackType
import tensorflow as tf
from tensorflow._api.v2 import math
import tensorflow_io as tfio
import matplotlib.pyplot as plt 
from SoundAugmentation1 import convert_range_and_rate_of_sound,trim_noise,sound_fade,convert_duration_of_sound_tensor
import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np
from transferFiles import transferFiles

from collections import Counter

def convert_range_of_sound(wav):
    """ this function convert range of sound tensor to range[-0.5,0.5] and convert the size to 10 sec

    Args:
        wav (tensor): tensor of sound file you want change range

    Returns:
        tensor: the new sound tensor after convert range and length
    """
    oldMax = wav.max()
    oldMin = wav.min()
    newMax = 0.5
    newMin = -0.5
    new_Wav = []
    for OldValue in wav:
        OldRange = (oldMax - oldMin)  
        NewRange = (newMax - newMin)  
        NewValue = (((OldValue - oldMin) * NewRange) / OldRange) + newMin
        new_Wav.append(NewValue)
    
    len_of_wav = len(new_Wav)
    len_of_wav_10_second = 171932
    lst = [0.0] * (len_of_wav_10_second-len_of_wav)
    wav_after_convert_durations = np.concatenate((new_Wav, lst))
    b = Counter(wav_after_convert_durations) #counter all values in wav 
    epsilon = round(round(b.most_common(1)[0][0],3)+0.002,3)
    return epsilon,wav_after_convert_durations

def Time_Shifting(wav,start_):
    """shift time of sound file if start_>0 shift to left else shift to right

    Args:
        wav (tensor): sound tensor
        start_ (int): amount of time you want shift

    Returns:
        tensor : sound tensor after shiffting
    """
    # start_ = int(np.random.uniform(-100000,10000))
    # start_ = -100000
    b = Counter(wav) #counter all values in wav 
    # print(b.most_common(1)[0][0])
    if start_ >= 0:
        wav_time_shift = np.r_[wav[start_:], np.random.uniform(b.most_common(1)[0][0],b.most_common(1)[0][0], start_)]
    else:
        wav_time_shift = np.r_[np.random.uniform(b.most_common(1)[0][0],b.most_common(1)[0][0], -start_), wav[:start_]]
    return wav_time_shift

def trim_noise_manually(tensor,epsilon,show_figure):
    """trim the noise from sound file but you must convert the rate & range

    Args:
        tensor (tensor): tensor of data file after convert the rate & range
        dir(string): dir when you want save
        filename (String): name of sound file you need trim noise
        filetype(String): type of sound file
        show_figure(boolean) :if you want show figure put true

    Returns:
        tensor: tensor of sound file after write and trim the noise, after durations
    """
    # filename = "{}_trim".format(filename)
    print(epsilon)
    position = tfio.audio.trim(tensor, axis=0,epsilon=epsilon)
    start = position[0]
    stop = position[1]
    processed = tensor[start:stop]
    # tens = convert_duration_of_sound_tensor(processed,dir,filename,filetype)
    if show_figure:
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(tensor)
        plt.title("wave")
        plt.subplot(2,1,2)
        plt.plot(processed)
        plt.title("trim wave")
        plt.show()
    return processed

def shift_time_and_reduce_noise(wav,time,show_figure):
    """this function shift time of sound tensor then reduce the noise and drow orginal wave ,
    after shift,after reduce noise

    Args:
        wav (tensor): sound tensor
        time (int): time you want shift if time>0 shift to left else shift to right
        show_figure (boolean) : if you want show figure put true

    Returns:
        tensor : sound tensor after shift and reduce the noise
    """
    new_wave = Time_Shifting(wav,time)
    _,new_wave = convert_range_of_sound(new_wave)
    reduced_noise = nr.reduce_noise(y=new_wave, sr=16000,freq_mask_smooth_hz=100,n_fft=512)
    epsilon,reduced_noise = convert_range_of_sound(reduced_noise)
    if show_figure:
        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(wav)
        plt.title("orginal wave")

        plt.subplot(3,1,2)
        plt.plot(new_wave)
        plt.title("after shift")

        plt.subplot(3,1,3)
        plt.plot(reduced_noise)
        plt.title("after reduce noise")
        plt.show()

    return epsilon,reduced_noise

def show_sound_files(dir_class_goal):
    """drow the sound files after trim noise , fade_1 ,fade_2 ,
    fade_3, after_Range_and_rate , after_reduce_noise

    Args:
        dir_class_goal (String): the dir of files you want show sounds files
    """
    for count_folder, folder_sound in enumerate(os.listdir(dir_class_goal)):
        # when you want continue to specefic number 
        # if count_folder < 191:
        #      continue
        
        dir = "{}/{}".format(dir_class_goal,folder_sound)
        print("folder_sound :",folder_sound)
        for count, file in enumerate(os.listdir(dir)):
            if "trim_after_convertduration" in file:
                file_dir = "{}/{}".format(dir,file)
                trim, sr = librosa.load(file_dir, sr=None)
            elif "fade_3_after_convertduration" in file: 
                file_dir = "{}/{}".format(dir,file)
                fade_3, sr = librosa.load(file_dir, sr=None)
            elif "fade_2_after_convertduration" in file:
                file_dir = "{}/{}".format(dir,file) 
                fade_2, sr = librosa.load(file_dir, sr=None)
            elif "fade_1_after_convertduration" in file: 
                file_dir = "{}/{}".format(dir,file)
                fade_1, sr = librosa.load(file_dir, sr=None)
            elif "after_Range_and_rate" in file: 
                file_dir = "{}/{}".format(dir,file)
                audio, sr = librosa.load(file_dir, sr=None)
            elif "after_reduce_noise" in file:
                file_dir = "{}/{}".format(dir,file)
                after_reduce_noise, sr = librosa.load(file_dir, sr=None)
    
        plt.figure(figsize=(20,20))
        plt.subplot(6,1,1)
        plt.plot(audio)
        plt.title("{} : audio orginal {}".format(count_folder+1,folder_sound),pad=10.0)

        plt.subplot(6,1,2)
        plt.plot(after_reduce_noise)
        plt.title("audio after reduce noise")

        plt.subplot(6,1,3)
        plt.plot(trim)
        plt.title("audio after trim ")

        plt.subplot(6,1,4)
        plt.plot(fade_1)
        plt.title("audio after fade1")

        plt.subplot(6,1,5)
        plt.plot(fade_2)
        plt.title("audio after fade2")

        plt.subplot(6,1,6)
        plt.plot(fade_3)
        plt.title("audio after fade3")
        plt.show()

def write_trim_after_correct_handle(dir,name_sound_file,time_shift_right,time_shift_left,show_figure):
    """correct the trim manual and rewrite fades 

    Args:
        dir (String): dir when your {name_sound_file}_sound_after_reduece_noise.wav read
        name_sound_file (String): number of sound file you want exchanged
        time_shift_right (int): time you want shift must be < 0
        time_shift_left ([type]): time you want shift must be > 0
        show_figure (boolean) : if you whant show figure put true
    """
    EPS = 1e-8
    # dir = '{}/{}'.format(dir_goal,"cts3")
    # name_sound_file= "217"
    file_path = '{}/{}/{}_after_reduce_noise.wav'.format(dir,name_sound_file,name_sound_file)
    wav, sr = librosa.load(file_path, sr=None)
    new_dir = "{}/new/{}/".format(dir,name_sound_file)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    
    new_dir_wave = '{}/{}_trim_after_convertduration.wav'.format(new_dir,name_sound_file)
    _,new_wave_front = shift_time_and_reduce_noise(wav,time_shift_right,show_figure)
    epsilon,new_wave = shift_time_and_reduce_noise(new_wave_front,time_shift_left,show_figure)
    
    
    trim = trim_noise_manually(new_wave,epsilon,show_figure)
    _,trim_sound_file = convert_range_of_sound(trim)
    sf.write(new_dir_wave, trim_sound_file, 16000)
    # tens = convert_duration_of_sound_tensor(processed,dir,filename,filetype)
    fade1,fade2,fade3 = sound_fade(trim,new_dir,name_sound_file,filetype="wav",i=1)
    if show_figure:
        plt.figure()
        plt.subplot(5,1,1)
        plt.plot(wav)
        plt.title(" audio orginal",pad=10.0)
        plt.subplot(5,1,2)
        plt.plot(trim_sound_file)
        plt.title("audio after Trim",pad=10.0)
        plt.subplot(5,1,3)
        plt.plot(fade1)
        plt.title("audio after fade1 ",pad=10.0)
        plt.subplot(5,1,4)
        plt.plot(fade2)
        plt.title("audio after fade2",pad=10.0)
        plt.subplot(5,1,5)
        plt.plot(fade3)
        plt.title("audio after fade3",pad=10.0)
        plt.show()

def create_sounds_preprocess_files(commands,dir_dataset,dir_goal):
    for i,sound_class in enumerate(commands):
        dir_class = '{}/{}'.format(dir_dataset,sound_class)
        dir_class_goal = "{}/{}".format(dir_goal,sound_class)
        print("in folder :",dir_class," goal folder :",dir_class_goal,"\n")
        for count, file in enumerate(os.listdir(dir_class)):
            filename ,filetype=file.split('.')
            print("filename ",filename," ",count, "\n")
            directory = "{}/{}".format(dir_class_goal,filename)
            if not os.path.exists(directory):
                os.makedirs(directory)
            soundfile = "{}/{}".format(dir_class,file)
            # audio = tfio.audio.AudioIOTensor(soundfile)
            # audio_slice= audio[100:]
            # audio_tensor = tf.squeeze(audio_slice,axis=[-1])
            # tensor = tf.cast(audio_tensor,tf.float32)/32768.0
            # # 1)
            file_after_rate = convert_range_and_rate_of_sound(soundfile,directory,filename,filetype)
            audio_after_rate = tfio.audio.AudioIOTensor(file_after_rate)
            audio_slice_after_rate = audio_after_rate[100:]
            audio_tensor_after_rate = tf.squeeze(audio_slice_after_rate,axis=[-1])
            tensor_after_rate= tf.cast(audio_tensor_after_rate,tf.float32)/32768.0
            # *
            file_after_reduced_noise = "{}/{}_after_reduce_noise.{}".format(directory,filename,filetype)
            #$
            reduced_noise = nr.reduce_noise(y=tensor_after_rate, sr=16000,stationary=False,win_length=512,n_fft=512,clip_noise_stationary=True, n_jobs=2)
            # sf.write(file_after_reduced_noise, reduced_noise, 16000)
            _,reduced_noise_with_rerange = convert_range_of_sound(reduced_noise)
            sf.write(file_after_reduced_noise, reduced_noise_with_rerange, 16000)
            # *
            # # 2)
            try:
                tensor_after_trim = trim_noise(reduced_noise_with_rerange,directory,filename,filetype)
            except:
                print("the error in {} trim file ==0".format(file))
            file_name_trim = "{}_trim_after_convertduration".format(filename)
            # trim = convert_duration_of_sound_tensor(tensor_after_trim,directory,file_name_trim,filetype)
            _,trim = convert_range_of_sound (tensor_after_trim)
            sf.write("{}/{}.wav".format(directory,file_name_trim), trim, 16000)
            # # 3)
            try:
                fade1,fade2,fade3 = sound_fade(tensor_after_trim,directory,filename,filetype,1)
            except:
                print("the error in {} ".format(file))

        
        print("files finish in class {} : {}\n".format(sound_class,count+1))

def show_sound_files_after_hand_modify(dir_class_goal,list):
    """drow the sound files after trim noise , fade_1 ,fade_2 ,fade_3 

    Args:
        dir_class_goal (String): the dir of files you want show sounds files
    """
    for count_folder, folder_sound in enumerate(os.listdir(dir_class_goal)):
        if folder_sound in list:
            dir = "{}/{}".format(dir_class_goal,folder_sound)
            print("folder_sound :",folder_sound)
            for count, file in enumerate(os.listdir(dir)):
                if "trim_after_convertduration" in file:
                    file_dir = "{}/{}".format(dir,file)
                    trim, sr = librosa.load(file_dir, sr=None)
                elif "fade_3_after_convertduration" in file: 
                    file_dir = "{}/{}".format(dir,file)
                    fade_3, sr = librosa.load(file_dir, sr=None)
                elif "fade_2_after_convertduration" in file:
                    file_dir = "{}/{}".format(dir,file) 
                    fade_2, sr = librosa.load(file_dir, sr=None)
                elif "fade_1_after_convertduration" in file: 
                    file_dir = "{}/{}".format(dir,file)
                    fade_1, sr = librosa.load(file_dir, sr=None)

            plt.figure()
            plt.subplot(4,1,1)
            plt.plot(trim)
            plt.title("audio after trim ")

            plt.subplot(4,1,2)
            plt.plot(fade_1)
            plt.title("audio after fade1")

            plt.subplot(4,1,3)
            plt.plot(fade_2)
            plt.title("audio after fade2")

            plt.subplot(4,1,4)
            plt.plot(fade_3)
            plt.title("audio after fade3")
            plt.show()
        else:
            continue



dir_dataset = 'DataSetSounds'
dir_goal = "soundAugmentation1Dataset"
# commands = ['cts','elearning','email','employeeservices', 'eserve','events', 'facultyservices','mainsite','mawared','news', 'studentservices' , 'tawasol']
len_of_classes = [221,300,314,163,156,116,163,174,149,162,147,168]
commands = ["eserve1"]

# for x in commands:
#     class_dir = "{}/{}".format(dir,x)
#     goal_dir = "{}/{}".format(dir_goal,x)
#     dir_class.append(class_dir)
#     dir_class_goal.append(goal_dir)


# 1)
create_sounds_preprocess_files(commands,dir_dataset,dir_goal)

# ====================================================================== 

# 2)
# run to show the sounds files in folder : 

for i,sound_class in enumerate(commands):
    dir_class = '{}/{}'.format(dir_dataset,sound_class)
    dir_class_goal = "{}/{}".format(dir_goal,sound_class)
    
    print("in folder :",dir_class," goal folder :",dir_class_goal,"\n")
    show_sound_files(dir_class_goal)

# =============================================================================

# 3)

# # to correct the trim manual

# # read file from dir = "soundAugmentation1Dataset/{dir}/{name_sound_file}/{{name_sound_file}_after_reduce_noise}
# # write the trim and fades in same dir but in {new} file 

dir = '{}/{}'.format(dir_goal,"eserve1")

# # 0
# name_sound_file= "72"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -110000,112000,True)

# # 1
# name_sound_file= "149"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,110000,True)
# # 2
# name_sound_file= "135"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -90000,101000,0.13,False)
# # 3
# name_sound_file= "4"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -133000,110000,-0.04,False)
# # 4
# name_sound_file= "76"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,110000,0.06,False)
# # 5
# name_sound_file= "132"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -110000,130000,0.0,False)
# # 6
# name_sound_file= "57"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,101000,0.08,False)
# # 7
# name_sound_file= "102"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,110000,0.01,False)
# # 8
# name_sound_file= "152"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -80000,101000,0.03,False)
# # 9
# name_sound_file= "64"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,101000,0.01,False)
# # 10
# name_sound_file= "120"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,110000,0.1,False)
# # 11
# name_sound_file= "67"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -110000,101000,0.06,False)
# # 12
# name_sound_file= "122"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -135000,149000,0.15,False)
# # 13
# name_sound_file= "79"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -90000,101000,0.0251,False)
# # 14
# name_sound_file= "7"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -90000,101000,0.01,False)
# # 15
# name_sound_file= "48"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -110000,101000,0.06,False)
# # 16
# name_sound_file= "44"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,101000,0.05,False)
# # 17
# name_sound_file= "113"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -95000,120000,0.01,False)
# # 18
# name_sound_file= "23"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -80000,110000,0.09,False)
# # 19
# name_sound_file= "20"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,101000,0.12,False)
# # 20
# name_sound_file= "164"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,110000,0.035,False)
# # 21
# name_sound_file= "206"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,110000,0.06,False)
# # 22
# name_sound_file= "171"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,101000,0.022,False)
# # 23
# name_sound_file= "192"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,110000,0.05,False)
# # 24
# name_sound_file= "182"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,101000,0.01,False)
# # 25
# name_sound_file= "188"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,110000,0.025,False)
# # 26
# name_sound_file= "33"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -120000,125000,0.03,False)
# # 27
# name_sound_file= "61"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,125000,0.04,False)
# # 28
# name_sound_file= "25"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,101000,0.03,False)
# # 29
# name_sound_file= "108"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,121000,0.02,False)
# # 30
# name_sound_file= "11"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,110000,0.08,False)
# # 31
# name_sound_file= "103"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,100000,0.07,False)
# # 32
# name_sound_file= "71"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,103000,0.06,False)
# # 33
# name_sound_file= "175"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -90000,90000,0.055,False)
# # 34
# name_sound_file= "178"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,101000,0.01,False)
# # 35
# name_sound_file= "109"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -110000,101000,0.125,False)
# # 36
# name_sound_file= "60"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,101000,0.05,False)
# # 37
# name_sound_file= "42"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -110000,125000,0.03,False)
# # 38
# name_sound_file= "127"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,100000,0.04,False)
# # 39
# name_sound_file= "55"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -90000,90000,0.052,False)
# # 40
# name_sound_file= "190"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -120000,110000,0.08,False)
# # 41
# name_sound_file= "104"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,120000,0.02,False)
# # 42
# name_sound_file= "19"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -70000,110000,0.0,False)
# # 43
# name_sound_file= "212"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,120000,0.02,False)
# # 44
# name_sound_file= "201"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,101000,-0.06,False)
# # 45
# name_sound_file= "145"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,110000,0.04,False)
# # 46
# name_sound_file= "207"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,101000,0.03,False)
# # 47
# name_sound_file= "10"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,110000,0.03,False)
# # 48
# name_sound_file= "126"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -80000,101000,0.05,False)
# # 49
# name_sound_file= "100"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,102000,0.09,False)
# # 50
# name_sound_file= "112"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,110000,0.01,False)
# # 51
# name_sound_file= "78"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -90000,101000,0.06,False)
# # 52
# name_sound_file= "1"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -127000,127000,0.06,False)
# # 53
# name_sound_file= "174"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -120000,120000,0.05,False)
# # 54
# name_sound_file= "83"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -90000,101000,0.03,False)
# # 55
# name_sound_file= "30"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,101000,0.03,False)
# # 56
# name_sound_file= "45"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,101000,0.03,False)
# # 57
# name_sound_file= "6"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -90000,100000,0.05,False)
# # 58
# name_sound_file= "65"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,110000,0.031,False)
# # 59
# name_sound_file= "125"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,101000,0.08,False)
# # 60
# name_sound_file= "210"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,110000,-0.04,False)
# # 61
# name_sound_file= "157"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,110000,0.07,False)
# # 62
# name_sound_file= "73"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,101000,0.07,False)
# # 63
# name_sound_file= "98"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,110000,0.06,False)
# # 64
# name_sound_file= "28"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -80000,101000,0.05,False)
# # 65
# name_sound_file= "202"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,110000,0.02,False)
# # 66
# name_sound_file= "172"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -90000,120000,0.03,False)
# # 67
# name_sound_file= "66"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,110000,0.082,False)
# # 68
# name_sound_file= "95"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -110000,120000,-0.08,False)
# # 69
# name_sound_file= "139"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,101000,0.07,False)
# # 70
# name_sound_file= "176"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,110000,0.06,False)
# # 71
# name_sound_file= "116"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -80000,120000,0.0,False)
# # 72
# name_sound_file= "159"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,103000,0.03,False)
# # 73
# name_sound_file= "85"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -90000,90000,0.15,False)
# # 74
# name_sound_file= "137"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -120000,101000,0.07,False)
# # 75
# name_sound_file= "187"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,101000,0.01,False)
# # 76
# name_sound_file= "194"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,101000,0.01,False)
# # 77
# name_sound_file= "162"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -123000,130000,0.07,False)
# # 78
# name_sound_file= "88"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -80000,90000,0.04,False)
# # 79
# name_sound_file= "209"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -90000,90000,0.052,False)

# # 80
# name_sound_file= "205"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -70000,70000,0.01,False)
# # 81
# name_sound_file= "199"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,110000,0.053,False)
# # 82
# name_sound_file= "63"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,101000,0.04,False)
# # 83
# name_sound_file= "154"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,110000,0.01,False)
# # 84
# name_sound_file= "47"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,101000,0.037,False)
# # 85
# name_sound_file= "153"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,110000,0.04,False)
# # 86
# name_sound_file= "197"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,110000,0.01,False)
# # 87
# name_sound_file= "123"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,110000,0.02,False)
# # 88
# name_sound_file= "183"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,101000,0.03,False)
# # 89
# name_sound_file= "138"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,120000,0.05,False)
# # 90
# name_sound_file= "130"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,110000,0.047,False)
# # 91
# name_sound_file= "213"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -133000,140000,0.11,False)
# # 92
# name_sound_file= "92"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,100000,0.01,False)
# # 93
# name_sound_file= "80"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -90000,101000,-0.04,False)
# # 94
# name_sound_file= "217"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -90000,101000,0.05,False)
# # 95
# name_sound_file= "75"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -90000,101000,0.025,False)
# # 96
# name_sound_file= "208"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -70000,101000,0.05,False)
# # 97
# name_sound_file= "51"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -110000,110000,0.01,False)
# # 98
# name_sound_file= "59"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,80000,0.02,False)
# # 99
# name_sound_file= "86"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,110000,0.01,False)
# # 100
# name_sound_file= "0"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,80000,0.01,False)
# # 101
# name_sound_file= "168"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,110000,-0.03,False)
# # 102
# name_sound_file= "195"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,110000,0.01,False)
# # 103
# name_sound_file= "22"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,101000,0.04,False)
# # 104
# name_sound_file= "158"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,110000,0.05,False)
# # 105
# name_sound_file= "148"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,125000,-0.01,False)
# # 106
# name_sound_file= "93"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,110000,0.05,False)
# # 107
# name_sound_file= "216"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,101000,0.05,False)
# # 108
# name_sound_file= "90"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -95000,110000,0.04,False)
# # 109
# name_sound_file= "142"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -80000,101000,0.05,False)
# # 110
# name_sound_file= "99"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,101000,0.15,False)
# # 111
# name_sound_file= "128"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -115000,130000,0.07,False)
# # 112
# name_sound_file= "8"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,100000,0.03,False)
# # 113
# name_sound_file= "21"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,103000,-0.05,False)
# # 114
# name_sound_file= "91"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -90000,100000,0.031,False)
# # 115
# name_sound_file= "180"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -90000,101000,0.01,False)
# # 116
# name_sound_file= "16"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -90000,101000,0.02,False)
# # 117
# name_sound_file= "32"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -90000,101000,0.05,False)
# # 118
# name_sound_file= "167"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,120000,0.06,False)
# # 119
# name_sound_file= "2"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -80000,90000,0.03,False)
# # 120
# name_sound_file= "191"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -80000,90000,0.06,False)
# # 121
# name_sound_file= "119"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -120000,110000,0.028,False)
# # 122
# name_sound_file= "96"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,110000,0.05,False)
# # 123
# name_sound_file= "204"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,100000,0.1,False)
# # 124
# name_sound_file= "200"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,110000,0.08,False)
# # 125
# name_sound_file= "15"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -80000,101000,0.01,False)
# # 126
# name_sound_file= "161"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,110000,0.01,False)
# # 127
# name_sound_file= "146"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,121000,0.025,False)
# # 128
# name_sound_file= "89"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,110000,0.0,False)
# # 129
# name_sound_file= "81"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,120000,0.01,False)
# # 130
# name_sound_file= "82"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,102000,0.0,False)
# # 131
# name_sound_file= "70"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,110000,0.01,False)
# # 132
# name_sound_file= "184"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -100000,101000,0.01,False)
# # 133
# name_sound_file= "211"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -110000,101000,0.03,False)
# # 134
# name_sound_file= "198"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -138000,125000,0.12,False)
# # 135
# name_sound_file= "141"
# print(name_sound_file)
# write_trim_after_correct_handle(dir,
#                                 name_sound_file,
#                                 -110000,101000,0.06,False)

# ===============================================================================

# # 4
 # # show file hand modify:
# print("the files after augmented")
# list = ["72", "149","135","4","76", "132","57","102","152","64","120","67","122","79","7","48","44","113","23","20"]
# list1 =["164", "206","171","192","182", "188","33","61","25","108","11","103","71","175","178","109","60","42","127","55"]
# list2 = ["190", "104","19","212","201", "145","207","10","126","100","112","78","1","174","83","30","45","6","65","125"]
# list3 =["210", "157","73","98","28", "202","172","66","95","139","176","116","159","85","137","187","194","162","88","209"]
# list4 = ["205", "199","63","154","47", "153","197","123","183","138","130","213","92","80","217","75","208","51","59","86"]
# list5 =["0", "168","195","22","158", "148","93","216","90","142","99","128","8","21","91","180","16","32","167","2"]
# list6 =["191", "119","96","204","200", "15","161","146","89","81","82","70","184","211","198","141"]
# dir = '{}/{}/new'.format(dir_goal,"cts1")
# show_sound_files_after_hand_modify(dir,list)
# show_sound_files_after_hand_modify(dir,list1)
# show_sound_files_after_hand_modify(dir,list2)
# show_sound_files_after_hand_modify(dir,list3)
# show_sound_files_after_hand_modify(dir,list4)
# show_sound_files_after_hand_modify(dir,list5)
# show_sound_files_after_hand_modify(dir,list6)

# ==================================================================================

# # 5)
# # transfer files after hand modify and merge the orginal files in dir {class}Clear:

# print('step 5 start')
# transferFiles('cts')

# ==================================================================================

# # 6)
# # cheack files after modify

# dir_goal = "{}/ctsClear".format(dir_goal)
# show_sound_files(dir_goal)





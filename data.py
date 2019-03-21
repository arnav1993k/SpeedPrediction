import os
import numpy as np
import cv2
import torch.utils.data as data
class Dataset(data.Dataset):
    def __init__(self, path, imsize, crop_size=(160,420), stride = 5, time_frame=10):
        self.path = path
        self.frames = sorted(os.listdir(path + "frames/"), key=self._file_sort)
        self.imsize = imsize
        self.crop_size = crop_size
        self.time_frame = time_frame
        self.labels = open(path + "train.txt").readlines()
        self.step_size = stride
        self.mean_RGB = np.array([123.68, 116.779, 103.939])

    def _file_sort(self, x):
        return int(x.split(".")[0])

    def _normalize(self, img):
        return (img - self.mean_RGB) / 255

    def _resize(self,img,size):
        return cv2.resize(img.copy(), *size)

    def __len__(self):
        'Denotes the total number of samples'
        return int(len(self.frames)/self.step_size) - self.time_frame

    def crop_frame(self, frame, frame_height, frame_width):
        '''Since the sky and the hood of the car doesnt
        stay still for most of the frames we remove those components
        with a little variance. We detect the approximate beginning
        points of the crop(192,99) and then introduce some randomness
        of +-5 in it'''
        if frame_height <= 0 or frame_height > len(frame):
            raise ValueError("frame_height out of scope")
        if frame_width <= 0 or frame_width > len(frame[0]):
            raise ValueError("frame_width out of scope")
        x_start = np.random.randint(187, 197)
        y_start = np.random.randint(94, 104)
        return frame[x_start:x_start + frame_height, y_start:y_start + frame_width]

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        x = np.empty((self.time_frame, *self.imsize, 3))
        for i in range(index, index + self.time_frame):
            frame = cv2.imread(os.path.join(self.path, "frames", self.frames[i*self.step_size]))
            frame = self.crop_frame(frame, *self.cropsize)
            frame = self._normalize(frame)
            frame = self._resize(frame,*self.imsize)
            x[i - index] = frame
        y = np.float32(self.labels[index + self.time_frame*self.step_size][:-1])
        x = np.transpose(x, [0, 3, 1, 2])
        x = np.float32(x)
        return x, y
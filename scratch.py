import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from pickle import load
import pandas as pd

def test1():
    nFrames = 23*59+3
    fPeriod = 20

    assert nFrames % fPeriod == 0

    cap = cv2.VideoCapture('free_will_ch1.MOV')

    frames = np.zeros((nFrames,1000,2000),dtype=np.uint8)

    for i in tqdm(range(nFrames)):

        success,frame = cap.read()
        
        if not success:
            print('could not grab frame')
            break
            exit()
        
        frameGs = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frameGsRr = cv2.resize(frameGs,(2000,1000))

        frames[i,:,:] = frameGsRr[:,:]

    # approach 1: sum of squared frame deltas

    framesCopy = frames.copy()

    print(f'framesCopy.shape={framesCopy.shape}')

    deltaFrames = np.concatenate((np.zeros((1,1000,2000)),framesCopy[1:] - framesCopy[:-1]), axis=0)

    print(f'deltaFrames.shape={deltaFrames.shape}')

    sumSquaredDeltaFrames = np.sum(deltaFrames ** 2,axis=(1,2))

    print(f'sumSquaredDeltaFrames.shape={sumSquaredDeltaFrames.shape}')

    sumSquaredDeltaFramesAve = np.sum(sumSquaredDeltaFrames.reshape((sumSquaredDeltaFrames.shape[0] // fPeriod,fPeriod)),axis=1)

    print(f'sumSquaredDeltaFramesAve.shape={sumSquaredDeltaFramesAve.shape}')

    plt.plot(sumSquaredDeltaFramesAve)
    plt.show()

if __name__ == '__main__':

    pass

import numpy as np
import cv2

from desktopmagic.screengrab_win32 import getRectAsImage
from win32 import win32api
from win32 import win32gui
from win32 import win32console as win32con
import winGuiAuto

class InputManager:
    """
    in: n/a                                 \n
    out: numpy array with realtime video
    """
    
    def __init__(self, outDim, inputtype, winKeyword=None, crop=None):
        self.inputtype = inputtype
        self.outDim = outDim

        if self.inputtype == 'test':
            self.capture = cv2.VideoCapture('free_will_ch1.MOV')
            self.getFrame = self.getFrameCap

        if self.inputtype == 'win':
            self.hwnd = winGuiAuto.findTopWindow(winKeyword)
            self.crop = crop
            self.getFrame = self.getFrameWin

            print(f'initialized InputManager with target win hwnd={self.hwnd}')

    def getFrameWin(self):
        # get corner coordinates of capture window
        position = win32gui.GetWindowRect(self.hwnd)

        # save pixels into array
        frame = getRectAsImage(position)
        frame = np.array(frame)
        
        frame = frame[self.crop[1]:self.crop[3] , self.crop[0]:self.crop[2]]
        frameGs = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frameGsRr = cv2.resize(frameGs,self.outDim)

        return frameGsRr

    def getFrameCap(self):
        success,frame = self.capture.read()

        if not success:
            print('could not grab frame')
            return None
        
        frameGs = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frameGsRr = cv2.resize(frameGs,self.outDim)

        return frameGsRr

    def getFrames(self, nFrames=1):
        """
        nFrames: number of frames to read
        """
        assert nFrames > 0

        frames = np.zeros((np.array([nFrames] + list(self.outDim)[::-1])),dtype=np.uint8)

        for i in range(nFrames):
            frame = self.getFrame()

            if frame is None:
                return None
                
            frames[i] = frame

        return frames

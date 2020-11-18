import numpy as np
import cv2
import threading
import matplotlib.pyplot as plt
import time

def live_plotter_xy(x_vec,y1_data,line,xLabel='',yLabel='',title='',pause_time=0.01):
    if line is None:
        plt.ion()
        fig = plt.figure(figsize=(13,6))
        ax = fig.add_subplot(111)
        line, = ax.plot(x_vec,y1_data,'r-o',alpha=0.8)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.title(title)
        plt.show()
        
    line.set_data(x_vec,y1_data)
    plt.xlim(np.min(x_vec),np.max(x_vec))
    if np.min(y1_data)<=line.axes.get_ylim()[0] or np.max(y1_data)>=line.axes.get_ylim()[1]:
        plt.ylim([np.min(y1_data)-np.std(y1_data),np.max(y1_data)+np.std(y1_data)])

    plt.pause(pause_time)
    
    return line

class AnalyzerThread(threading.Thread):
    """
    input: InputManager                         \n
    output: single image of each new spread
    """

    def __init__(self, InputManager, transIngressQueue, frameAverage=60):
        threading.Thread.__init__(self)

        self.InputManager = InputManager
        self.transIngressQueue = transIngressQueue
        self.frameAverage = frameAverage

        self.killed = False

    def run(self):
        frameCountS = 0
        frameCountE = 0
        sTime = time.time()
        timeHistory = []
        motionScoreHistory = []
        line = None

        pageTurned = False
        converged = False

        while not self.killed:
            frames = self.InputManager.getFrames(self.frameAverage)

            if frames is None:
                break

            frameCountS = frameCountE
            frameCountE += self.frameAverage

            motionScore = self.getMotionScore(frames)
            motionScoreHistory.append(motionScore)
            timeHistory.append(round(time.time()-sTime,1))

            if not pageTurned and motionScore > (0.8 * (10 ** 9)):
                pageTurned = True
            
            if all(map(lambda s: s < (0.6 * (10 ** 9)), motionScoreHistory[:-7:-1])):
                converged = True
            else:
                converged = False

            if pageTurned and converged:
                print('\nFOUND NEW SPREAD, **PASS FRAME TO TRANS THREAD AND RESET STATE MACHINE**\n')
                
                # choose best frame (CAN BE IMPROVED -- BUT IS IT NECESSARY?)
                bestFrame = frames[0]

                # pass to translation thread
                self.transIngressQueue.put(bestFrame)

                pageTurned = False

            line = live_plotter_xy(timeHistory,motionScoreHistory,line, 'time', 'motion score', 'Motion Score over Time')

            print(f'frames=[{frameCountS},{frameCountE}]   motion score={motionScore}      ave fps={round(frameCountE / (time.time() - sTime), 2)}     pt={pageTurned}   cv={converged}')
        

    def getMotionScore(self, frames):
        deltaFrames = frames[1:] - frames[:-1]

        motionScore = np.sum(deltaFrames ** 2)

        return motionScore

    def kill(self):
        self.killed = True
        print('analyzer killed')
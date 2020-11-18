import threading
import queue
import time
import cv2
import numpy as np
import pandas as pd
from pickle import dump

class TranslatorThread(threading.Thread):
    """
    input: trans ingress queue (images of new spreads)  \n
    output: text output in file (txt,pdf,etc)
    """

    def __init__(self, ocrModule, parserModule, transIngressQueue):
        threading.Thread.__init__(self)

        self.OCR = ocrModule
        self.Parser = parserModule
        self.ingressQueue = transIngressQueue
        self.killed = False

    def run(self):
        while not self.killed:
            found = False
            # wait until next item, then get it
            try:
                frame = self.ingressQueue.get(timeout=1)
                found=True
            except queue.Empty:
                time.sleep(1)

            if found:
                data = self.OCR.processImage(frame)
                self.putDataOnFrame(frame, data)

    def putDataOnFrame(self, frame, data):
        outFrame = cv2.cvtColor(frame.copy(), cv2.COLOR_GRAY2BGR)

        df = pd.DataFrame.from_dict(data)
        with open('firstSpreadData.p','wb') as f:
            dump(df,f)

        numWords = len(data['text'])

        numSkipped = 0
        for i in range(numWords):
            word = data['text'][i]
            left = data['left'][i]
            top = data['top'][i]
            width = data['width'][i]
            height = data['height'][i]
            conf = int(data['conf'][i])

            if conf < 0 or word in ['',' ']:
                # print('skipped ' + word + 'with conf=' + str(conf))
                numSkipped += 1
                continue

            subFrame = outFrame[top:top+height , left:left+width]
            whiteRect = np.ones(subFrame.shape, dtype=np.uint8) * 255
            blended = 0.4 * subFrame + 0.6 * whiteRect
            outFrame[top:top+height , left:left+width] = blended

            cvWord = word.replace("'", '')

            cv2.putText(
                outFrame,
                text=cvWord,
                org=(left,int(top+0.75*height)),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale = 1,
                color=(0,0,255),
                thickness=2,
            )

        #print(f'skipped {numSkipped}/{numWords} words')

        cv2.imshow('text boxes', outFrame)
        cv2.waitKey(1000)
            

    def kill(self):
        self.killed = True
        print('translator killed')
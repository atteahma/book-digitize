from InputManager import InputManager
from Analyzer import AnalyzerThread
from OCR import OCR
from Parser import Parser
from Translator import TranslatorThread

from queue import Queue
import time

if __name__ == '__main__':

    # initialize cross thread communication structs
    transIngressQueue = Queue()

    # initialize pre process modules
    inputmanager = InputManager(
        outDim=(2000,1000), # wxh
        inputtype='test',
        winKeyword='Zoom Meeting',
        crop=(325,62,325+1280,62+960), #x1,y1,x2,y2  (TL,BR)
    )

    # initialize pre process manager thread
    analyzer = AnalyzerThread(
        inputmanager,
        transIngressQueue,
        frameAverage=20,
    )

    # initialize post process modules
    ocr = OCR()
    parser = Parser()

    # initialize post process manager thread
    translator = TranslatorThread(ocr, parser, transIngressQueue)

    # start threads
    analyzer.start()
    translator.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        analyzer.kill()
        translator.kill()

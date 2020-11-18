import numpy as np
import cv2
import pytesseract as tsrct
from PIL import Image

tsrct.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class OCR:
    """
    in: numpy array of new spread image     \n
    out: text and metadata from tesseract
    """

    def __init__(self):
        pass

    def processImage(self, im):
        
        # pre process image for ocr
        preprocIm = self.preprocess(im)

        # do ocr
        ocrData = tsrct.image_to_data(
            image=preprocIm,
            lang='eng',
            config='',
            output_type=tsrct.Output.DICT,
        )

        return ocrData
    
    def preprocess(self, im):

        SHOW_IMS = False

        if SHOW_IMS:
            cv2.imshow('raw', im)
            cv2.waitKey()

        # binarize
        im = self._binarize(im)

        if SHOW_IMS:
            cv2.imshow('binarized', im)
            cv2.waitKey()

        # blur
        #im = self._blur(im)

        if SHOW_IMS:
            cv2.imshow('blur + bin', im)
            cv2.waitKey()

        return im

    def _binarize(self, im):
        
        binCrop = (172,0,172+1500,1000) # x1,y1,x2,y2 (TL, BR)

        # find adequate threshold
        threshIm = im[binCrop[1]:binCrop[3] , binCrop[0]:binCrop[2]]

        optThresh,_ = cv2.threshold(
            threshIm,
            0,
            255,
            cv2.THRESH_OTSU,
        )

        # apply binarization with found threshold
        _,binIm = cv2.threshold(
            im,
            optThresh,
            255,
            cv2.THRESH_BINARY
        )

        return binIm

    def _blur(self, im):
        blurIm = cv2.medianBlur(im,3)

        return blurIm

if __name__ == '__main__':
    # ocr testing

    ocr = OCR()

    im = cv2.imread('firstSpread.jpg')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('raw', im)
    # cv2.waitKey()

    data = ocr.processImage(im)

    # out = ''
    # for t in text:
    #     out += t
    #     out += ' '

    # print(out)

    
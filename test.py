#-------------------------------------------------------------------------------
# Name:        FeatureMatching
# Purpose:
#
# Author:      kadota masayuki
#
# Created:     2019/02/24
# Copyright:   (c) kadota masayuki 2019
# Licence:     BSD Licence
#-------------------------------------------------------------------------------

import numpy as np
import cv2
import math
import os
from pathlib import Path

ENABLE_DEBUG = False

def enableDebug(enableDebug):
    global ENABLE_DEBUG
    ENABLE_DEBUG = enableDebug

def templateMatch1to1(templateImage, sceneImage, method=cv2.TM_CCOEFF_NORMED):
    # method:
    #  'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
    #  'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED'
    result = cv2.matchTemplate(sceneImage, templateImage, method)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
    if (ENABLE_DEBUG):
        print("template match:{0}/{1}/{2}/{3}".format(minVal, maxVal, minLoc, maxLoc))
    if (method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]):
        topLeft = minLoc
        score = 1 / minVal
    else:
        topLeft = maxLoc
        score = maxVal
    resultImage = sceneImage.copy()
    cv2.rectangle(resultImage, (topLeft), (topLeft[0] + templateImage.shape[1], topLeft[1] + templateImage.shape[0]), (0, 255, 0), 3)
    return resultImage, score

def featureMatch1to1(templateImage, sceneImage, thresholdFactor=0.6, knn=False):
    try:
        #特徴抽出機の生成
        detector = cv2.AKAZE_create()

        #kpは特徴的な点の位置 destは特徴を現すベクトル
        kp1, des1 = detector.detectAndCompute(templateImage, None)
        kp2, des2 = detector.detectAndCompute(sceneImage, None)

        #特徴点の比較機
        bf = cv2.BFMatcher()

        if (knn):
            # Kマッチ
            matches = bf.knnMatch(des1, des2, k=2)
            # 2点間の割合試験を適用。隣接する2点のうちdistanceの割合が一定以上大きいもののみ格納する
            good = [[m] for m, n in matches if float(m.distance) < thresholdFactor * float(n.distance)]
            # cv2.drawMatches***は適合している点を結ぶ画像を生成する
            resultImage = cv2.drawMatchesKnn(templateImage, kp1, sceneImage, kp2, good, None, flags=2)
            return resultImage, len(good)
        else:
            # 全マッチ
            matches = bf.match(des1, des2)
            # 良い順 (distance が小さい順) に並べる
            matches = sorted(matches, key = lambda x:x.distance)
            # マッチしたdistanceの最大値との比率でgood配列を作る
            good = []
            if (len(matches) >= 1):
                maxDistance = float(matches[len(matches) - 1].distance)
                # 最小distanceに近いmatch値のみを取得
                good = [i for i in matches if float(i.distance) < thresholdFactor * maxDistance]
            else:
                good = matches
            # cv2.drawMatches***は適合している点を結ぶ画像を生成する
            resultImage = cv2.drawMatches(templateImage, kp1, sceneImage, kp2, good, None, flags=2)
            return resultImage, len(good)
    except:
        return templateImage, 0

def match(scenes):
    # 'm' で、保存したテンプレート郡と一括マッチング (表示後、キー入力待ちで止める)
    guesses = []
    p = Path('./')
    for i in range(len(scenes)):
        maxScore = 0
        maxTemplate = ''
        for f in p.glob('template_*.png'):
            tmpl = cv2.imread(f.as_posix(), 0)
            #guessFrame, score = featureMatch1to1(expand(tmpl, 2, 2), expand(scenes[i], 2, 2), 0.2)
            guessFrame, score = templateMatch1to1(expand(tmpl, 2, 2), expand(scenes[i], 2, 2)) # グレー画像のみ
            if (score > maxScore):
                maxScore = score
                maxTemplate = str(f)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(guessFrame, "scene {0}".format(i), (10, 20), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(guessFrame, "name {0}".format(f), (10, 40), font, 0.5, (0,0,255), 2, cv2.LINE_AA)
            cv2.putText(guessFrame, "score {0}".format(score), (10, 60), font, 0.5, (0,0,255), 2, cv2.LINE_AA)
            cv2.imshow('result_{0}'.format(f), guessFrame)
        if (len(maxTemplate) > len("template_")):
            guesses.append(maxTemplate[len("template_")])
        else:
            guesses.append("")
    return guesses

def addWeight(srcFrame, gain=7.2, gamma=1.2):
    # コントラストとガンマで重み付け
    if (len(srcFrame.shape) >=3):
        h, w, c = srcFrame.shape
        resultFrame = srcFrame.copy()
    else:
        h, w = srcFrame.shape
        c = 1
        resultFrame = np.empty((h, w, 1), dtype=np.uint8)
        resultFrame[:,:,0] = srcFrame
    lookuptable = np.ones((256, 1), np.uint8)
    for i in range(256):
        lookuptable[i] = int(255.0 / (1 + math.exp(-gain * (i - 128) / 255.0)));
    for i in range(c):
        resultFrame[:,:,i] = cv2.LUT(resultFrame[:,:,i], lookuptable)
    if (gamma != 1):
        gammaLookuptable = np.ones((256, 1), np.uint8)
        for i in range(256):
            gammaLookuptable[i] = int(((i / 255.0) ** (1.0 / gamma)) * 255.0);
        for i in range(c):
            resultFrame[:,:,i] = cv2.LUT(resultFrame[:,:,i], gammaLookuptable)
    return resultFrame if c != 1 else resultFrame[:,:,0]

def stretchLevelRange(srcFrame, minLevel=0, maxLevel=255):
    # 色レベルを min - max間 に配置する
    if (len(srcFrame.shape) >=3):
        h, w, c = srcFrame.shape
        resultFrame = srcFrame.copy()
    else:
        h, w = srcFrame.shape
        c = 1
        resultFrame = np.empty((h, w, 1), dtype=np.uint8)
        resultFrame[:,:,0] = srcFrame
    for i in range(c):
        min = np.min(resultFrame[:,:,i])
        max = np.max(resultFrame[:,:,i])
        resultFrame[:,:,i] = (resultFrame[:,:,i] - min) * 255.0 / (max - min)
    return resultFrame if c != 1 else resultFrame[:,:,0]

def arrangeLevelRange(srcFrame, baseLevel=127, minLevel=0, maxLevel=255):
    # レベル補正する(複数画像のレベルを合わせるときに使用)
    if (len(srcFrame.shape) >=3):
        h, w, c = srcFrame.shape
        resultFrame = srcFrame.copy()
    else:
        h, w = srcFrame.shape
        c = 1
        resultFrame = np.empty((h, w, 1), dtype=np.uint8)
        resultFrame[:,:,0] = srcFrame
    lookuptable = np.zeros((256,1), np.uint8)
    for i in range(c):
        mean = int(np.mean(resultFrame[:,:,i]))
        min = int(np.min(resultFrame[:,:,i]))
        max = int(np.max(resultFrame[:,:,i]))
        if (ENABLE_DEBUG):
            std = np.std(resultFrame[:,:,i])
        for j in range(min, mean):
            lookuptable[j][0] = (float(j) - (min - minLevel)) * float(baseLevel - minLevel) / (mean - min)
        for j in range(mean, max + 1):
            lookuptable[j][0] = (float(j) - (mean - baseLevel)) * float(maxLevel - baseLevel) / (max - mean)
        resultFrame[:,:,i] = cv2.LUT(resultFrame[:,:,i], lookuptable)
        if (ENABLE_DEBUG):
            mean2 = int(np.mean(resultFrame[:,:,i]))
            min2 = int(np.min(resultFrame[:,:,i]))
            max2 = int(np.max(resultFrame[:,:,i]))
            std2 = np.std(resultFrame[:,:,i])
            print("arrangeLevel:c={0}, max={1}/{2}, min={3}/{4}, mean={5:.3f}/{6:.3f}, std={7:.3f}/{8:.3f}".format(i, max, max2, min, min2, mean, mean2, std, std2))
            cv2.waitKey(0)
    return resultFrame if c != 1 else resultFrame[:,:,0]

def arrangeLevelStddev(srcFrame, baseLevel=127, stdDevWeight=127):
    # レベル補正する(複数画像のレベルを合わせるときに使用)
    if (len(srcFrame.shape) >=3):
        h, w, c = srcFrame.shape
        resultFrame = srcFrame.copy()
    else:
        h, w = srcFrame.shape
        c = 1
        resultFrame = np.empty((h, w, 1), dtype=np.uint8)
        resultFrame[:,:,0] = srcFrame
    for i in range(c):
        mean = np.mean(resultFrame[:,:,i])
        std = np.std(resultFrame[:,:,i])
        if (ENABLE_DEBUG):
            min = np.min(resultFrame[:,:,i])
            max = np.max(resultFrame[:,:,i])
        resultFrame[:,:,i] = ((resultFrame[:,:,i] - mean) / std) * stdDevWeight + baseLevel
        if (ENABLE_DEBUG):
            mean2 = int(np.mean(resultFrame[:,:,i]))
            min2 = int(np.min(resultFrame[:,:,i]))
            max2 = int(np.max(resultFrame[:,:,i]))
            std2 = np.std(resultFrame[:,:,i])
            print("arrangeLevelStddev:c={0}, max={1}/{2}, min={3}/{4}, mean={5:.3f}/{6:.3f}, std={7:.3f}/{8:.3f}".format(i, max, max2, min, min2, mean, mean2, std, std2))
            cv2.waitKey(0)
    return resultFrame if c != 1 else resultFrame[:,:,0]

def expand(srcFrame, expandRatioX=1.5, expandRatioY=1.5):
    # 画像を伸縮 (各種画像処理前に実施しないとボケる)
    return cv2.resize(srcFrame, None, fx=expandRatioX, fy=expandRatioY)

def gray(srcFrame):
    # グレー画像
    return cv2.cvtColor(srcFrame, cv2.COLOR_BGR2GRAY)

def blur(srcFrame, blurX=1, blurY=1):
    # ボケ(1より大きい奇数)
    if (blurX > 0):
        blurX = int(blurX / 2) * 2 + 1
    if (blurY > 0):
        blurY = int(blurY / 2) * 2 + 1
    if ((blurX > 0) and (blurY > 0)):
        return cv2.GaussianBlur(srcFrame, (blurX, blurY), 0)
    else:
        return srcFrame

def threshold(srcFrame, thresholdBlockSize=3, thresholdConstant=2):
    # 二値化(3より大きい奇数)
    if (thresholdBlockSize >= 3):
        return cv2.adaptiveThreshold(srcFrame, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY,
                                     thresholdBlockSize, thresholdConstant)
    else:
        return srcFrame

def thresholdOtsu(srcFrame):
    # 大津の二値化 blur() してから適用すると良好
    # http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html
    _, resultFrame = cv2.threshold(srcFrame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return resultFrame

def erode(srcFrame, erodeSizeX=1, erodeSizeY=1, erodeTime=1):
    # 黒を拡大
    if (erodeSizeX > 0 and erodeSizeY > 0 and erodeTime > 0):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erodeSizeX, erodeSizeY))
        return cv2.erode(srcFrame, kernel, iterations=erodeTime)
    else:
        return srcFrame

def canny(srcFrame, cannyMaxVal=200, cannyMinVal=100, cannySobelSize=3, cannyGradient=False):
    # エッジ検出(sobelSizeは3より大きい奇数)
    if (cannySobelSize > 0):
        cannySobelSize = int(cannySobelSize / 2) * 2 + 1
    if (cannyMaxVal > 0 and cannyMinVal > 0 and cannySobelSize >= 3):
        return cv2.Canny(srcFrame, cannyMaxVal, cannyMinVal,
                         None, cannySobelSize, cannyGradient)
    else:
        return srcFrame

def invert(srcFrame):
    # 色反転
    return cv2.bitwise_not(srcFrame)

def addMergin(srcFrame, merginSize=40, merginColor=0):
    # グレー画像に余白を付ける(Canny後の背景色は黒)
    if (len(srcFrame.shape) < 3):
        resultFrame = np.ones((h + merginSize * 2, w + merginSize * 2), np.uint8)
        resultFrame[:,:] = merginColor
        resultFrame[merginSize:merginSize + h, merginSize:merginSize + w] = srcFrame
        return resultFrame
    else:
        return srcFrame

def addKeypoints(srcFrame):
    # 特徴点を描画した画像を返す
    detector = cv2.AKAZE_create()
    kp1, des1 = detector.detectAndCompute(srcFrame, None)
    return cv2.drawKeypoints(srcFrame, kp1, None,
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

def main():
    expandRatio = 4
    mergin = 0
    frameTopX = 300
    frameTopY = 200
    frameSizeX = 120
    frameSizeY = 50
    frames = ((300, 200, 330, 240), (330, 200, 360, 240), (360, 200, 390, 240), (390, 200, 420, 240))

    template = np.zeros((frameSizeX, frameSizeY), np.uint8)
    currentFragment = 0
    autoMatch = 0
    while (True):
        ret, srcFrame = cap.read()
        srcViewFrame = srcFrame.copy()
        cv2.rectangle(srcViewFrame,
                      (frameTopX, frameTopY),
                      (frameTopX + frameSizeX, frameTopY + frameSizeY),
                      (128, 128, 0),
                      4)
        for i in range(len(frames)):
            cv2.rectangle(srcViewFrame,
                          (frames[i][0], frames[i][1]),
                          (frames[i][2], frames[i][3]),
                          (0, 128, 255),
                          2)
        srcViewFrame = expand(srcViewFrame, 0.7, 0.7)
        cv2.imshow('view', srcViewFrame)
        
        baseFrame = expand(srcFrame[frameTopY-int(mergin/expandRatio):frameTopY+frameSizeY+int(mergin/expandRatio),
                                 frameTopX-int(mergin/expandRatio):frameTopX+frameSizeX+int(mergin/expandRatio)],
                                 expandRatio, expandRatio)
        cv2.imshow('expended', baseFrame)

        aFrame = baseFrame.copy()
        
        aFrame = gray(aFrame)
        aFrame = erode(aFrame, 5, 5, 1)
        aFrame = addWeight(aFrame, gain=14.2, gamma=4.3)
#        aFrame = arrangeLevelRange(aFrame, baseLevel=130, minLevel=10, maxLevel=245)
        aFrame = stretchLevelRange(aFrame, minLevel=0, maxLevel=255)

#        aFrame = gray(aFrame)
#        aFrame = invert(aFrame)
#        aFrame = threshold(aFrame, thresholdBlockSize=21, thresholdConstant=9)
        aFrame = blur(aFrame, 5, 5)
        aFrame = thresholdOtsu(aFrame)
        aKPFrame = addKeypoints(aFrame)
        cv2.rectangle(aKPFrame,
                      (mergin, mergin),
                      (mergin + int(frameSizeX * expandRatio), mergin + int(frameSizeY * expandRatio)),
                      (128, 128, 0),
                      4)
        scenes = []
        for i in range(len(frames)):
            cv2.rectangle(aKPFrame,
                          (mergin + int((frames[i][0] - frames[0][0]) * expandRatio), mergin + int((frames[i][1] - frames[0][1]) * expandRatio)),
                          (mergin + int((frames[i][2] - frames[0][0]) * expandRatio), mergin + int((frames[i][3] - frames[0][1]) * expandRatio)),
                          (0, 0, 255) if (i == currentFragment) else (0, 255, 128),
                          4 if (i == currentFragment) else 2)
            scenes.append(aFrame[(mergin + int((frames[i][1] - frames[0][1]) * expandRatio)):(mergin + int((frames[i][3] - frames[0][1]) * expandRatio)),
                                 (mergin + int((frames[i][0] - frames[0][0]) * expandRatio)):(mergin + int((frames[i][2] - frames[0][0]) * expandRatio))])
        cv2.imshow('scene', aKPFrame)

        #instantGuessFrame, score = featureMatch1to1(template, aFrame, 0.2)
        instantGuessFrame, score = templateMatch1to1(template, aFrame)
        cv2.imshow('instant guess', instantGuessFrame)
        
        if (autoMatch > 0):
            resultFrame = baseFrame.copy()
            guesses = match(scenes)
            font = cv2.FONT_HERSHEY_SIMPLEX
            for i in range(len(scenes)):
                cv2.putText(resultFrame, "'{0}'".format(guesses[i]),
                            (20 + mergin + int((frames[i][0] - frames[0][0]) * expandRatio), 50 + mergin + int((frames[i][1] - frames[0][1]) * expandRatio)),
                            font, 1.1, (0,255,0), 2, cv2.LINE_AA)
            cv2.imshow('result', resultFrame)
            if (autoMatch == 1):
            	autoMatch = 0

        k = cv2.waitKey(2) & 0xff
        if (k == 27):
            # ESC で終了
            autoMatch = 0
            break
        elif (k >= ord('0') and k <= ord('9')):
            autoMatch = 0
            if (k - ord('0') < len(scenes)):
                currentFragment = k - ord('0')
        elif (k == ord('t')):
            autoMatch = 0
            # 't' でテンプレート取り込み
            template = scenes[currentFragment]
        elif (k == ord('s')):
            autoMatch = 0
            # 's' でテンプレート保存
            k2 = cv2.waitKey(0) & 0xff
            if (k2 != 27):
                # ESC 以外で、その文字と関連付けて保存
                t = chr(k2)
                n = 0
                while (True):
                    # 同一ファイル名なら追番を付与
                    templateFilePath = "./template_" + t + "_" + str(n) + ".png"
                    if (not os.path.exists(templateFilePath)):
                        cv2.imwrite(templateFilePath, template)
                        break
                    n = n + 1
        elif (k == ord('m')):
            autoMatch = 1
        elif (k == ord('a')):
            autoMatch = 2

if __name__ == '__main__':
    cap = cv2.VideoCapture(1)
    main()
    cap.release()
    cv2.destroyAllWindows()

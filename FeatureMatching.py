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

def match1to1(template_image, scene_image, threshold_factor=0.6, knn=False):
    try:
        #特徴抽出機の生成
        detector = cv2.AKAZE_create()

        #kpは特徴的な点の位置 destは特徴を現すベクトル
        kp1, des1 = detector.detectAndCompute(template_image, None)
        kp2, des2 = detector.detectAndCompute(scene_image, None)

        #特徴点の比較機
        bf = cv2.BFMatcher()

        if (knn):
            # Kマッチ
            matches = bf.knnMatch(des1, des2, k=2)
            # 2点間の割合試験を適用。隣接する2点のうちdistanceの割合が一定以上大きいもののみ格納する
            good = [[m] for m, n in matches if float(m.distance) < threshold_factor * float(n.distance)]
            # cv2.drawMatches***は適合している点を結ぶ画像を生成する
            result_image = cv2.drawMatchesKnn(template_image, kp1, scene_image, kp2, good, None, flags=2)
            return result_image, len(good)
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
                good = [i for i in matches if float(i.distance) < threshold_factor * maxDistance]
            else:
                good = matches
            # cv2.drawMatches***は適合している点を結ぶ画像を生成する
            result_image = cv2.drawMatches(template_image, kp1, scene_image, kp2, good, None, flags=2)
            return result_image, len(good)
    except:
        return template_image, 0

def addWeight(srcFrame, gain=7.2, gamma=1.2):
    # コントラストとガンマで重み付け
    resultFrame = srcFrame.copy()
    h, w, c = srcFrame.shape
    lookuptable = np.ones((256, 1), np.uint8)
    for i in range(256):
        lookuptable[i] = int(255.0 / (1 + math.exp(-gain * (i - 128) / 255.0)));
    for i in range(c):
        resultFrame[:,:,i] = cv2.LUT(srcFrame[:,:,i], lookuptable)
    if (gamma != 1):
        gammaLookuptable = np.ones((256, 1), np.uint8)
        for i in range(256):
            gammaLookuptable[i] = int(((i / 255.0) ** (1.0 / gamma)) * 255.0);
        for i in range(c):
            resultFrame[:,:,i] = cv2.LUT(resultFrame[:,:,i], gammaLookuptable)
    return resultFrame
def stretchLevelRange(srcFrame, minLevel=0, maxLevel=255):
    # 色レベルを min - max間 に配置する
    resultFrame = srcFrame.copy()
    h, w, c = srcFrame.shape
    for i in range(c):
        min = np.min(srcFrame[:,:,i])
        max = np.max(srcFrame[:,:,i])
        resultFrame[:,:,i] = (srcFrame[:,:,i] - min) * 255.0 / (max - min)
    return resultFrame
def arrangeLevelRange(srcFrame, baseLevel=127, minLevel=0, maxLevel=255):
    # レベル補正する(複数画像のレベルを合わせるときに使用)
    resultFrame = srcFrame.copy()
    h, w, c = srcFrame.shape
    lookuptable = np.zeros((256,1), np.uint8)
    for i in range(c):
        mean = int(np.mean(srcFrame[:,:,i]))
        min = int(np.min(srcFrame[:,:,i]))
        max = int(np.max(srcFrame[:,:,i]))
        if (ENABLE_DEBUG):
            std = np.std(srcFrame[:,:,i])
        for j in range(min, mean):
            lookuptable[j][0] = (float(j) - (min - minLevel)) * float(baseLevel - minLevel) / (mean - min)
        for j in range(mean, max + 1):
            lookuptable[j][0] = (float(j) - (mean - baseLevel)) * float(maxLevel - baseLevel) / (max - mean)
        resultFrame[:,:,i] = cv2.LUT(srcFrame[:,:,i], lookuptable)
        if (ENABLE_DEBUG):
            mean2 = int(np.mean(resultFrame[:,:,i]))
            min2 = int(np.min(resultFrame[:,:,i]))
            max2 = int(np.max(resultFrame[:,:,i]))
            std2 = np.std(resultFrame[:,:,i])
            print("arrangeLevel:c={0}, max={1}/{2}, min={3}/{4}, mean={5:.3f}/{6:.3f}, std={7:.3f}/{8:.3f}".format(i, max, max2, min, min2, mean, mean2, std, std2))
            cv2.waitKey(0)
    return resultFrame
def arrangeLevelStddev(srcFrame, baseLevel=127, stdDevWeight=127):
    # レベル補正する(複数画像のレベルを合わせるときに使用)
    resultFrame = srcFrame.copy()
    h, w, c = srcFrame.shape
    for i in range(c):
        mean = np.mean(srcFrame[:,:,i])
        std = np.std(srcFrame[:,:,i])
        if (ENABLE_DEBUG):
            min = np.min(srcFrame[:,:,i])
            max = np.max(srcFrame[:,:,i])
        resultFrame[:,:,i] = ((srcFrame[:,:,i] - mean) / std) * stdDevWeight + baseLevel
        if (ENABLE_DEBUG):
            mean2 = int(np.mean(resultFrame[:,:,i]))
            min2 = int(np.min(resultFrame[:,:,i]))
            max2 = int(np.max(resultFrame[:,:,i]))
            std2 = np.std(resultFrame[:,:,i])
            print("arrangeLevelStddev:c={0}, max={1}/{2}, min={3}/{4}, mean={5:.3f}/{6:.3f}, std={7:.3f}/{8:.3f}".format(i, max, max2, min, min2, mean, mean2, std, std2))
            cv2.waitKey(0)
    return resultFrame
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
    if (invert1):
        return cv2.bitwise_not(srcFrame)
    else:
        return srcFrame
def addMergin(srcFrame, merginSize=40, merginColor=0):
    # グレー画像に余白を付ける(Canny後の背景色は黒)
    h,w,c = srcFrame.shape
    if (c == 1):
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
    expandRatio = 3.1
    mergin = 20
    frameTopX = 300
    frameTopY = 200
    frameSizeX = 100
    frameSizeY = 100
    template = np.zeros((frameSizeX, frameSizeY), np.uint8)
    while (True):
        ret, srcFrame = cap.read()
        srcViewFrame = srcFrame.copy()
        cv2.rectangle(srcViewFrame, (frameTopX, frameTopY, frameSizeX, frameSizeY), (128, 128, 0), 2)
        srcViewFrame = expand(srcViewFrame, 0.7)
        cv2.imshow('view', srcViewFrame)
        
        aFrame = expand(srcFrame[frameTopY-int(mergin/expandRatio):frameTopY+frameSizeY+int(mergin/expandRatio),
                                 frameTopX-int(mergin/expandRatio):frameTopX+frameSizeX+int(mergin/expandRatio)],
                                 expandRatio, expandRatio)
        cv2.imshow('expended', aFrame)

        aFrame = blur(aFrame, 1, 1)
        #aFrame = arrangeLevelRange(aFrame, baseLevel=127, minLevel=10, maxLevel=245)
        aFrame = stretchLevelRange(aFrame, minLevel=0, maxLevel=255)
        aFrame = addWeight(aFrame, gain=8.2, gamma=1.2)

        aFrame = gray(aFrame)
        aFrame = erode(aFrame, 2, 2, 1)
        aKPFrame = addKeypoints(aFrame)
        cv2.rectangle(aKPFrame, (mergin, mergin, int(frameSizeX * expandRatio), int(frameSizeY * expandRatio)), (128, 128, 0), 2)
        cv2.imshow('scene', aKPFrame)

        sceneFrame = aFrame

        resultFrame, score = match1to1(template, sceneFrame, 0.6)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(resultFrame, "score {0}".format(score), (10, 30), font, 0.6, (0,0,255), 2, cv2.LINE_AA)
        cv2.imshow('result', resultFrame)

        k = cv2.waitKey(2) & 0xff
        if (k == 27):
            # ESC で終了
            break
        elif (k == ord('t')):
            # 't' でテンプレート取り込み
            template = sceneFrame
        elif (k == ord('s')):
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
            # 'm' で、保存したテンプレート郡と一括マッチング (表示後、キー入力待ちで止める)
            p = Path('./')
            maxScore = 0
            maxTemplate = ''
            for f in p.glob('template_*.png'):
                tmpl = cv2.imread(f.as_posix())
                guessFrame, score = match1to1(tmpl, sceneFrame, 0.6)
                if (score > maxScore):
                    maxScore = score
                    maxTemplate = str(f)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(guessFrame, "name {0}".format(f), (10, 20), font, 0.5, (0,0,255), 2, cv2.LINE_AA)
                cv2.putText(guessFrame, "score {0}".format(score), (10, 40), font, 0.5, (0,0,255), 2, cv2.LINE_AA)
                cv2.imshow('result_{0}'.format(f), guessFrame)
            if (len(maxTemplate) > len("template_")):
                guess = maxTemplate[len("template_")]
            else:
                guess = ""
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(resultFrame, "guess '{0}'".format(guess), (10, 60), font, 0.6, (0,0,255), 2, cv2.LINE_AA)
            cv2.imshow('result', resultFrame)
            cv2.waitKey(0)


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    main()
    cap.release()
    cv2.destroyAllWindows()

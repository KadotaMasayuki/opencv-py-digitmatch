#-------------------------------------------------------------------------------
# Name:        FeatureMatching
# Purpose:
#
# Author:      kadota masayuki
#
# Created:     2019/03/02
# Copyright:   (c) kadota masayuki 2019
# Licence:     BSD Licence
#-------------------------------------------------------------------------------

import numpy as np
import cv2
import math

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

def addWeight(srcImage, gain=7.2, gamma=1.2):
    # コントラストとガンマで重み付け
    if (len(srcImage.shape) >=3):
        h, w, c = srcImage.shape
        resultImage = srcImage.copy()
    else:
        h, w = srcImage.shape
        c = 1
        resultImage = np.empty((h, w, 1), dtype=np.uint8)
        resultImage[:,:,0] = srcImage
    lookuptable = np.ones((256, 1), np.uint8)
    for i in range(256):
        lookuptable[i] = int(255.0 / (1 + math.exp(-gain * (i - 128) / 255.0)));
    for i in range(c):
        resultImage[:,:,i] = cv2.LUT(resultImage[:,:,i], lookuptable)
    if (gamma != 1):
        gammaLookuptable = np.ones((256, 1), np.uint8)
        for i in range(256):
            gammaLookuptable[i] = int(((i / 255.0) ** (1.0 / gamma)) * 255.0);
        for i in range(c):
            resultImage[:,:,i] = cv2.LUT(resultImage[:,:,i], gammaLookuptable)
    return resultImage if c != 1 else resultImage[:,:,0]

def stretchLevelRange(srcImage, minLevel=0, maxLevel=255):
    # 色レベルを min - max間 に配置する
    if (len(srcImage.shape) >=3):
        h, w, c = srcImage.shape
        resultImage = srcImage.copy()
    else:
        h, w = srcImage.shape
        c = 1
        resultImage = np.empty((h, w, 1), dtype=np.uint8)
        resultImage[:,:,0] = srcImage
    for i in range(c):
        min = np.min(resultImage[:,:,i])
        max = np.max(resultImage[:,:,i])
        resultImage[:,:,i] = (resultImage[:,:,i] - min) * 255.0 / (max - min)
    return resultImage if c != 1 else resultImage[:,:,0]

def arrangeLevelRange(srcImage, baseLevel=127, minLevel=0, maxLevel=255):
    # レベル補正する(複数画像のレベルを合わせるときに使用)
    if (len(srcImage.shape) >=3):
        h, w, c = srcImage.shape
        resultImage = srcImage.copy()
    else:
        h, w = srcImage.shape
        c = 1
        resultImage = np.empty((h, w, 1), dtype=np.uint8)
        resultImage[:,:,0] = srcImage
    lookuptable = np.zeros((256,1), np.uint8)
    for i in range(c):
        mean = int(np.mean(resultImage[:,:,i]))
        min = int(np.min(resultImage[:,:,i]))
        max = int(np.max(resultImage[:,:,i]))
        if (ENABLE_DEBUG):
            std = np.std(resultImage[:,:,i])
        for j in range(min, mean):
            lookuptable[j][0] = (float(j) - (min - minLevel)) * float(baseLevel - minLevel) / (mean - min)
        for j in range(mean, max + 1):
            lookuptable[j][0] = (float(j) - (mean - baseLevel)) * float(maxLevel - baseLevel) / (max - mean)
        resultImage[:,:,i] = cv2.LUT(resultImage[:,:,i], lookuptable)
        if (ENABLE_DEBUG):
            mean2 = int(np.mean(resultImage[:,:,i]))
            min2 = int(np.min(resultImage[:,:,i]))
            max2 = int(np.max(resultImage[:,:,i]))
            std2 = np.std(resultImage[:,:,i])
            print("arrangeLevel:c={0}, max={1}/{2}, min={3}/{4}, mean={5:.3f}/{6:.3f}, std={7:.3f}/{8:.3f}".format(i, max, max2, min, min2, mean, mean2, std, std2))
            cv2.waitKey(0)
    return resultImage if c != 1 else resultImage[:,:,0]

def arrangeLevelStddev(srcImage, baseLevel=127, stdDevWeight=127):
    # レベル補正する(複数画像のレベルを合わせるときに使用)
    if (len(srcImage.shape) >=3):
        h, w, c = srcImage.shape
        resultImage = srcImage.copy()
    else:
        h, w = srcImage.shape
        c = 1
        resultImage = np.empty((h, w, 1), dtype=np.uint8)
        resultImage[:,:,0] = srcImage
    for i in range(c):
        mean = np.mean(resultImage[:,:,i])
        std = np.std(resultImage[:,:,i])
        if (ENABLE_DEBUG):
            min = np.min(resultImage[:,:,i])
            max = np.max(resultImage[:,:,i])
        resultImage[:,:,i] = ((resultImage[:,:,i] - mean) / std) * stdDevWeight + baseLevel
        if (ENABLE_DEBUG):
            mean2 = int(np.mean(resultImage[:,:,i]))
            min2 = int(np.min(resultImage[:,:,i]))
            max2 = int(np.max(resultImage[:,:,i]))
            std2 = np.std(resultImage[:,:,i])
            print("arrangeLevelStddev:c={0}, max={1}/{2}, min={3}/{4}, mean={5:.3f}/{6:.3f}, std={7:.3f}/{8:.3f}".format(i, max, max2, min, min2, mean, mean2, std, std2))
            cv2.waitKey(0)
    return resultImage if c != 1 else resultImage[:,:,0]

def expand(srcImage, expandRatioX=1.5, expandRatioY=1.5):
    # 画像を伸縮 (各種画像処理前に実施しないとボケる)
    return cv2.resize(srcImage, None, fx=expandRatioX, fy=expandRatioY)

def gray(srcImage):
    # グレー画像
    return cv2.cvtColor(srcImage, cv2.COLOR_BGR2GRAY)

def blur(srcImage, blurX=1, blurY=1):
    # ボケ(1より大きい奇数)
    if (blurX > 0):
        blurX = int(blurX / 2) * 2 + 1
    if (blurY > 0):
        blurY = int(blurY / 2) * 2 + 1
    if ((blurX > 0) and (blurY > 0)):
        return cv2.GaussianBlur(srcImage, (blurX, blurY), 0)
    else:
        return srcImage

def thresholdAdaptive(grayImage, thresholdBlockSize=3, thresholdConstant=2):
    # 二値化(3より大きい奇数)
    if (thresholdBlockSize >= 3):
        return cv2.adaptiveThreshold(grayImage, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY,
                                     thresholdBlockSize, thresholdConstant)
    else:
        return grayImage

def thresholdBinary(grayImage, threshold=127, maxValue=255):
    # 二値化 : maxValue if pixel > threshold else 0
    _, resultImage = cv2.threshold(grayImage, threshold, maxValue, cv2.THRESH_BINARY)
    return resultImage

def thresholdBinaryInvert(grayImage, threshold=127, maxValue=255):
    # 二値化 : 0 if pixel > threshold else maxValue
    _, resultImage = cv2.threshold(grayImage, threshold, maxValue, cv2.THRESH_BINARY_INV)
    return resultImage

def thresholdSetMaxValueOverThreshold(grayImage, threshold=127, maxValue=255):
    # 閾値以上を閾値に統一する : threshold if pixel > threshold else pixel
    _, resultImage = cv2.threshold(grayImage, threshold, maxValue, cv2.THRESH_TRUNC)
    return resultImage

def thresholdSetZeroUnderThreshold(grayImage, threshold=127, maxValue=255):
    # 閾値以下をゼロに統一する : pixel if pixel > threshold else 0
    _, resultImage = cv2.threshold(grayImage, threshold, maxValue, cv2.THRESH_TOZERO)
    return resultImage

def thresholdSetZeroOverThreshold(grayImage, threshold=127, maxValue=255):
    # 閾値以上をゼロに統一する : 0 if pixel > threshold else pixel
    _, resultImage = cv2.threshold(grayImage, threshold, maxValue, cv2.THRESH_TOZERO_INV)
    return resultImage

def thresholdOtsu(grayImage):
    # 大津の二値化 blur() してから適用すると良好
    # http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html
    _, resultImage = cv2.threshold(grayImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return resultImage

def erode(srcImage, erodeSizeX=1, erodeSizeY=1, erodeTime=1):
    # 黒を拡大
    if (erodeSizeX > 0 and erodeSizeY > 0 and erodeTime > 0):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erodeSizeX, erodeSizeY))
        return cv2.erode(srcImage, kernel, iterations=erodeTime)
    else:
        return srcImage

def canny(srcImage, cannyMaxVal=200, cannyMinVal=100, cannySobelSize=3, cannyGradient=False):
    # エッジ検出(sobelSizeは3より大きい奇数)
    if (cannySobelSize > 0):
        cannySobelSize = int(cannySobelSize / 2) * 2 + 1
    if (cannyMaxVal > 0 and cannyMinVal > 0 and cannySobelSize >= 3):
        return cv2.Canny(srcImage, cannyMaxVal, cannyMinVal,
                         None, cannySobelSize, cannyGradient)
    else:
        return srcImage

def invert(srcImage):
    # 色反転
    return cv2.bitwise_not(srcImage)

def addMergin(srcImage, merginSize=40, merginColor=0):
    # グレー画像に余白を付ける(Canny後の背景色は黒)
    if (len(srcImage.shape) < 3):
        resultImage = np.ones((h + merginSize * 2, w + merginSize * 2), np.uint8)
        resultImage[:,:] = merginColor
        resultImage[merginSize:merginSize + h, merginSize:merginSize + w] = srcImage
        return resultImage
    else:
        return srcImage

def addKeypoints(srcImage):
    # 特徴点を描画した画像を返す
    detector = cv2.AKAZE_create()
    kp1, des1 = detector.detectAndCompute(srcImage, None)
    return cv2.drawKeypoints(srcImage, kp1, None,
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

def align(templateImage, sceneImage, matchRequired=10, goodMatchRate=0.15):
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html#feature-homography

    detector = cv2.AKAZE_create()
    kp1, des1 = detector.detectAndCompute(templateImage, None)
    kp2, des2 = detector.detectAndCompute(sceneImage, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # 上位 goodMatchRate ぶんだけを good[] に格納する
    matches = sorted(matches, key=lambda x: x.distance)
    good = matches[:int(len(matches) * goodMatchRate)]

    # cv2.drawMatches***は適合している点を結ぶ画像を生成する
    relationImage = cv2.drawMatches(templateImage, kp1, sceneImage, kp2, good, None, flags=2)

    # 取得した特徴量を使って入力画像を変形する
    if (len(good) > matchRequired):
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Find homography
        homography, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        # Use homography
        height, width = templateImage.shape[:2]
        resultImage = cv2.warpPerspective(sceneImage, homography, (width, height))
        return 1, resultImage, homography, relationImage
    else:
        return 0, templateImage, np.zeros((3, 3)), relationImage

if __name__ == '__main__':
    pass

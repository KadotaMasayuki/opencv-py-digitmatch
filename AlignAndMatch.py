#-------------------------------------------------------------------------------
# Name:        FeatureMatching
# Purpose:
#
# Author:      kadota masayuki
#
# Created:     2019/03/03
# Copyright:   (c) kadota masayuki 2019
# Licence:     BSD Licence
#-------------------------------------------------------------------------------

import numpy as np
import cv2
import os
from pathlib import Path
import process as p

def match(scenes, showProcess=False):
    # 'm' で、保存したテンプレート郡と一括マッチング (表示後、キー入力待ちで止める)
    guesses = []
    scores = []
    for i in range(len(scenes)):
        # マッチング用に拡大
        targetScene = p.expand(scenes[i], 2, 2)
        maxScore = 0
        maxTemplate = ''
        for f in Path('./').glob('template_*.png'):
            # マッチング用に拡大
            targetTemplate = cv2.imread(f.as_posix(), 0)
            targetTemplate = p.expand(targetTemplate, 2, 2)
            # マッチング
            #guessImage, score = p.featureMatch1to1(targetTemplate, targetScene, 0.2)
            guessImage, score = p.templateMatch1to1(targetTemplate, targetScene) # グレー画像のみ
            if (score > maxScore):
                maxScore = score
                maxTemplate = str(f)
            scores.append("scene={0},template={1},score={2}".format(i, f, score))
            if (showProcess):
                viewImage = cv2.cvtColor(guessImage, cv2.COLOR_GRAY2BGR)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(viewImage, "scene {0}".format(i), (10, 20), font, 0.5, (0, 64, 255), 2, cv2.LINE_AA)
                cv2.putText(viewImage, "name {0}".format(f), (10, 40), font, 0.5, (0,64,255), 2, cv2.LINE_AA)
                cv2.putText(viewImage, "score {0:.4f}".format(score), (10, 60), font, 0.5, (0,64,255), 2, cv2.LINE_AA)
                cv2.imshow('scene_{0},result_{1}'.format(i, f), viewImage)
        if (len(maxTemplate) > len("template_")):
            guesses.append(maxTemplate[len("template_")])
        else:
            guesses.append("")
    return guesses, scores

def main():
    expandRatio = 3.3

    # 機器全体のモデル画像 (比較識別しやすいように画像処理済みのもの)
    templateImage1 = cv2.imread('./whole.png', 0)
    templateImage1 = None

    # 桁ウィンドウのモデル画像 (比較識別しやすいように画像処理済みのもの)
    templateImage2 = cv2.imread('./window.png', 0)
    templateImage2 = None

    # マッチ実施するか
    autoMatch = 0

    # 現在の操作桁
    currentDigit = 0

    # 一時的なモデル画像
    tmpTemplate = np.zeros((50, 50), np.uint8)

    while (True):
        # 撮影
        ret, srcImage = cap.read()
        #cv2.imshow('src', srcImage)

        # 位置決め
        if ((not templateImage1 is None) or (not templateImage2 is None)):
            # 拡大
            sceneImage = p.expand(srcImage, expandRatio, expandRatio)

            # 位置決め1
            if (not templateImage1 is None):
                # モデル画像と比較するための画像処理
                aImage = sceneImage.copy()
                aImage = p.gray(aImage)
                aImage = p.thresholdBinary(aImage)
                # 位置決め
                ret, aImage, homography1, relationImage1 = p.align(templateImage1, aImage)
                # 元画像から桁ウィンドウを取り出し
                height, width = templateImage1.shape[:2]
                windowImage = cv2.warpPerspective(sceneImage, homography1, (width, height))
                cv2.imshow('whole scene', wholeImage)

            # 位置決め2
            if (not templateImage2 is None):
                # モデル画像と比較するための画像処理
                aImage = sceneImage.copy()
                aImage = p.gray(aImage)
                aImage = p.thresholdBinary(aImage)
                # 位置決め
                ret, aImage, homography2, relationImage2 = p.align(templateImage2, aImage)
                # 元画像から桁ウィンドウを取り出し
                height, width = templateImage2.shape[:2]
                windowImage = cv2.warpPerspective(windowImage, homography2, (width, height))
                cv2.imshow('digit window', windowImage)

        # 桁ウィンドウ描画
        windowTopLeft = (260, 200)
        windowSize = (119, 50)
        digitTopLeft = ((0, 0), (28, 0), (63, 0), (91, 0))
        digitSize = (28, 50)

        # 全体像 + 枠
        sceneImage = srcImage.copy()
        cv2.rectangle(sceneImage,
                      (windowTopLeft),
                      (windowTopLeft[0] + windowSize[0], windowTopLeft[1] + windowSize[1]),
                      (255, 255, 0),
                      3)
        sceneImage = p.expand(sceneImage, 0.7, 0.7)
        cv2.imshow('preview', sceneImage)

        # 桁ウィンドウ
        windowImage = p.expand(srcImage[windowTopLeft[1]:windowTopLeft[1] + windowSize[1],
                                        windowTopLeft[0]:windowTopLeft[0] + windowSize[0]],
                               expandRatio, expandRatio)

        # 桁ウィンドウに枠描画
        windowDigitImage = windowImage.copy()
        for i in range(4):
            # 枠描画
            cv2.rectangle(windowDigitImage,
                          (int(digitTopLeft[i][0] * expandRatio), int(digitTopLeft[i][1] * expandRatio)),
                          (int((digitTopLeft[i][0] + digitSize[0]) * expandRatio), int((digitTopLeft[i][1] + digitSize[1]) * expandRatio)),
                          (0, 128, 255) if (i == currentDigit) else (255, 0, 128),
                          4 if (i == currentDigit) else 2)
        cv2.imshow('window', windowDigitImage)

        # 桁取り出し
        digits = []
        for i in range(4):
            # 桁取り出し
            digitImage = windowImage[int(digitTopLeft[i][1] * expandRatio):int((digitTopLeft[i][1] + digitSize[1]) * expandRatio),
                                     int(digitTopLeft[i][0] * expandRatio):int((digitTopLeft[i][0] + digitSize[0]) * expandRatio)]
            # 桁マッチングのための画像処理
            digitImage = p.gray(digitImage)
            digitImage = p.stretchLevelRange(digitImage)
            digitImage = p.thresholdOtsu(digitImage)
            cv2.imshow("digit-{0}".format(i), digitImage)
            digits.append(digitImage)

        # 仮モデル
        cv2.imshow('temp-template', tmpTemplate)

        # 仮モデルとマッチ
        instantGuessImage, score = p.templateMatch1to1(tmpTemplate, digits[currentDigit])
        cv2.imshow('instant guess', instantGuessImage)

        # 一括マッチ
        if (autoMatch > 0):
            guesses, scores = match(digits, showProcess=False)
            resultImage = windowImage.copy()
            outImage = np.ones((200, 500, 3))
            font = cv2.FONT_HERSHEY_SIMPLEX
            for i in range(len(digits)):
                cv2.putText(resultImage,
                            "'{0}'".format(guesses[i]),
                            (int((10 + digitTopLeft[i][0]) * expandRatio), int((10 + digitTopLeft[i][1]) * expandRatio)),
                            font, 1.3, (0,255,0), 2, cv2.LINE_AA)
                cv2.putText(outImage,
                            "{0}".format(guesses[i]),
                            (50 + i * 100, 120),
                            font, 2.5, (255, 0, 128), 3, cv2.LINE_AA)
            cv2.imshow('result', resultImage)
            cv2.imshow('out', outImage)
            for i in range(len(scores)):
                print(scores[i])
            autoMatch &= 0xfe

        k = cv2.waitKey(1) & 0xff
        if (k == 27):
            # ESCキーで終了
            break
        elif (ord('0') <= k <= ord('9')):
            # '0' - '9' で操作桁を移動
            if (k - ord('0') < len(digits)):
                currentDigit = k - ord('0')
        elif (k == ord('t')):
            # 't' でテンプレートを仮登録 ('s'で保存)
            tmpTemplate = digits[currentDigit]
        elif (k == ord('s')):
            # 's' でテンプレート保存
            k = cv2.waitKey(0) & 0xff
            if (k != 27):
                # ESC 以外で、その文字と関連付けて保存
                t = chr(k)
                n = 0
                while (True):
                    # 同一ファイル名なら追番を付与
                    templateFilePath = "./template_" + t + "_" + str(n) + ".png"
                    if (not os.path.exists(templateFilePath)):
                        cv2.imwrite(templateFilePath, tmpTemplate)
                        break
                    n = n + 1
        elif (k == ord('m')):
            # 'm' で1回マッチ
            autoMatch = 1
        elif (k == ord('a')):
            # 'a' で連続マッチ
            autoMatch = 2

if __name__ == '__main__':
    cap = cv2.VideoCapture(1)
    main()
    cap.release()
    cv2.destroyAllWindows()

import cv2
import numpy as np

class matcher:
    def __init__(self, siftOrSurf): #constructorul pentru cei doi matcheri
        self.siftOrSurf = siftOrSurf
        if siftOrSurf == 'sift':
            self.featureMatcher = cv2.xfeatures2d.SIFT_create()  # feature Matcher bazat pe algoritmul SIFT
        elif siftOrSurf == 'surf':
            self.featureMatcher = cv2.xfeatures2d.SURF_create()  # feature Matcher bazat pe algoritmul SURF
            index_params = dict(algorithm=0, trees=5)  # parametrii trebuie pasati ca doua dictionare
            search_params = dict(checks=50)
            self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def matchSift(self, im1, im2):
        kp1, des1 = self.getFeatures(im1)
        kp2, des2 = self.getFeatures(im2)  # pentru algoritmul de sift, se foloseste un brute-force matcher.
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)  # aceste matches au doua valori pentru fiecare instanta, queryID si trainID
        good = []
        for m in matches:  # pentru o distanta mai mica decat un threshhold, se vor alege match-urile bune
            if m[0].distance < 0.7 * m[1].distance:
                good.append(m)
        matches = np.asarray(good)
        if len(good) > 4:
            pointsCurrent = kp1
            pointsPrev = kp2
            matchedCurrent = np.float32(  # matchedCurrent reprezinta features care au avut un match in imaginea 2, din imaginea 1
                [pointsCurrent[m.queryIdx].pt for m in matches[:, 0]]
            )
            matchedPrev = np.float32(
                [pointsPrev[m.trainIdx].pt for m in matches[:, 0]]
            )
            H, _ = cv2.findHomography(matchedCurrent, matchedPrev, cv2.RANSAC, 5)  # pe baza acestor features,
            # folosim o functie openCV care gaseste o matrice de transformare 3x3, care va transforma una din imagini,
            # in asa fel incat sa se mapeze pentru punctele celei de-a doua => H. Aceasta matrice va fi folosite pentru a
            # aplica un efect de warp pe una dintre imagini, pentru a o suprapune cu cea de-a doua
            return H
        return None

    def matchSurf(self, im1, im2):  # proces identic cu matchSift, dar se foloseste un alt tip de matcher.
        kp1, des1 = self.getFeatures(im1)
        kp2, des2 = self.getFeatures(im2)
        matches = self.flann.knnMatch(des1, des2, k=2)
        good = []
        for m in matches:
            if m[0].distance < 0.7 * m[1].distance:
                good.append(m)
        matches = np.asarray(good)
        if len(good) > 4:
            pointsCurrent = kp1
            pointsPrev = kp2
            matchedCurrent = np.float32(
                [pointsCurrent[m.queryIdx].pt for m in matches[:, 0]]
			)
            matchedPrev = np.float32(
                [pointsPrev[m.trainIdx].pt for m in matches[:, 0]]
            )
            H, _ = cv2.findHomography(matchedCurrent, matchedPrev, cv2.RANSAC, 5)
            return H
        return None

    def match(self, im1, im2):  #
        if self.siftOrSurf == 'sift':
            return self.matchSift(im1, im2)
        else:
            return self.matchSurf(im1, im2)

    def getFeatures(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        kp, des = self.featureMatcher.detectAndCompute(gray, None)  # calculeaza o serie de features si descriptori pentru o
        # imagine. Pe baza acestora vom afla punctele care sunt comune in cele doua imagini.
        return kp, des  # keypoints si descriptori

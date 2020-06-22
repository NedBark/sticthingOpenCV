import glob
import cv2
import sys
import numpy as np
from matcher import matcher
debugShowIm = False

class Stitching:
    #Constructor stitching
    def __init__(self, args):
        #Salvam argumentul de intrare in program (path-ul catre directorul cu poze si setam algoritm surf/sift
        self.path = args
        self.siftOrSurf = 'surf'
        
        #Citim imaginile din director, le facem un resize si le memoram in images
        self.images = [cv2.resize(cv2.imread(file), (700, 610)) for file in glob.glob(self.path + "*.jpg")]
       
        #Salvam numarul de imagini
        self.count = len(self.images)
        print("Imagini gasite: ", self.count)
        
        #Initializare de array-uri
        self.leftList, self.rightList = [], []
        
        #Initializam un obiect de tip matcher caruia ii pasam tipul de algoritm definit mai sus
        self.matcher_obj = matcher(siftOrSurf=self.siftOrSurf)



    def addLeftToRight(self, indexMid):
        #Memoram imaginea din centru
        left = self.images[indexMid]
        
        #Parcurgem catre dreapta urmatoarele imagini
        for i in range(indexMid, self.count):
            #Salvam urmatoarea imaginie
            right = self.images[i]
            
            #Determinam match-ul intre imaginea dreapta si stanga
            H = self.matcher_obj.match(right, left)

            #Facem un warp pe imaginea ce urmeaza sa fie adaugata in dreapta
            right = cv2.warpPerspective(right, H, (right.shape[1] +  left.shape[1], right.shape[0]))
            
            #Debug imshow
            if debugShowIm:
                imgName='warped-LTR-'+str(i)+'.jpg'
                cv2.imshow(imgName, right)
                cv2.waitKey()
                
            #Cautam coloana dupa care urmeaza sa cropam imaginea din stanga, conditia de cropare:
            #gasirea celei mai indepartat pixel (de marginea din dreapta) ce are valoarea [0,0,0].
            #Cautarea se face pe prima linie si pe ultima linie
            value = left.shape[1]
            for i in reversed(range(left.shape[1])):
                if left[0][i][0] == 0 and left[0][i][1] == 0 and left[0][i][2] == 0:
                    value = i
                if left[-1][i][0] == 0 and left[-1][i][1] == 0 and left[-1][i][2] == 0:
                    value = i
                    
            #Cropam imaginea din stanga
            left = left[:, 0:value, :]
            #Adaugam peste imaginea warpped (rightImage) imaginea din stanga - prima iteratie: imaginea din stanga = imaginea centrala)
            # a doua iteratie: imaginea din stanga= imaginea centrala+imaginea warpped la prima iteratie
            # ...
            #Pentru aceast pas am realizat croparea de la linia 63, deoarece daca nu realizam croparea
            #imaginea de suprapus continea parti negre care suprascriau informatia din imaginea warpped 
            #parti negre obtinute in urma warp-ului
            right[:, 0:left.shape[1]] = left
            
            #Debug imshow
            if debugShowIm:
                imgName='right-LTR-'+str(i)+'.jpg'
                cv2.imshow(imgName, right)
                cv2.waitKey()
            left = right
            
        #Debug imshow
        if debugShowIm:
            imgName='final-addLeftToRight.jpg'
            cv2.imshow(imgName, left)
            cv2.waitKey()
        #Memoram rezultatul metodei
        self.rightImage = left




    def addRightToLeft(self, indexMid):
        #Memoram imaginea din centru
        right = self.images[indexMid]
        
        #Parcurgem catre stanga urmatoarele imagini
        for i in reversed(range(indexMid)):
            #Salvam urmatoarea imaginie
            left = self.images[i]
            
            #Determinam match-ul intre imaginea dreapta si stanga
            H = self.matcher_obj.match(right, left)
            # Ii facem inversa matricei de transformare deoarece in mod normal orientarea warpului
            #era '/' iar noi pentru a adauga la dreapta avem nevoie de orientare '\'
            H = np.linalg.inv(H)
            #Am adaugat si un offset pe axa X, deoarece fara el o parte din imaginea
            #warpped era in afara ferestrei de afisare
            H[0][2] = H[0][2]+left.shape[1]
                
            #Facem un warp pe imaginea ce urmeaza sa fie adaugata in stanga
            left = cv2.warpPerspective(left, H, (left.shape[1], left.shape[0]))

            #Debug imshow
            if debugShowIm:
                imgName='warped-RTL-'+str(i)+'.jpg'
                cv2.imshow(imgName, left)
                cv2.waitKey()

            #Unim cele 2 imagini
            left = np.concatenate((left[:,0:right.shape[1],:], right), axis=1)

            # Similar cu croparea de mai sus, doar ca de data aceasta vom sterge partea neagra
            #care a ramas in stanga imaginii
            valueUp = -2
            valueDown = -2
            for i in range(left.shape[1]):
                if left[0][i][0] != 0 and left[0][i][1] != 0 and left[0][i][2] != 0:
                    valueUp = i - 1
                    if valueDown != -2:
                        break
                if left[-1][i][0] != 0 and left[-1][i][1] != 0 and left[-1][i][2] != 0:
                    valueDown = i - 1
                    if valueUp != -2:
                        break
                        
            #Realizam croparea
            left = left[:, max(valueUp, valueDown):left.shape[1], :]
            
            #Debug imshow
            if debugShowIm:
                imgName='warped-LTR-'+str(i)+'.jpg'
                cv2.imshow(imgName, right)
                cv2.waitKey()

            right = left
        #Memoram rezultatul metodei
        self.leftImage = right

if __name__ == '__main__':

    args = sys.argv[1]
    #Instantie obiectul
    stitch_obj = Stitching(args)
    
    #Apelarea metodelor
    midIndex = stitch_obj.count//2
    stitch_obj.addLeftToRight(midIndex)
    stitch_obj.addRightToLeft(midIndex)
    
    imRight = stitch_obj.rightImage
    imLeft = stitch_obj.leftImage
    
    #Generam rezultatul final
    imageFinal = np.concatenate((imLeft, imRight[:, stitch_obj.images[midIndex].shape[1]:]), axis=1)    
    cv2.imwrite('Image.jpg', imageFinal)

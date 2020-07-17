import cv2
import numpy as np
import os

class Processing:
    

    def SaveImage(self, imagepath, savepath):
        self.imagepath = imagepath
        self.savepath = savepath
        
        img = cv2.imread(imagepath)

        B = img[:, :, 0].astype(float) # float conversion, maybe we can do better. But this results in correct answer
        G = img[:, :, 1].astype(float) #
        R = img[:, :, 2].astype(float)


        B_ = np.copy(B) 
        G_ = np.copy(G)
        R_ = np.copy(R)


        K = np.zeros_like(B) 
        C = np.zeros_like(B) 
        M = np.zeros_like(B) 
        Y = np.zeros_like(B) 


        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                B_[i, j] = B[i, j]/255
                G_[i, j] = G[i, j]/255
                R_[i, j] = R[i, j]/255

                K[i, j] = 1 - max(B_[i, j], G_[i, j], R_[i, j])
                if (B_[i, j] == 0) and (G_[i, j] == 0) and (R_[i, j] == 0):
                # black
                    C[i, j] = 0
                    M[i, j] = 0  
                    Y[i, j] = 0
                else:

                    C[i, j] = (1 - R_[i, j] - K[i, j])/float((1 - K[i, j]))
                    M[i, j] = (1 - G_[i, j] - K[i, j])/float((1 - K[i, j]))
                    Y[i, j] = (1 - B_[i, j] - K[i, j])/float((1 - K[i, j]))

        C = (1-C)
        M = (1-M)
        Y = (1-Y)
        K = (1-K)


        os.chdir(savepath)
        cv2.imwrite('Cyan.png', C*255)
        cv2.imwrite('Magenta.jpg', M*255)
        cv2.imwrite('Yellow.jpg', Y*255)
        cv2.imwrite('Black.jpg', K*255)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def ShowImage(self, ImagePath):
        img = cv2.imread(ImagePath)

        B = img[:, :, 0].astype(float) # float conversion, maybe we can do better. But this results in correct answer
        G = img[:, :, 1].astype(float) #
        R = img[:, :, 2].astype(float)


        B_ = np.copy(B) 
        G_ = np.copy(G)
        R_ = np.copy(R)


        K = np.zeros_like(B) 
        C = np.zeros_like(B) 
        M = np.zeros_like(B) 
        Y = np.zeros_like(B) 


        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                B_[i, j] = B[i, j]/255
                G_[i, j] = G[i, j]/255
                R_[i, j] = R[i, j]/255

                K[i, j] = 1 - max(B_[i, j], G_[i, j], R_[i, j])
                if (B_[i, j] == 0) and (G_[i, j] == 0) and (R_[i, j] == 0):
                # black
                    C[i, j] = 0
                    M[i, j] = 0  
                    Y[i, j] = 0
                else:

                    C[i, j] = (1 - R_[i, j] - K[i, j])/float((1 - K[i, j]))
                    M[i, j] = (1 - G_[i, j] - K[i, j])/float((1 - K[i, j]))
                    Y[i, j] = (1 - B_[i, j] - K[i, j])/float((1 - K[i, j]))

        C = (1-C)
        M = (1-M)
        Y = (1-Y)
        K = (1-K)

        cv2.imshow('C',C)
        cv2.imshow('M',M)
        cv2.imshow('Y',Y)
        cv2.imshow('K',K)   
        cv2.waitKey(0)
        cv2.destroyAllWindows()

import numpy as np
import cv2
import os
import pickle
import sys
import math

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton

from scipy.spatial.distance import euclidean


class GenNeirest:
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
       Neirest neighbor method: it select the image in videoSke that has the skeleton closest to the skeleton
    """
    def __init__(self, videoSkeTgt):
        self.videoSkeletonTarget = videoSkeTgt

    # def generate(self, ske):           
    #     """ generator of image from skeleton """
    #     # TP-TODO
    #     empty = np.ones((64,64, 3), dtype=np.uint8)
    #     return empty


    def generate(self, ske):
        """
        Generator of image from skeleton
        """
        min_distance = float('inf')
        nearest_image = None

        # Parcourir tous les squelettes dans le dataset
        for idx in range(self.videoSkeletonTarget.skeCount()):
            current_skeleton = self.videoSkeletonTarget.ske[idx]
            
            # Calculer la distance entre le squelette d'entrée et le squelette courant
            distance = ske.distance(current_skeleton)
            
            # Mettre à jour l'image la plus proche si une distance plus petite est trouvée
            if distance < min_distance:
                min_distance = distance
                nearest_image = self.videoSkeletonTarget.readImage(idx)
                #nearest_image=cv2.cvtColor(nearest_image, cv2.COLOR_BGR2RGB)
        # Si aucune image n'est trouvée, retourner une image vide
        if nearest_image is None:
            print("No nearest image found")
            return np.ones((64, 64, 3), dtype=np.uint8)

        return nearest_image

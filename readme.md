**Posture-guided image synthesis of a person**

#Usage

To train models for image generation, follow these steps:

    Generate images from skeleton input: Run GenVanillaNNfromSke.py.
    Generate images from image input: Run GenVanillaNNfromImage.py.
    Generate images using GAN (input size = 64): Run GenGAN2.py.
    Generate images using GAN (input size = 128): Run GenGAN128.py.
    
    
    
For inference, you can test the different models by running DanceDemo.py and modifying the 'GEN_TYPE' variable accordingly.
**GEN_TYPE = 2** : using model from skeleton
**GEN_TYPE = 3** : using model from image
**GEN_TYPE = 4** : using model from GAN


1. Using a different approaches to train a model with differents architectures. and to get a good performances :

 2. Utilization of the Adam optimizer: to ensure stable gradient descent.
 3. Learning rate scheduler: Reduces the learning rate by a factor of 0.1 every 30 epochs, allowing the model to stabilize toward the end of training.

4. Early Stopping: If the  loss does not improve for 20 consecutive epochs, the training stops automatically to prevent overfitting.

6. Best Model Saving:model is saved whenever a better loss is observed, ensuring that the best parameters are retained.


Le meilleur résultat est obtenu avec GAN (size input = 128) en utilisant les fonctionnalités suivantes:

1. Fonctions de perte : La perte GAN (MSE) est utilisée pour l'adversarial training, complétée par une perte L1 (pixel loss) pour améliorer la qualité des images générées.

2. Optimisation : Utilisation d'Adam avec des taux d'apprentissage différents pour le générateur (0.0001) et le discriminateur (0.0004), ainsi qu'un scheduler pour réduire le taux d'apprentissage tous les 30 pas.

3. Entraînement par lots alternés : Le générateur est mis à jour à chaque itération, mais le discriminateur seulement tous les 4 lots.

4. Early Stopping : Si la perte du générateur n'améliore pas pendant un certain nombre d'époques, l'entraînement est arrêté pour éviter le surentraînement.

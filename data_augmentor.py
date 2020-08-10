# ref; https://github.com/aleju/imgaug

import imageio
import imgaug as ia
import imgaug.augmenters as iaa

import numpy as np
import cv2
import os


class DataAugmentator(object):
    def __init__(self, num_labels):
        
        self.num_labels = num_labels
        self.id = 0
        """
        self.seq = iaa.Sequential([
            iaa.Add((-1, 1))
            ])
        """
        self.seq = iaa.Sequential([
            iaa.OneOf([
                # change brightness of images (by -10 to 10 of original value
                iaa.Add((-100, 100)),
                # change brightness of images (by -10 to 10 of original value in each pixels)
                iaa.Add((-100, 100), per_channel=0.5),
                # change hue and saturation
                iaa.AddToHueAndSaturation((-20, 20)),
                # Convert each image to grayscale and then overlay the
                # result with the original with random alpha. I.e. remove
                # colors with varying strengths.
                iaa.Grayscale(alpha=(0.0, 1.0)),
                # Strengthen or weaken the contrast in each image.
                iaa.ContrastNormalization((0.5, 1.5))
            ]),
            iaa.SomeOf((0, 3),
            [
                # add gaussian noise to images
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255)),
                # Make some images brighter and some darker.
                # In 20% of all cases, we sample the multiplier once per channel,
                # which can end up changing the color of the images.
                iaa.Multiply((0.8, 1.2)),
                iaa.CoarseDropout((0.10, 0.20), size_percent=(0.01, 0.02)),
                iaa.GaussianBlur(sigma=(0, 3.0))
            ]),
            # horizontal flips
            # in processing data, already flip randomly
            # iaa.Fliplr(0.3)
        ])
        
    def show_img(self, img, land, state):
        os.makedirs("sample_aug", exist_ok=True)

        # print("Keypoints as array:", land)
        print("Shape:", np.array(land).shape)
        land = np.array(land[0])

        for (x, y) in land.astype(np.int32):
            cv2.circle(img, (x, y), 1, (0, 0, 255))
        cv2.imwrite("./sample_aug/" + str(self.id) + state + ".jpg", img)
        self.id += 1


    def augment_image(self, image, landmark):
        h, w, _ = image.shape
        landmark = landmark.reshape(-1, 2) * [h, w]
        landmark = [[(land[0], land[1]) for land in landmark]]
        if self.id < 10:
            self.show_img(image, landmark, "original")

        image_aug, landmark_aug = self.seq(image=image, keypoints=landmark)
        if self.id < 10:
            self.show_img(image_aug, landmark_aug, "aug")

        landmark_aug = np.array(landmark_aug) / [h, w]
        landmark_aug = landmark_aug.reshape(-1, )

        return [image_aug, landmark_aug]

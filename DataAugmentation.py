import cv2
import numpy as np
import tensorflow as tf

class DataAugmentation:
    def __init__(self, config):
        self.config = config

        return

    def __read_image_from_disk_cv2(self, path):
        def read_cv2(path_):
            image = cv2.imread(str(np.core.defchararray.decode(path_.numpy())))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # cv2.imshow('mask', image)
            # cv2.destroyAllWindows()
            # cv2.waitKey()

            return image

        # img = tf.io.read_file(img)
        # img = tf.image.decode_jpeg(img)

        return tf.py_function(read_cv2, [path], np.uint8)

    def preprocess(self, image_path):
        image = self.__read_image_from_disk_cv2(image_path)
        normalized_image = (image - self.config.image_average) / self.config.image_std

        return normalized_image
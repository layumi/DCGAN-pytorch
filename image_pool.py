import random
import numpy as np
import torch
from torch.autograd import Variable

class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0   # number of stored images
            self.images = []

    def query(self, images):  # get image from pool
        if self.pool_size == 0: # The no replay  version
            return Variable(images)
        return_images = []
        for image in images:
            image = torch.unsqueeze(image, 0)  # image to vector
            if self.num_imgs < self.pool_size:  # pool is not full, add image
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0,1)   
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size-1)  #random select
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image  # replace the pool image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))
        return return_images

from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
from typing import Tuple
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt



def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
    """
    Compute the output size given input size and target long side length.
    """
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)

image = cv2.imread('images/dog.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
newh, neww = get_preprocess_shape(image.shape[0], image.shape[1], 1024)

import time
st = time.time()
orig = np.array(resize(to_pil_image(image), (newh, neww)))
custom = cv2.resize(image, (neww, newh), interpolation=cv2.INTER_LINEAR)
et = time.time()
print(et-st)
diff = orig-custom
print(diff.mean())
print(diff.max())
plt.imshow(orig)



from PIL import Image
import numpy as np
from example_utils import run_example
import matplotlib.pyplot as plt


def _load_image_as_np_array(img_path):
    img = Image.open(img_path)
    img = np.array(img)
    img = np.expand_dims(img, 0).astype(np.uint8)

    return img


input_image = _load_image_as_np_array("sample_images/input.jpg")
masks = _load_image_as_np_array("sample_images/mask.png")

area_stats = run_example(input_image, masks)
print(area_stats)

# visualize modified input image
plt.imshow(input_image[0])
plt.show()

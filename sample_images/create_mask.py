import numpy as np
from keras_segmentation.pretrained import pspnet_50_ADE_20K
from PIL import Image

model = pspnet_50_ADE_20K()
out = model.predict_segmentation(inp="input.jpg", out_fname="prediction.png")

# map original labels to a continuous range
vals = np.unique(out)
labels = np.arange(len(vals))

for val, label in zip(vals, labels):
    out[out == val] = label

out = out.astype(np.uint8)

pil_image = Image.fromarray(out)
pil_image = pil_image.resize((519, 400), resample=Image.NEAREST)
pil_image.save("mask.png", "PNG")

#!/usr/local/bin/python3
from PIL import Image

# Open Paddington
img = Video.open("ants3.mp4")

# Resize smoothly down to 16x16 pixels
imgSmall = img.resize((16,16),resample=Image.BILINEAR)

# Scale back up using NEAREST to original size
result = imgSmall.resize(img.size,Image.NEAREST)

# Save
result.save('result.mp4')
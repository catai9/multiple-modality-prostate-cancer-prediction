import numpy as np
import matplotlib.pyplot as plt

t2w_img = np.load(
    "data/raw/ProstateX-0008/images/T2w.npy"
)
mask_img = np.load(
    "data/raw/ProstateX-0008/lesion_masks/T2w.npy"
)

print(t2w_img.shape)
print(mask_img.shape)

for slc in range(t2w_img.shape[2]):
    print(f"Slice {slc}")

    plt.imshow(t2w_img[:, :, slc], cmap='gray')
    plt.axis("off")
    plt.savefig(f"temp/t2w-{slc}.png")

    plt.imshow(mask_img[:, :, slc], cmap='gray')
    plt.axis("off")
    plt.savefig(f"temp/t2w_mask-{slc}.png")

    plt.close()

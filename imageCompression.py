from PIL import Image
import numpy as np
from pprint import pprint
from scipy import fftpack
import os
import io


def get_2D_dct(img):
    # Get 2D Cosine Transform of Image
    return fftpack.dct(fftpack.dct(img.T, norm='ortho').T, norm='ortho')


def get_2d_idct(coefficients):
    # Get 2D Inverse Cosine Transform of Image
    return fftpack.idct(fftpack.idct(coefficients.T, norm='ortho').T, norm='ortho')


def get_reconstructed_image(raw):
    # returns reconstructed image
    img = raw.clip(0, 255)
    img = img.astype('uint8')
    img = Image.fromarray(img)
    return img


if __name__ == '__main__':
    #import image
    in_filename = input("Enter the image file you want to compress: ")
    img = Image.open(in_filename)
    # img.show()

    # convert image into np array
    print("Compressing...")
    pixels = np.array(img, dtype=np.float)
    # pprint(pixels)
    dct_size = pixels.shape[0]
    dct = get_2D_dct(pixels)
    reconstructed_images = []

    for ii in range(dct_size):
        dct_copy = dct.copy()
        dct_copy[ii:, :] = 0
        dct_copy[:, ii:] = 0

        # Reconstructed image
        r_img = get_2d_idct(dct_copy)
        reconstructed_image = get_reconstructed_image(r_img)

        # Create a list of images
        reconstructed_images.append(reconstructed_image)

    quality = input("Enter the quality of output image [0-100]: ")
    quality = int(quality)
    quality = int(quality / 100 * 256)
    # reconstructed_images[quality].show()

    out_filename = os.path.join('./lena.jpeg')
    reconstructed_images[quality].save(out_filename, "JPEG")

    # calculating compression ratio
    in_size = int(os.path.getsize(in_filename))
    out_size = int(os.path.getsize(out_filename))
    ratio = (1 - out_size / in_size) * 100
    print("Original filesize:\t", in_size,
          " Bytes\nOutput filesize:\t", out_size, " Bytes")
    print("Compressed: {0:.2f} %".format(ratio))

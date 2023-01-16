import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.linalg as la
from ChessImage import ChessImage


class ChessLens():

    img_cropped_to_physical = None
    img_cropped_to_surface = None
    img_cropped_to_playarea = None

    def __init__(self, img):
        """
        initialization of one ChessLens instance. Make a different instance for each image (at this point).

        # todo: extract information from several images of the same position

        :param img: image from which chess position shall be detected. Image will not be changed.
        """

        # remember original image
        self.original = img
        self.img_original = ChessImage(img)

        # convert to grayscale
        self.img_gray = self.img_original.to_grayscale()

        # Blur the image for better edge detection
        img_blur = self.img_gray.to_blurred()

        # Canny Edge Detection
        self.img_canny = img_blur.to_outlines()

        # set outline image to the active image
        self.img_active = self.img_canny

    def plot(self, which_image = None, lines = None, title=None):
        """"
        plots chosen image. Choices for which_image are:
        None or "active": current active image
        "gray": original image converted to grayscale
        "original": original image
        "outlines" or "outline" or "canny": image of the outlines obtained with Canny algorithm
        """

        if which_image is None or which_image == "active":
            self.img_active.plot(lines=lines, title=title)
        elif which_image == "gray":
            self.img_gray.plot(lines=lines, title=title)
        elif which_image == "original":
            self.img_original.plot(lines=lines, title=title)
        elif which_image in ["outlines", "outline", "canny"]:
            self.img_canny.plot(lines=lines, title=title)
        elif which_image == "physical":
            self.img_cropped_to_physical.plot(lines=lines, title=title)
        else:
            print("called ChessLens.plot with invalid argument for which_image: {}".format(which_image))

    def crop_to_physical(self, min_pixels=2):
        """successively crops the active image until no rows or columns are left with less than min_pixels"""

        if self.img_active != self.img_canny:
            raise RuntimeError("ChessLens.identify_physical called when active image is not the outline image")

        img = self.img_active.img / 255
        bool_cut_rows = True
        bool_cut_cols = True

        while bool_cut_cols or bool_cut_rows:

            img, bool_cut_rows = self.cut_image(img, axis=1, min_pixels=min_pixels)
            img, bool_cut_cols = self.cut_image(img, axis=0, min_pixels=min_pixels)

        self.img_cropped_to_physical = ChessImage(img, type="physical", parent_image=self.img_canny)
        return self.img_cropped_to_physical

    def cut_image(self, img, axis, min_pixels=2):
        """finds the first row (axis = 1) or column (axis = 0) index for which the image contains less than min_pixels.
        Then finds the next entry for which the row / column has more than min_pixels entries again.
        Then cuts along both rows / columns and keeps the outside one with the larger number of pixels.
        """

        # compute how many pixels are on each row (if axis = 1) or each column (if axis = 0)
        norms = la.norm(img, axis=axis)

        # identify first almost-zero entry
        zero_index = (norms < min_pixels).argmax()

        if zero_index == 0 and norms[0] >= min_pixels:
            # no almost-empty row (axis = 1) or column (axis = 0) was found
            return img, False

        non_zero_index = (norms[zero_index:] >= min_pixels).argmax()
        if non_zero_index == 0 and norms[zero_index + non_zero_index] < min_pixels:
            # no entry right of zero_index is above min_pixels
            non_zero_index = -1
        else:
            non_zero_index += zero_index

        if axis == 1:
            # cut the rows
            top = img[:zero_index, :]
            bottom = img[non_zero_index:, :]
        else:
            # cut the columns
            top = img[:, :zero_index]
            bottom = img[:, non_zero_index:]

        # make sure not to discard the majority of information
        if la.norm(top) + la.norm(bottom) < 0.9 * la.norm(img):
            return img, False

        # keep image part with the most pixels (= the most information)
        if la.norm(top) > la.norm(bottom):
            return top, True
        else:
            return bottom, True




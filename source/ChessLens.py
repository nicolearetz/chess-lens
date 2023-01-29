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

        # rescale image to only contain zeros and ones, such that the norm is indeed the number of pixels
        img = self.img_active.img / 255
        bool_cut_left = True
        bool_cut_right = True
        bool_cut_top = True
        bool_cut_bottom = True
        reduction = 0.5

        while any([bool_cut_left, bool_cut_right, bool_cut_top, bool_cut_bottom]):

            img, bool_cut_left = self.restrict_image_from_side(img, axis=0, direction=1, min_pixels=min_pixels, max_reduction=reduction)
            img, bool_cut_top = self.restrict_image_from_side(img, axis=1, direction=1, min_pixels=min_pixels,
                                                               max_reduction=reduction)
            img, bool_cut_right = self.restrict_image_from_side(img, axis=0, direction=-1, min_pixels=min_pixels,
                                                               max_reduction=reduction)
            img, bool_cut_bottom = self.restrict_image_from_side(img, axis=1, direction=-1, min_pixels=min_pixels,
                                                              max_reduction=reduction)
            reduction /= 2

        self.img_cropped_to_physical = ChessImage(img * 255, type="physical", parent_image=self.img_canny)
        return self.img_cropped_to_physical

    def restrict_image_from_side(self, img, axis, direction, min_pixels=2, max_reduction=0.1):
        """
        going from one of the four sides, this function determines the furthest position inward it can cut the image
        such that the row / column one pixel further outward has strictly less than <min_pixels> pixels. Returns the
        original imagee if no such row / column was found. The sides can be specified as follows:

        cut from left: axis = 0, direction = 1
        cut from right: axis = 0, direction = -1
        cut from top: axis = 1, direction = 1
        cut from bottom: axis = 1, direction = -1

        The function only looks so far inward that the image height / width is not reduced by more than
        <max_reduction>*100%.

        :param img: image to be cut, scaled such that entries are between 0 and 1
        :param axis: 0 for a vertical cut, 1 for a horizontal cut
        :param direction: 1 or -1
        :param min_pixels: pixel threshold to consider a column or row empty
        :param reduction: restriction to how much of the image may be cut off
        :return:
        """

        if np.max(img) > 1 or np.min(img) < 0:
            raise RuntimeError("In ChessLens.restrict_image_from_side: img not scaled to entries within [0, 1]")

        if axis not in [0, 1]:
            raise RuntimeError("In ChessLens.restrict_image_from_side: invalid axis (received axis={})".format(axis))

        if direction not in [1, -1]:
            raise RuntimeError("In ChessLens.restrict_image_from_side: received invalid direction {}".format(direction))

        # compute how many pixels are on each row (if axis = 1) or each column (if axis = 0)
        norms = la.norm(img, axis=axis, ord=1)
        len = norms.shape[0]

        # determine how many rows / columns we actually need to look at
        scrutiny = int(max_reduction * len)
        if scrutiny <= 2:
            return img, False

        # restrict norms to the considered rows / columns
        norms = norms[::direction]
        norms = norms[:scrutiny+1]

        # identify last almost-zero index
        zero_index = scrutiny - 1
        while norms[zero_index] >= min_pixels:
            zero_index -= 1
            if zero_index == -1:
                # every norm entry is >= min_pixels
                return img, False
        non_zero_index = zero_index + 1

        # identify new starting and stopping indices
        start = np.max([0, direction * non_zero_index])
        stop = np.min([len, len + direction * non_zero_index])

        if axis == 0:
            # cutting from left or right
            return img[:, start:stop], True
        else:
            # cutting from top or bottom
            return img[start:stop, :], True



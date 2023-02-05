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

    def plot(self, which_image=None, lines=None, title=None):
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
        elif which_image == "surface":
            self.img_cropped_to_surface.plot(lines=lines, title=title)
        else:
            print("called ChessLens.plot with invalid argument for which_image: {}".format(which_image))

    def crop_to_physical(self, min_pixels=2, bool_set_active=True):
        """successively crops the active image until no rows or columns are left with less than min_pixels"""

        if self.img_active != self.img_canny:
            print("ChessLens.identify_physical called when active image is not the outline image")
            self.img_active = self.img_canny

        # introduce some blur to deal with "too good" outlines
        img = cv2.GaussianBlur(self.img_active.img, (5, 5), 0)
        # img = self.img_active.img

        # rescale image to only contain zeros and ones, such that the norm is indeed the number of pixels
        img = img / 255

        bool_cut_left = True
        bool_cut_right = True
        bool_cut_top = True
        bool_cut_bottom = True
        reduction = 0.5

        while any([bool_cut_left, bool_cut_right, bool_cut_top, bool_cut_bottom]):
            img, bool_cut_left = self.restrict_image_from_side(img, axis=0, direction=1, min_pixels=min_pixels,
                                                               max_reduction=reduction)
            img, bool_cut_top = self.restrict_image_from_side(img, axis=1, direction=1, min_pixels=min_pixels,
                                                              max_reduction=reduction)
            img, bool_cut_right = self.restrict_image_from_side(img, axis=0, direction=-1, min_pixels=min_pixels,
                                                                max_reduction=reduction)
            img, bool_cut_bottom = self.restrict_image_from_side(img, axis=1, direction=-1, min_pixels=min_pixels,
                                                                 max_reduction=reduction)
            reduction /= 2  # shrink how much we allow to be cut off to avoid cutting off the board

        new_image = ChessImage(img * 255, type="physical", parent_image=self.img_canny)
        self.img_cropped_to_physical = new_image
        if bool_set_active:
            self.img_active = new_image
        return new_image

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
        norms = norms[:scrutiny + 1]

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

    def is_top_down(self, bool_inform_image=True):
        """
        decides if a picture is taken from a top-down birdseye perspective. Picture must be cropped to physical board.

        Note:
        It might also be worth to check the number of pixels in the bottom left and bottom right triangles. Right now
        the decision is purely based on how far the extremal pixels are apart on the left, bottom, and the right sides.

        Note:
        the non-zero pixels at the top are not considered as these may come from pieces and not the board outline.

        # todo: include a buffer?

        :param bool_inform_image:
        :return:
        """

        if self.img_active != self.img_cropped_to_physical:
            raise RuntimeError("active image is not cropped to the physical board")

        pt_bottom_left = self.img_active.outside_bounds["bottom_left"]
        pt_bottom_right = self.img_active.outside_bounds["bottom_right"]

        if pt_bottom_right - pt_bottom_left > 0.2 * self.img_active.cols:
            print("Rightmost and leftmost pixels on bottom row are far apart. Image is top-down.")
            if bool_inform_image:
                self.img_active.set_key("is_top_down", True)
            return True

        pt_left_top = self.img_active.outside_bounds["left_top"]
        pt_left_bot = self.img_active.outside_bounds["left_bot"]

        if pt_left_bot - pt_left_top > 0.2 * self.img_active.rows:
            print("Topmost and bottommost pixels on left side are far apart. Image is top-down.")
            if bool_inform_image:
                self.img_active.set_key("is_top_down", True)
            return True

        pt_right_top = self.img_active.outside_bounds["right_top"]
        pt_right_bot = self.img_active.outside_bounds["right_bot"]

        if pt_right_bot - pt_right_top > 0.2 * self.img_active.rows:
            print("Topmost and bottommost pixels on left side are far apart. Image is top-down.")
            if bool_inform_image:
                self.img_active.set_key("is_top_down", True)
            return True

        if bool_inform_image:
            self.img_active.set_key("is_top_down", False)
        print("Image has perspective.")
        return False


    def crop_to_surface(self, bool_set_active=True):
        """
        identifies the surface area of the physical board. If the image is in top-down format, the surface area is
        already the physical area. If the image has perspective, the surface area is found approximately by lifting
        the outline of the bottom board edges to the height of the board, estimated by the difference in the extremal
        pixels on the left and right side. The image is then cropped such that the intersection between the lifted
        board edges lies on the (new) bottom edge. The intersection is the surface's corner at the bottom.

        The active image needs to be cropped to the physical board.

        :param bool_set_active: if true, replace active image with its cropped version
        :return:
        """

        if self.img_active != self.img_cropped_to_physical:
            raise RuntimeError("ChessLens.crop_to_surface called when active image is not cropped to physical")

        bool_is_top_down = self.img_active.get_key("is_top_down")
        bool_is_top_down = self.is_top_down(bool_inform_image=False) if bool_is_top_down is None else bool_is_top_down

        if bool_is_top_down:
            print("Image is in top down perspective. The surface is the same as the physical board")
            self.img_cropped_to_surface = self.img_cropped_to_physical
            if bool_set_active:
                self.img_active = self.img_cropped_to_surface
            return None

        # get the positions of the limiting outside points
        pt_bottom_left = self.img_active.outside_bounds["bottom_left"]
        pt_bottom_right = self.img_active.outside_bounds["bottom_right"]
        pt_left_top = self.img_active.outside_bounds["left_top"]
        pt_left_bot = self.img_active.outside_bounds["left_bot"]
        pt_right_top = self.img_active.outside_bounds["right_top"]
        pt_right_bot = self.img_active.outside_bounds["right_bot"]

        # variables for image size for better access
        n_rows = self.img_active.rows
        n_cols = self.img_active.cols

        # get the slope for going for the two lines at the bottom of the chess board
        slope_left = (n_rows-1-pt_left_bot) / pt_bottom_left
        # bottom-most non-zero pixel on left side to left-most pixel at bottom of image
        slope_right = -(n_rows-1-pt_right_bot) / (n_cols - 1 - pt_bottom_right)
        # bottom-most non-zero pixel on right side to right-most pixel at bottom of image

        # lift the previous lines such that they start at the topmost non-zero pixels at the right and left side
        # compute point where the lines intersect
        col_intersect = pt_left_top - pt_right_top + slope_right * (n_cols - 1)
        col_intersect /= (slope_right - slope_left)
        col_intersect = int(col_intersect)
        row_intersect = int(pt_left_top + slope_left * col_intersect)
        # f(x) = pt_left_top + slope_left * x
        # g(x) = pt_right_top + slope_right * (x - (n_cols-1))
        # f(x) = g(x) iff (slope_right - slope_left) * x = pt_left_top - pt_right_top _ slope_right * (n_cols - 1)

        # crop image and save as new ChessImage
        img = self.img_active.img[:row_intersect+1, :]
        new_image = ChessImage(img, type="surface", parent_image=self.img_cropped_to_physical)
        new_image.set_key("bottom-surface-corner", col_intersect)
        self.img_cropped_to_surface = new_image
        if bool_set_active:
            self.img_active = new_image

        # return outlines for the surface area
        return [[[0, col_intersect], [pt_left_top, row_intersect]],
                [[col_intersect, n_cols-1], [row_intersect, pt_right_top]]]



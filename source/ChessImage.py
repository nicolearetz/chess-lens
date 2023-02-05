import numpy as np
import matplotlib.pyplot as plt
import cv2

class ChessImage():

    child_images = []

    def __init__(self, img, type = None, parent_image = None):
        """
        Class for chess imagess. Initialize with an image in numpy.ndarray format. Can specify image type and parent
        image (if given image is cropped).

        Possible image types:
        - "original"

        todo: set outside bounds when image is in color?

        :param img:  numpy.ndarray
        :param type:  string
        :param uncropped: ChessImage
        """

        # initialization
        self.img = img  # what image you are
        self.type = "original" if type is None else type  # type of the image
        self.parent_image = parent_image  # where the image was cropped from

        if len(img.shape) == 3:
            __, self.rows, self.cols = img.shape  # dimensions of the image
            self.outside_bounds = None  # no use for outside bounds yet when image is in color
        else:
            self.rows, self.cols = img.shape  # dimensions of the image
            self.outside_bounds = self.find_outside_bounds()

        self.descriptors = {} if parent_image is None else parent_image.descriptors.copy()

    def set_key(self, key, val):
        """
        gives the ChessImage.descriptors dictionary the provided key <key> with the provided value <val>. If
        key is already present, a warning is printed (not raised) and the old value is replaced.

        todo: decide if we should raise a warning if key is changed

        :param key: string
        :param val:
        :return: no return
        """
        if key in self.descriptors.keys():
            print("Replacing ChessImage.descriptors key {} from value {} to value {}".format(key, self.descriptors[key], val))
        self.descriptors[key] = val

    def get_key(self, key, default=None):
        """
        provides the value of key <key> if present in self.descriptors. Returns provided default value if None is found.

        :param key: string
        :param default: default value to be returned if key <key> does not exist
        :return:
        """
        return self.descriptors.get(key, default)

    def vals_along_line(self, val_start, val_stop, f_start, f_stop):
        """
        linear interpolation between points (val_start, f_start) and (val_stop, f_stop). Returns
        1. numpy array with integer values from val_start to val_stop (both included)
        2. numpy array with the function values such that f_start is the first entry and f_stop the last entry

        :param val_start: int
        :param val_stop: int
        :param f_start: any scalar
        :param f_stop: any scalar
        :return:
        """

        if not isinstance(val_start, int) or not isinstance(val_stop, int):
            raise RuntimeError("ChessImage.vals_along_line called with non-integer starting or stopping values")

        direction = -1 if val_start < val_stop else 1
        slope = (f_stop - f_start) / (val_stop - val_start)

        vals = np.arange(val_start, val_stop + direction, direction, dtype=int)
        f_vals = (vals - val_start) * slope + f_start

        return vals, f_vals

    def positions_along_line(self, val_start, val_stop, f_start, f_stop):
        """
        returns pixel positions along line from (val_start, f_start) to (val_stop, f_stop), both points included, as
        two numpy arrays:
        - first from val_start to val_stop (both included)
        - second from f_start to f_stop (both included)
        The length of the returned arrays is the maximum of np.abs(f_stop - f_start) and np.abs(val_stop - val_start)
        such that the line does not skip pixels

        :param val_start: int
        :param val_stop: int
        :param f_start: int
        :param f_stop: int
        :return: np.array, np.array
        """
        # check that all values are integers
        if not all([isinstance(v, int) for v in [val_start, val_stop, f_start, f_stop]]):
            raise RuntimeError("ChessImage.positions_along_line called with non-integers")

        if np.abs(f_stop - f_start) > np.abs(val_stop - val_start):
            f_vals, vals = self.vals_along_line(val_start=f_start, val_stop=f_stop, f_start=val_start, f_stop=val_stop)
        else:
            vals, f_vals = self.vals_along_line(val_start=val_start, val_stop=val_stop, f_start=f_start, f_stop=f_stop)

        return np.rint(vals).astype(int), np.rint(f_vals).astype(int)

    def pixels_along_line(self, row_vals, col_vals):
        """
        returns the pixel values at entries (row_vals[i], col_vals[i]) where row_vals, col_vals are integer arrays

        :param row_vals: numpy array, dtype int
        :param col_vals: numpy array, dtype int, same size as row_vals
        :return: numpy array of the same length as row_vals
        """
        if len(self.img.shape) == 3:
            return self.img[:, row_vals, col_vals]
        return self.img[row_vals, col_vals]

    def count_nonzero_pixels_along_line(self, val_start, val_stop, f_start, f_stop, bool_horizontal=True, eps=0):
        """
        Counts how many non-zero entries (or entries with values > eps) are along the line
        if bool_horizontal: from column val_start, row f_start to colum val_stop, row f_stop
        otherwise: from row val_start, column f_start to row val_stop, column f_top


        :param val_start: int
        :param val_stop: int
        :param f_start: int
        :param f_stop: int
        :param bool_horizontal: bool
        :param eps: any scalar
        :return: int
        """
        vals, f_vals = self.positions_along_line(val_start, val_stop, f_start, f_stop)

        if bool_horizontal:
            pixels = self.pixels_along_line(row_vals=f_vals, col_vals=vals)
        else:
            pixels = self.pixels_along_line(row_vals=vals, col_vals=f_vals)

        return int(sum(pixels > eps))

    def plot(self, lines=None, title=None, bool_plot=True):
        """
        plots self.image. Adds additional lines if specified.

        The argument lines must be a list. Its entries are lists, either
        - of lengths 2 containing two numpy arrays with column-position, row-position
        - of length 4 containing column-start, column-stop, row-start, row-stop

        If bool_plot is False, nothing happens.

        :param lines: list
        :param title: string
        :return: None
        """
        if not bool_plot:
            return

        fig, ax = plt.subplots(1, 1)

        plt.imshow(self.img)

        if lines is not None:
            for i in range(len(lines)):
                line = lines[i]
                if len(line) == 2:
                    plt.plot(line[0], line[1])
                else:
                    plt.plot([line[0], line[1]], [line[2], line[3]])

        ax.set_title(title)
        ax.set_xlabel("cols")
        ax.set_ylabel("rows")

    def to_grayscale(self):
        """
        returns a new instance of the ChessImage class containing a grayscale version of this instance's image

        uses cv2.cvtColor with color cv2.COLOR_BGR2GRAY
        """

        img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        new_chess_image = ChessImage(img_gray, type="grayscale", parent_image=self)
        self.child_images.append(new_chess_image)
        new_chess_image.set_key("is_grayscale", True)
        return new_chess_image

    def to_blurred(self):
        """
        returns a new instance of the ChessImage class containing a blurred version of this instance's image

        uses cv2.GaussianBlur

        # todo: optimize kernel for blurring (in a differnt class)
        """

        img_blurred = cv2.GaussianBlur(self.img, (3,3), 0)
        new_chess_image = ChessImage(img_blurred, type="grayscale", parent_image=self)
        self.child_images.append(new_chess_image)
        return new_chess_image

    def to_outlines(self, canny_lower=10, canny_upper=55):
        """returns a new instance of the ChessImage class, containing a black-and-white image
        (pixels have values 0 or 255 only) with the edges found in this instance's image.

        Uses cv2.Canny to identify edges.

        # todo: optimize hyper-parameters
        """

        canny = cv2.Canny(image=self.img, threshold1=canny_lower, threshold2=canny_upper)
        new_chess_image = ChessImage(canny, type="grayscale", parent_image=self)
        self.child_images.append(new_chess_image)
        new_chess_image.set_key("has_canny_outlines", True)
        return new_chess_image

    def find_outside_bounds(self, buffer = 0):
        """
        identifies the extremal non-zero pixel positions at each side of the image, i.e. the left-most and the right-most
        non-zero pixels on the bottom and the top row, and the top-most and bottom-most non-zero pixels on the left and
        right columns. Returns the first and last pixel on each side if the side contains only zeros.

        Returns dictionary with keys:
        left_bot, left_top, right_bot, right_top for extremal pixels on the left and right sides
        bottom_left, bottom_right, top_left, top_right for extremal pixels on the top and bottom sides

        :return: dictionary
        """
        outside_bounds = {}
        outside_bounds["left_top"], outside_bounds["left_bot"] = self.find_bounds(self.img[:, 0+buffer])  # left column
        outside_bounds["right_top"], outside_bounds["right_bot"] = self.find_bounds(self.img[:, -1-buffer])  # right column
        outside_bounds["bottom_left"], outside_bounds["bottom_right"] = self.find_bounds(self.img[-1-buffer, :])  # bottom row
        outside_bounds["top_left"], outside_bounds["top_right"] = self.find_bounds(self.img[0+buffer, :])  # top row
        return outside_bounds


    def find_bounds(self, line, bool_outside_points_if_all_zero = True):
        """
        Identifies the indices of the first and last non-zero pixels on the provided line. Returns the indices of the
        first and last zero pixel if line contains zeros only (by default). Can return Nones if variable
        <bool_outside_points_if_all_zero> set to False.

        Line must be one-dimensional, and contain only values >= 0.

        :param line: array
        :return: int, int
        """

        if len(line.shape) > 1:
            raise RuntimeError("In ChessImage.find_bounds: invalid line encountered: line shape {} (not 1-dimensional)".format(line.shape))

        if min(line) < 0:
            raise RuntimeError("In ChessImage.find_bounds: invalid line encountered: minimum value {} < 0".format(min(line)))

        nonzeros = np.where(line > 0)[0]
        if nonzeros.shape[0] == 0:
            if bool_outside_points_if_all_zero:
                return 0, line.shape[0]-1
            return None, None

        first = np.min(nonzeros)
        last = np.max(nonzeros)
        return first, last




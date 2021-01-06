""" Pyramid Blending


References
----------
See the following papers, available on T-square under references:

(1) "The Laplacian Pyramid as a Compact Image Code"
        Burt and Adelson, 1983

(2) "A Multiresolution Spline with Application to Image Mosaics"
        Burt and Adelson, 1983

Notes
-----
    Created WITHOUT using cv2.pyrUp or cv2.pyrDown (only numpy and scipy)

"""
import numpy as np
import scipy as sp
import scipy.signal  # one option for a 2D convolution library
import cv2


def generatingKernel(a):
    """Return a 5x5 generating kernel based on an input parameter (i.e., a
    square "5-tap" filter.)

    Parameters
    ----------
    a : float
        The kernel generating parameter in the range [0, 1] used to generate a
        5-tap filter kernel.

    Returns
    -------
    output : numpy.ndarray
        A 5x5 array containing the generated kernel
    """
    # DO NOT CHANGE THE CODE IN THIS FUNCTION
    kernel = np.array([0.25 - a / 2.0, 0.25, a, 0.25, 0.25 - a / 2.0])
    return np.outer(kernel, kernel)


def reduce_layer(image, kernel=generatingKernel(0.4)):
    """Convolve the input image with a generating kernel and then reduce its
    width and height each by a factor of two.

    For grading purposes, it is important that you use a reflected border
    (i.e., padding equivalent to cv2.BORDER_REFLECT101) and only keep the valid
    region (i.e., the convolution operation should return an image of the same
    shape as the input) for the convolution. Subsampling must include the first
    row and column, skip the second, etc.

    Example (assuming 3-tap filter and 1-pixel padding; 5-tap is analogous):

                          fefghg
        abcd     Pad      babcdc   Convolve   ZYXW   Subsample   ZX
        efgh   ------->   fefghg   -------->  VUTS   -------->   RP
        ijkl    BORDER    jijklk     keep     RQPO               JH
        mnop   REFLECT    nmnopo     valid    NMLK
        qrst              rqrsts              JIHG
                          nmnopo

    A "3-tap" filter means a 3-element kernel; a "5-tap" filter has 5 elements.
    Please consult the lectures for a more in-depth discussion of how to
    tackle the reduce function.

    Parameters
    ----------
    image : numpy.ndarray
        A grayscale image of shape (r, c). The array may have any data type
        (e.g., np.uint8, np.float64, etc.)

    kernel : numpy.ndarray (Optional)
        A kernel of shape (N, N). The array may have any data type (e.g.,
        np.uint8, np.float64, etc.)

    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        An image of shape (ceil(r/2), ceil(c/2)). For instance, if the input is
        5x7, the output will be 3x4.
    """

    # WRITE YOUR CODE HERE.
    # define
    #print("--------entering reduce_size---------")
    ker = kernel
    height_pad = int((kernel.shape[0] - 1) / 2)
    #print("pad height ",height_pad)

    width_pad = int((kernel.shape[1] - 1) / 2)
    #print("pad width ",height_pad)

    # padding
    #print("shape before border ",image.shape)
    image = cv2.copyMakeBorder(image, height_pad, height_pad, width_pad, width_pad, borderType=cv2.BORDER_REFLECT101)
    #print("shape after border ",image.shape)
    # Convolve
    image = cv2.filter2D(image, ddepth=-1, kernel=kernel)
    #print("shape after filter2D ",image.shape)
    #print(image)
    #cv2.imshow("test",image)

    # remove padding
    image = image[height_pad:image.shape[0]-height_pad, width_pad:image.shape[1]-width_pad]
    #print("shape after removing padding ",image.shape)

    # resize
    rows_num = int(image.shape[0]/2 + 0.9) #make sure to get ceiling on divide
    cols_num = int(image.shape[1]/2 + 0.9) #make sure to get ceiling on divide
    image = cv2.resize(image, (cols_num,rows_num),  interpolation=cv2.INTER_AREA) #i guess resize is c,r instead of r,c
    # check size
    #print("shape after resize ",image.shape)
    #print("image like this \n", image[0:10,0:10])
    # return
    # needs to be float64
    image = image.astype(np.float64)

    return image


def expand_layer(image, kernel=generatingKernel(0.4)):
    """Upsample the image to double the row and column dimensions, and then
    convolve it with a generating kernel.

    Upsampling the image means that every other row and every other column will
    have a value of zero (which is why we apply the convolution after). For
    grading purposes, it is important that you use a reflected border (i.e.,
    padding equivalent to cv2.BORDER_REFLECT101) and only keep the valid region
    (i.e., the convolution operation should return an image of the same
    shape as the input) for the convolution.

    Finally, multiply your output image by a factor of 4 in order to scale it
    back up. If you do not do this (and you should try it out without that)
    you will see that your images darken as you apply the convolution.
    You must explain why this happens in your submission PDF.

    Example (assuming 3-tap filter and 1-pixel padding; 5-tap is analogous):

                                          000000
             Upsample   A0B0     Pad      0A0B0B   Convolve   zyxw
        AB   ------->   0000   ------->   000000   ------->   vuts
        CD              C0D0    BORDER    0C0D0D     keep     rqpo
        EF              0000   REFLECT    000000    valid     nmlk
                        E0F0              0E0F0F              jihg
                        0000              000000              fedc
                                          0E0F0F

                NOTE: Remember to multiply the output by 4.

    A "3-tap" filter means a 3-element kernel; a "5-tap" filter has 5 elements.
    Please consult the lectures for a more in-depth discussion of how to
    tackle the expand function.

    Parameters
    ----------
    image : numpy.ndarray
        A grayscale image of shape (r, c). The array may have any data
        type (e.g., np.uint8, np.float64, etc.)

    kernel : numpy.ndarray (Optional)
        A kernel of shape (N, N). The array may have any data
        type (e.g., np.uint8, np.float64, etc.)

    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        An image of shape (2*r, 2*c). For instance, if the input is 3x4, then
        the output will be 6x8.
    """

    # WRITE YOUR CODE HERE.
    # define
    ker = kernel
    height_pad = int((kernel.shape[0] - 1) / 2)
    #print("pad height ",height_pad)

    width_pad = int((kernel.shape[1] - 1) / 2)
    #print("pad width ",height_pad)

    #print("the image shape starts as \n", image.shape)
    nrow = image.shape[0]
    ncol = image.shape[1]
    elements = nrow * ncol
    #print("original image top left \n", image[0:8,0:8])

    # Add zeros
    # add columns
    #print("mean before adding columns ", np.mean(image))
    image = image.reshape(1,elements).ravel()
    zeros = np.zeros((1,elements)).ravel()
    new_image = [None]*elements*2
    new_image[::2] = image
    new_image[1::2] = zeros
    new_image = np.asarray(new_image)
    new_image = new_image.reshape(nrow,ncol*2)

    #add rows
    newer_image = np.zeros((new_image.shape[0]*2, new_image.shape[1]))
    newer_image[::2] = new_image
    #print("shape of new image ", new_image.shape)
    #print("shape of newer image ",newer_image.shape)
    image = newer_image
    #print("expanded matrix like \n",image[0:8,0:8])

    # Add padding
    image = cv2.copyMakeBorder(image, height_pad, height_pad, width_pad, width_pad, borderType=cv2.BORDER_REFLECT101)
    #print("expanded matrix wtih padding like \n",image[0:8,0:8])

    # convolve
    #print("mean before convole ", np.mean(image))
    image = cv2.filter2D(image, ddepth=-1, kernel=kernel)
    #print("shape after filter2D ",image.shape)
    #print("mean AFTER convolve ", np.mean(image))
    #print(image)
    #cv2.imshow("test expand",image)

    # multiply by 4
    image = image * 4
    #print("mean AFTER multiplying ", np.mean(image))
    #print("shape after multiplying ", image.shape)

    # Remove padding
    # resize
    #print("about to remove padding (still in expand)")
    image = image[height_pad:image.shape[0]-height_pad, width_pad:image.shape[1]-width_pad]
    #print("shape after removing padding ",image.shape)


    # check size
    #print("shape after resize (still in expand) ",image.shape)
    #print("expanded matrix like ",image)
    return image


def gaussPyramid(image, levels):
    """Construct a pyramid from the image by reducing it by the number of
    levels specified by the input.

    You must use your reduce_layer() function to generate the pyramid.

    Parameters
    ----------
    image : numpy.ndarray
        An image of dimension (r, c).

    levels : int
        A positive integer that specifies the number of reductions to perform.
        For example, levels=0 should return a list containing just the input
        image; levels = 1 should perform one reduction and return a list with
        two images. In general, len(output) = levels + 1.

    Returns
    -------
    list<numpy.ndarray(dtype=np.float)>
        A list of arrays of dtype np.float. The first element of the list
        (output[0]) is layer 0 of the pyramid (the image itself). output[1] is
        layer 1 of the pyramid (image reduced once), etc.
    """

    # WRITE YOUR CODE HERE.
    #print("----------entering gauss---------")
    # running expand_layer and reuce_layer so I can get the print statements
    xyz = expand_layer(image)
    abc = reduce_layer(image)
    g_list =[] # this will hold my list of arrays (guassian levels)

    image = image.astype(np.float64)
    g_list.append(image) # level 0 is the image itself
    for level in range(levels):
        image = reduce_layer(image)
        image = image.astype(np.float64)
        g_list.append(image)
    #print("len of g_list is ",len(g_list))
    #print("last element in g_list is ",g_list[-1])
    #for i in range(len(g_list)):
        #print("shape of ",i," is ",g_list[i].shape)
    return(g_list)


def laplPyramid(gaussPyr):
    """Construct a Laplacian pyramid from a Gaussian pyramid; the constructed
    pyramid will have the same number of levels as the input.

    You must use your expand_layer() function to generate the pyramid. The
    Gaussian Pyramid that is passed in is the output of your gaussPyramid
    function.

    Parameters
    ----------
    gaussPyr : list<numpy.ndarray(dtype=np.float)>
        A Gaussian Pyramid (as returned by your gaussPyramid function), which
        is a list of numpy.ndarray items.

    Returns
    -------
    list<numpy.ndarray(dtype=np.float)>
        A laplacian pyramid of the same size as gaussPyr. This pyramid should
        be represented in the same way as guassPyr, as a list of arrays. Every
        element of the list now corresponds to a layer of the laplacian
        pyramid, containing the difference between two layers of the gaussian
        pyramid.

        NOTE: The last element of output should be identical to the last layer
              of the input pyramid since it cannot be subtracted anymore.

    Notes
    -----
        (1) Sometimes the size of the expanded image will be larger than the
        given layer. You should crop the expanded image to match in shape with
        the given layer. If you do not do this, you will get a 'ValueError:
        operands could not be broadcast together' because you can't subtract
        differently sized matrices.

        For example, if my layer is of size 5x7, reducing and expanding will
        result in an image of size 6x8. In this case, crop the expanded layer
        to 5x7.
    """

    # define
    #print("inside l pyr")
    #print(gaussPyr)
    #print("this is the type of gaussPry ",type(gaussPyr))
    #print("this is the len of gaussPry ",len(gaussPyr))
    l_list = []
    len_of_g = len(gaussPyr)
    for level in range(1,len_of_g):
        level_down = gaussPyr[level - 1]
        level_expanded = expand_layer(gaussPyr[level])[0:level_down.shape[0],0:level_down.shape[1]]
        #print("level is ", level)
        #print("gaussPyr[level - 1] is type ", type(level_down))
        #print("expand_layer(gaussPyr[level]) is type ", type(level_expanded))
        l_list.append(gaussPyr[level - 1] - level_expanded)
        #print(type(l_list))
    l_list.append(gaussPyr[-1])
    #for i in range(len(gaussPyr)):
        #print("iterating through i level ", i)
        #print("shape of ",i," is ",l_list[i].shape)
        #print("type of ", type(expand_layer(gaussPyr[level])))
    return l_list

def blend(laplPyrWhite, laplPyrBlack, gaussPyrMask):
    """Blend two laplacian pyramids by weighting them with a gaussian mask.

    You should return a laplacian pyramid that is of the same dimensions as the
    input pyramids. Every layer should be an alpha blend of the corresponding
    layers of the input pyramids, weighted by the gaussian mask.

    Therefore, pixels where current_mask == 1 should be taken completely from
    the white image, and pixels where current_mask == 0 should be taken
    completely from the black image.

    (The variables `current_mask`, `white_image`, and `black_image` refer to
    the images from each layer of the pyramids. This computation must be
    performed for every layer of the pyramid.)

    Parameters
    ----------
    laplPyrWhite : list<numpy.ndarray(dtype=np.float)>
        A laplacian pyramid of an image constructed by your laplPyramid
        function.

    laplPyrBlack : list<numpy.ndarray(dtype=np.float)>
        A laplacian pyramid of another image constructed by your laplPyramid
        function.

    gaussPyrMask : list<numpy.ndarray(dtype=np.float)>
        A gaussian pyramid of the mask. Each value should be in the range
        [0, 1].

    Returns
    -------
    list<numpy.ndarray(dtype=np.float)>
        A list containing the blended layers of the two laplacian pyramids

    Notes
    -----
        (1) The input pyramids will always have the same number of levels.
        Furthermore, each layer is guaranteed to have the same shape as
        previous levels.
    """

    black_mask = np.array(gaussPyrMask) - 1
    black_mask = black_mask * -1
    black_mask = list(black_mask)
    answer = np.multiply(laplPyrWhite,  gaussPyrMask) + np.multiply(laplPyrBlack, black_mask)

    return answer


def collapse(pyramid):
    """Collapse an input pyramid.

    Approach this problem as follows: start at the smallest layer of the
    pyramid (at the end of the pyramid list). Expand the smallest layer and
    add it to the second to smallest layer. Then, expand the second to
    smallest layer, and continue the process until you are at the largest
    image. This is your result.

    Parameters
    ----------
    pyramid : list<numpy.ndarray(dtype=np.float)>
        A list of numpy.ndarray images. You can assume the input is taken
        from blend() or laplPyramid().

    Returns
    -------
    numpy.ndarray(dtype=np.float)
        An image of the same shape as the base layer of the pyramid.

    Notes
    -----
        (1) Sometimes expand will return an image that is larger than the next
        layer. In this case, you should crop the expanded image down to the
        size of the next layer. Look into numpy slicing to do this easily.

        For example, expanding a layer of size 3x4 will result in an image of
        size 6x8. If the next layer is of size 5x7, crop the expanded image
        to size 5x7.
    """

    num_layers = len(pyramid)
    result = []
    #print("Number of layers is ", num_layers)
    combined = pyramid[-1]
    for i in range(2,len(pyramid)+1):
        index = -1 * i
        combined = expand_layer(combined)[0:pyramid[index].shape[0],0:pyramid[index].shape[1]]
        combined = pyramid[index] + combined
        #print("this last item had a shape of ",combined.shape)
    return combined

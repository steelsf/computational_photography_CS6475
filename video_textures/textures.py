""" Video Textures
    NOTE:
    Please do not copy this code for the use of completing assignments at Georgia Tech.
"""
import numpy as np
import scipy as sp
import cv2
import scipy.signal


def videoVolume(images):
    """ Create a video volume (4-d numpy array) from the image list.

    Parameters
    ----------
    images : list
        A list of frames. Each element of the list contains a numpy array
        representing a color image. You may assume that each frame has the same
        shape: (rows, cols, 3).

    Returns
    -------
    numpy.ndarray(dtype: np.uint8)
        A 4D numpy array. This array should have dimensions
        (num_frames, rows, cols, 3).
    """
    images_len = len(images)
    nrow = images[0].shape[0]
    ncol = images[0].shape[1]
    images_array = np.zeros((images_len, nrow, ncol, 3)).astype(np.uint8)
    for i in range(images_len):
        images_array[i,:,:,:] = images[i]
    return images_array


def computeSimilarityMetric(video_volume):
    """Compute the differences between each pair of frames in the video volume.

    The goal, of course, is to be able to tell how good a jump between any two
    frames might be so that the code you write later on can find the optimal
    loop. The closer the similarity metric is to zero, the more alike the two
    frames are.

    Loop through each pair (i, j) of start and end frames in the video volume.
    Calculate the root sum square deviation (rssd) score for each pair and
    store the value in cell (i, j) of the output:

        rssd = sum( (start_frame - end_frame) ** 2 ) ** 0.5

    Finally, divide the entire output matrix by the average value of the matrix
    in order to control for resolution differences and distribute the values
    over a consistent range.

    Hint: Remember the matrix is symmetrical, so when you are computing the
    similarity at i, j, its the same as computing the similarity at j, i so
    you don't have to do the math twice.  Also, the similarity at all i,i is
    always zero, no need to calculate it.

    Parameters
    ----------
    video_volume : numpy.ndarray
        A 4D numpy array with dimensions (num_frames, rows, cols, 3).

        This can be produced by the videoVolume function.

    Returns
    -------
    numpy.ndarray(dtype: np.float64)
    A square 2d numpy array where output[i,j] contains the similarity
    score between the start frame at i and the end frame at j of the
    video_volume.  This matrix is symmetrical with a diagonal of zeros.
    """

    num_images = video_volume.shape[0]
    nrow = video_volume.shape[1]
    ncol = video_volume.shape[2]
    video_volume = video_volume.astype(np.float64)
    similarity_metric = np.zeros((num_images, num_images))#np.zeros((num_images, nrow, ncol, 3))
    for i in range(0, num_images-1):
        for j in range(i+1, num_images):
            rssd = np.sum( (video_volume[i] - video_volume[j]) ** 2 ) ** 0.5
            #print("rssd:\n",rssd)
            similarity_metric[i,j] = rssd
            similarity_metric[j,i] = rssd
    similarity_metric_avg = np.average(similarity_metric)
    #print("similarity metric average is ", similarity_metric_avg)
    similarity_metric = np.divide(similarity_metric, similarity_metric_avg)
    #print("here is the similarity_metric\n", similarity_metric)
    #print("here is the similarity matrix\n",similarity_metric)
    #print("it is type ", type(similarity_metric), similarity_metric.dtype)
    return similarity_metric


def transitionDifference(similarity):
    """Compute the transition costs between frames accounting for dynamics.

    Iterate through each cell (i, j) of the similarity matrix (skipping the
    first two and last two rows and columns).  For each cell, calculate the
    weighted sum:

        diff = sum ( binomial * similarity[i + k, j + k]) for k = -2...2

    Hint: There is an efficient way to do this with 2d convolution. Think about
          the coordinates you are using as you consider the preceding and
          following frame pairings.

    Parameters
    ----------
    similarity : numpy.ndarray
        A similarity matrix as produced by your similarity metric function.

    Returns
    -------
    numpy.ndarray
        A difference matrix that takes preceding and following frames into
        account. The output difference matrix should have the same dtype as
        the input, but be 4 rows and columns smaller, corresponding to only
        the frames that have valid dynamics.
    """
    binomial = binomialFilter5()
    identity_kernel = np.identity(5)
    #print("identity_kernel\n",identity_kernel)
    kernel = binomial * identity_kernel
    #print("kernel is \n", kernel)
    #print("the sim array was shape ", similarity.shape)
    transition_array_pre = cv2.filter2D(similarity, -1, kernel)
    #print("the array after convolution is shape ", transition_array_pre.shape)
    #print("and it looks like this \n",transition_array_pre)
    transition_array = transition_array_pre[2:-2,2:-2]
    #print("the array after splicing is shape ", transition_array.shape)
    #print("and it looks like this \n",transition_array)
    return transition_array


def findBiggestLoop(transition_diff, alpha):
    """Find the longest and smoothest loop for the given difference matrix.

    For each cell (i, j) in the transition differences matrix, find the
    maximum score according to the following metric:

        score = alpha * (j - i) - transition_diff[j, i]

    The pair i, j correspond to the start and end indices of the longest loop.

    **************************************************************************
      NOTE: Remember to correct the indices from the transition difference
        matrix to account for the rows and columns dropped from the edges
                    when the binomial filter was applied.
    **************************************************************************

    Parameters
    ----------
    transition_diff : np.ndarray
        A square 2d numpy array where each cell contains the cost of
        transitioning from frame i to frame j in the input video as returned
        by the transitionDifference function.

    alpha : float
        A parameter for how heavily you should weigh the size of the loop
        relative to the transition cost of the loop. Larger alphas favor
        longer loops, but may have rough transitions. Smaller alphas give
        shorter loops, down to no loop at all in the limit.

    Returns
    -------
    int, int
        The pair of (start, end) indices of the longest loop after correcting
        for the rows and columns lost due to the binomial filter.
    """
    #score = alpha * (j - i) - transition_diff[j, i]
    #print("shape of transition_array is ", transition_diff.shape)
    score_array = np.zeros((transition_diff.shape[0], transition_diff.shape[1]))
    for i in range(score_array.shape[0]):
        for j in range(score_array.shape[1]):
            score = alpha * (j - i) - transition_diff[j, i]
            score_array[i,j] = score
    #print("score array after for loops: \n",score_array)
    max_element = np.amax(score_array)
    #print("max_element is ", max_element)
    max_element_r_c = np.where(score_array == max_element)
    #print("max_element_r_c ", max_element_r_c)
    max_index_row = max_element_r_c[0][0]
    max_index_col = max_element_r_c[1][0]
    return max_index_row + 2, max_index_col + 2


def synthesizeLoop(video_volume, start, end):
    """Pull out the given loop from the input video volume.

    Parameters
    ----------
    video_volume : np.ndarray
        A (time, height, width, 3) array, as created by your videoVolume
        function.

    start : int
        The index of the starting frame.

    end : int
        The index of the ending frame.

    Returns
    -------
    list
        A list of arrays of size (height, width, 3) and dtype np.uint8,
        similar to the original input to the videoVolume function.
    """
    video_clip = []
    for index in range(start, end+1):
        temp_image = video_volume[index,:,:,:]
        video_clip.append(temp_image)
    return video_clip


def binomialFilter5():
    """Return a binomial filter of length 5.

    NOTE: DO NOT MODIFY THIS FUNCTION.

    Returns
    -------
    numpy.ndarray(dtype: np.float)
        A 5x1 numpy array representing a binomial filter.
    """
    return np.array([1 / 16., 1 / 4., 3 / 8., 1 / 4., 1 / 16.], dtype=float)

'''
fake_image_1 = np.array(np.tile(range(25),3)).reshape(5,5,3)
fake_image_2 = np.array(np.repeat(range(25),3)).reshape(5,5,3)
fake_images_list = [fake_image_2, fake_image_2+1, fake_image_2+2, fake_image_2+3, fake_image_2+2, fake_image_2+1, fake_image_2+-1]
#print(fake_image_2)
vidvol = videoVolume(fake_images_list)
print("vidvol output is ",vidvol.dtype, type(vidvol), vidvol.shape)
print("output of vidvol has shape ", vidvol.shape) #, " and looks like this \n", vidvol)
sim = computeSimilarityMetric(vidvol)
print("sim is like this ", sim.dtype, type(sim), sim.shape)
print("Here is sim: \n",sim)
trans = transitionDifference(sim)
print("trans is like ",trans.dtype, type(trans), trans.shape)
print("trans \n",trans)
i,j = findBiggestLoop(trans, 200)
print("i and j are thiese types ", type(i), type(j))
print("and i, j are ", i,j)
image_list = synthesizeLoop(vidvol, i, j)
print("now printing image list\n")
print(image_list)
'''

import numpy as np

def relative_normalization_one_image(master_im, slave_im, position):
    # This function normalize an image relative to its one bordering image as described in
    # "Intensity Mapping for Mask Projection based Photopolymerization".
    #
    # Positions:
    #   0
    # 3 x 2
    #   1
    #
    # 0: slave_im above master_im
    # 1: slave_im below master_im
    # 2: slave_im right for master_im
    # 3: slave_im left for master_im

    M = np.ones(np.shape(master_im))

    if position == 0:

        boarder_master = master_im[0:1, :]
        boarder_slave = slave_im[-2:-1, :]

    elif position == 1:

        boarder_master = master_im[-2:-1, :]
        boarder_slave = slave_im[0:1, :]

    elif position == 2:

        boarder_master = master_im[:, -2:-1]
        boarder_slave = slave_im[:, 0:1]

    else:

        boarder_master = master_im[:, 0:1]
        boarder_slave = slave_im[:, -2:-1]

    factor = boarder_master / boarder_slave

    # multiply by filter
    M = factor * M

    return M, factor


def relative_normalization_two_images(master_im1, master_im2, slave_im, position):
    # This function normalize an image relative to it's two bordering images as described in
    # "Intensity Mapping for Mask Projection based Photopolymerization".
    #
    # Positions:
    # 1 2 3
    # 4 5 6
    # 7 8 9
    #
    # 0: master_im1 = 2, master_im2 = 6, slave_im = 3
    # 1: master_im1 = 6, master_im2 = 8, slave_im = 9
    # 2: master_im1 = 8, master_im2 = 4, slave_im = 7
    # 3: master_im1 = 4, master_im2 = 2, slave_im = 1

    # Initialize weights
    W1 = np.zeros(np.shape(master_im1))
    W2 = np.zeros(np.shape(master_im2))

    m, n = np.shape(W1)

    if position == 0:

        for h in range(1, m + 1):
            for w in range(1, n + 1):
                W1[h - 1, w - 1] = h / (n + h - w)
                W2[h - 1, w - 1] = (n - w) / (n + h - w)

        M11, factor11 = relative_normalization_one_image(master_im1, slave_im, 2)
        M22, factor22 = relative_normalization_one_image(master_im2, slave_im, 0)

    elif position == 1:

        # Calculate weight matrix
        for h in range(1, m + 1):
            for w in range(1, n + 1):
                W1[h - 1, w - 1] = h / (h + w)
                W2[h - 1, w - 1] = w / (h + w)

        M11, factor11 = relative_normalization_one_image(master_im1, slave_im, 1)
        M22, factor22 = relative_normalization_one_image(master_im2, slave_im, 2)

    elif position == 2:

        for h in range(1, m + 1):
            for w in range(1, n + 1):
                W1[h - 1, w - 1] = h / (n + h - w)
                W2[h - 1, w - 1] = (n - w) / (n + h - w)

        W1 = np.rot90(W1, 2)
        W2 = np.rot90(W2, 2)

        M11, factor11 = relative_normalization_one_image(master_im1, slave_im, 3)
        M22, factor22 = relative_normalization_one_image(master_im2, slave_im, 1)

    else:

        # Calculate weight matrix
        for h in range(1, m + 1):
            for w in range(1, n + 1):
                W1[h - 1, w - 1] = h / (h + w)
                W2[h - 1, w - 1] = w / (h + w)

        W1 = np.rot90(W1, 2)
        W2 = np.rot90(W2, 2)

        M11, factor11 = relative_normalization_one_image(master_im1, slave_im, 0)
        M22, factor22 = relative_normalization_one_image(master_im2, slave_im, 3)

    M = W1 * factor11 + W2 * factor22

    return M
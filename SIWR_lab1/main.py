import numpy as np


def main():
    # TODO 1 -----------------------------------------------------------------------------------------------------------

    cumulative_arr = np.array([[[0.108, 0.012],
                                [0.016, 0.064]],
                               [[0.072, 0.008],
                                [0.144, 0.576]]])

    # TODO 2 -----------------------------------------------------------------------------------------------------------

    p_too = np.sum(np.sum(cumulative_arr, axis=1), axis=1)
    print('P_too =\n', p_too)

    # TODO 3 -----------------------------------------------------------------------------------------------------------

    p_cav = np.sum(np.sum(cumulative_arr, axis=0), axis=1)
    print('P_cav =\n', p_cav)

    # TODO 4 -----------------------------------------------------------------------------------------------------------

    p_too_giv_cav = np.sum(cumulative_arr, axis=2) / p_cav
    print('P_too_giv_cav =\n', p_too_giv_cav)

    # TODO 5 -----------------------------------------------------------------------------------------------------------

    p_too_or_cat = p_too[0] + np.sum(cumulative_arr, axis=1)[1][0]
    p_cav_and_too_or_cat = np.sum(np.sum(cumulative_arr, axis=0), axis=1) - np.array(
        [cumulative_arr[1][0][1], cumulative_arr[1][1][1]])
    p_cav_giv_too_or_cat = p_cav_and_too_or_cat / p_too_or_cat
    print('P_cav_giv_too_or_cat =\n', p_cav_giv_too_or_cat)

    # TODO 6 -----------------------------------------------------------------------------------------------------------
    # Wielkość tablicy rośnie wykładniczo zgodnie z funkcją 2^x, gdzie x jest liczbą zmiennych.

    # TODO 7 -----------------------------------------------------------------------------------------------------------

    memory = round(np.power(2, 32) * 32 / (8 * 1024 * 1024 * 1024), 2)
    print('Memory: ', memory, 'GB')

    # TODO 8 -----------------------------------------------------------------------------------------------------------

    transposed_cumm = cumulative_arr.transpose(1, 0, 2)
    alpha_too_cat = 1 / np.sum(np.sum(transposed_cumm, axis=1), axis=1)
    p_too_and_cat_giv_cav = np.broadcast_to(alpha_too_cat, (2, 2, 2)).transpose(2, 1, 0) * transposed_cumm
    p_too_and_cat_giv_cav_times_p_cav = np.broadcast_to(p_cav, (2, 2, 2)).transpose(2, 1, 0) * p_too_and_cat_giv_cav

    print(p_too_and_cat_giv_cav)

    p_cav_giv_too_and_cat_1 = np.empty((2, 2, 2))
    for i in range(2):
        for j in range(2):
            ij_sum = p_too_and_cat_giv_cav_times_p_cav[0][i][j] + p_too_and_cat_giv_cav_times_p_cav[1][i][j]
            alpha_cav = 1 / ij_sum
            p_cav_giv_too_and_cat_1[0][i][j] = p_too_and_cat_giv_cav_times_p_cav[0][i][j] * alpha_cav
            p_cav_giv_too_and_cat_1[1][i][j] = p_too_and_cat_giv_cav_times_p_cav[1][i][j] * alpha_cav

    print('P_cav_giv_too_and_cat_1 =\n', p_cav_giv_too_and_cat_1)

    # TODO 9 -----------------------------------------------------------------------------------------------------------
    # Zmienne Toothache i Catch są od siebie zależne ale mając dane Cavity stają się warunkowo niezależne.

    # TODO 10 ----------------------------------------------------------------------------------------------------------

    print(p_too_giv_cav)
    p_cat_giv_cav = (np.sum(cumulative_arr, axis=0) / p_cav[:, np.newaxis]).T
    print(p_cat_giv_cav)
    print('P_cat_giv_cav  =\n', p_cat_giv_cav)
    cav_yes = np.dot(p_too_giv_cav[:, 0][:, np.newaxis], p_cat_giv_cav[:, 0][np.newaxis, :]) * p_cav[0]
    cav_no = np.dot(p_too_giv_cav[:, 1][:, np.newaxis], p_cat_giv_cav[:, 1][np.newaxis, :]) * p_cav[1]

    p_cav_giv_too_and_cat_2 = np.empty((2, 2, 2))
    for i in range(2):
        for j in range(2):
            ij_sum = cav_no[i][j] + cav_yes[i][j]
            alpha_cav = 1 / ij_sum
            p_cav_giv_too_and_cat_2[0][i][j] = cav_yes[i][j] * alpha_cav
            p_cav_giv_too_and_cat_2[1][i][j] = cav_no[i][j] * alpha_cav

    print('P_cav_giv_too_and_cat_2 =\n', p_cav_giv_too_and_cat_2)

    # TODO 11 ----------------------------------------------------------------------------------------------------------

    cav_yes_new = np.dot(p_too_giv_cav[:, 0][:, np.newaxis], p_cat_giv_cav[:, 0][np.newaxis, :]) * p_cav[0]
    cav_no_new = np.dot(p_too_giv_cav[:, 1][:, np.newaxis], p_cat_giv_cav[:, 1][np.newaxis, :]) * p_cav[1]

    cumulative_arr_new = np.array([cav_yes_new, cav_no_new]).transpose(1, 0, 2)

    print('New cumulative array =\n', cumulative_arr_new)



if __name__ == '__main__':
    main()

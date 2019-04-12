import numpy as np


def convert_ids(original_ids):
    '''
    Convert scattered ids (from 0 to 42) to smooth range of ids (0 to 31)

    Parameters
    ----------
    orignal_ids : np.array of original ids

    Returns
    ----------
    new ids in range 0 to 31

    '''

    # all possible ids
    all_ids = [0., 1., 4., 5., 6., 8., 9., 10., 11., 12., 15., 16., 18.,
               19., 20., 21., 22., 23., 25., 26., 27., 28., 29., 30., 31., 32.,
               33., 34., 35., 38., 40., 42.]

    # get index of ids in list
    if original_ids.size > 1:
        return np.array([all_ids.index(id) for id in original_ids]).flatten()

    else:
        return np.array(all_ids.index(original_ids))


def convert_back_ids(converted_ids):
    '''
    Convert smooth range of ids (0 to 31) back to scattered ids (from 0 to 42)

    Parameters
    ----------
    converted_ids : np.array of converted ids

    Returns
    ----------
    original ids in range 0 to 42

    '''

    FAKE_TO_REAL_ID = {0: 0.0, 1: 1.0, 2: 4.0, 3: 5.0, 4: 6.0, 5: 8.0,
                       6: 9.0, 7: 10.0, 8: 11.0, 9: 12.0, 10: 15.0,
                       11: 16.0, 12: 18.0, 13: 19.0, 14: 20.0, 15: 21.0,
                       16: 22.0, 17: 23.0, 18: 25.0, 19: 26.0, 20: 27.0,
                       21: 28.0, 22: 29.0, 23: 30.0, 24: 31.0, 25: 32.0,
                       26: 33.0, 27: 34.0, 28: 35.0, 29: 38.0, 30: 40.0,
                       31: 42.0}

    if converted_ids.size > 1:
        return np.array([FAKE_TO_REAL_ID[label] for label in converted_ids]).flatten()

    else:
        return FAKE_TO_REAL_ID[int(converted_ids)]

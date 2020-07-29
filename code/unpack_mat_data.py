from scipy.io import loadmat


def load_mat_data(file_path=r'C:\Users\Nabbefeld\Desktop\NMA\AttentionRivalryModel\matlab_timecourse.mat'):
    mat_data = loadmat(file_path)['results']
    variable_names = mat_data.dtype.names

    results = dict()
    for variable in variable_names:
        results[variable] = dict()
        for key in mat_data[variable][0, 0].dtype.names:
            results[variable][key] = mat_data[variable][0, 0][0, 0][key]
        #
    #

    return results
#

import os


def folder_list(folder_path, endswith=None):
    rtn = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            rtn.append(os.path.join(root, file))

    if endswith is None:
        return rtn
    else:
        return [file for file in rtn if file.endswith(endswith)]


def unique_path(path):
    if os.path.exists(path):
        path = path + "_"
        i = 2
        while os.path.exists(path + str(i)):
            i += 1
        path = path + str(i)
    return path
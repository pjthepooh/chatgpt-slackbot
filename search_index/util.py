import os


def read_from_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
        return data


def load_files_from_directory(directory_path):
    data_dict = {}

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        if os.path.isfile(file_path) and file_path.endswith(".txt"):
            data = read_from_file(file_path)

            if data is not None:
                data_dict[filename] = data

    return data_dict

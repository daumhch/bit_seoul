import os



def remove_file(path):
    for file in os.scandir(path):
        os.unlink(file.path)
remove_file('./model')

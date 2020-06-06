import os

def num_to_vector(num):
    vector = [int(b) for b in bin(num)[2:]]
    vector = (16 - len(vector)) * [0] + vector

    return vector

def create_dir(logdir, model_name):
    nr = None
    dirs = os.listdir(logdir)

    try:
        nr = max([int(d.split('-')[0]) for d in dirs if d.split('-')[-1] == model_name])
    except:
        nr = -1
    nr += 1

    path = os.path.join(logdir, '{}-{}'.format(nr, model_name))
    os.makedirs(path)

    return path

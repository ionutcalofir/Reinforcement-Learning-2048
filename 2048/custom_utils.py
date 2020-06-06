def num_to_vector(num):
    vector = [int(b) for b in bin(num)[2:]]
    vector = (16 - len(vector)) * [0] + vector

    return vector

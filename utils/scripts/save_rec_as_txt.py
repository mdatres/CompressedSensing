def save_rec_as_txt(name, signal):
    with open(name, 'w') as f:
        # create the csv writer
        np.savetxt(name,signal,delimiter=',')
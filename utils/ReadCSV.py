import pandas as pd
import datatable as dt
import numpy as np

class ReadCSV():  # S79D61
    def __init__(self, filepath) -> None:
        self.data = dt.fread(filepath).to_pandas()

    def arrayReshape(self, a, num_of_split):
        return np.vstack(np.hsplit(a, indices_or_sections=num_of_split))
        
    def readAll(self, block_shape=(10296, 79*61)):
        array = np.array(self.data.iloc[1:block_shape[0]+1, 6:6+79*61]).astype(np.float)
        target_array = [i+1 for i in range(79)]
        target_array = np.array(target_array, dtype=np.float)
        target_array = np.expand_dims(target_array, axis=0).repeat(block_shape[0], axis=0)

        array = self.arrayReshape(array, 79)
        target_array = self.arrayReshape(target_array, 79)

        return array, target_array

if __name__ == '__main__':
    object = ReadCSV('data/All.csv')
    object.readAll()
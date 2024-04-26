
import itertools

import numpy as np 


part_names = ['cup', 'handle']

target_pcds = {'cup' : [1, 2, 3], 'handle': [1,2]}
target_labels = np.array(list(itertools.chain(*[[i for _ in range(len(target_pcds[part]))] for i, part in enumerate(part_names)])))
print(target_labels)
import numpy as np
np.random.seed(42)



indoor_indices = np.arange(5700, 8570) - 2095
outdoor_indices = np.concatenate([np.arange(3800, 5450),
                                np.arange(8890, 10730)]) - 2095

delta = 1
test_portion = 0.1
scenes = ['indoor', 'outdoor']
indices = [indoor_indices, outdoor_indices]

for s_i, scene in enumerate(scenes):
    idx = indices[s_i]
    np.random.shuffle(idx)
    num_test = int(idx.shape[0]*test_portion)
    num_train = idx.shape[0] - num_test
    
    train_idx = np.empty((num_train*2, 2), dtype=np.int32)
    train_idx[:num_train, 0] = idx[:num_train]
    train_idx[:num_train, 1] = idx[:num_train] + delta
    train_idx[num_train:, 0] = idx[:num_train] + delta
    train_idx[num_train:, 1] = idx[:num_train] 

    test_idx = np.empty((num_test, 2), dtype=np.int32)
    test_idx[:, 0] = idx[num_train:]
    test_idx[:, 1] = idx[num_train:] + delta
    
    np.savetxt(scene+"_train.csv", train_idx, fmt='%i', delimiter=",")
    np.savetxt(scene+"_test.csv", test_idx, fmt='%i',delimiter=",")


transition_indices = np.concatenate([np.arange(5450, 5699),
                                np.arange(8570, 8900)]) - 2095
np.random.shuffle(transition_indices)
idx = np.empty((transition_indices.shape[0], 2), dtype=np.int32)
idx[:, 0] = transition_indices
idx[:, 1] = transition_indices + delta
np.savetxt("transition.csv", idx, fmt='%i', delimiter=",")
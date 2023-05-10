import jax
import numpy as np
import copy
import pickle


# replay buffer
class replay_buffer(object):
    def __init__(self, data_limit, dataset_dir):
        self.data = None
        self.size = data_limit
        self.dataset_dir = dataset_dir
        self.load_idx = 0

    def push(self, data):
        if self.data is None:
            self.data = jax.tree_util.tree_map(lambda x : x[:self.size], data)
        else:
            self.data = jax.tree_util.tree_map(lambda *x: np.concatenate(x, axis=0)[:self.size], data, self.data)

    def set_val_data(self):
        self.val_data = copy.deepcopy(self.data)
        self.data = None
        self.dataset_dir = self.dataset_dir[self.load_idx:]
        self.load_idx = 0

    def sample(self, size=500000, type='train'):
        if type=='train':
            idx = np.random.choice(np.arange(self.get_train_size()), size=(size,), replace=False)
            return jax.tree_util.tree_map(lambda x : x[idx], self.data)
        elif type=='val':
            idx = np.random.choice(np.arange(self.get_val_size()), size=(size,), replace=False)
            return jax.tree_util.tree_map(lambda x : x[idx], self.val_data)

    def load(self):
        with open(self.dataset_dir[self.load_idx], 'rb') as f:
            self.push(pickle.load(f))
        self.load_idx += 1
        self.load_idx = self.load_idx%len(self.dataset_dir)

    def shuffle(self, jkey):
        idx = np.random.permutation(self.get_size())
        # idx = jax.random.permutation(jkey, jnp.arange(self.get_size()))
        self.data = jax.tree_util.tree_map(lambda x : x[idx], self.data)

    def data_slice(self, size):
        self.data = jax.tree_util.tree_map(lambda x : x[:size], self.data)

    def get_size(self):
        # return self.data[0].shape[0]
        return jax.tree_util.tree_leaves(self.data)[0].shape[0]

    def get_train_size(self):
        # return int(self.get_size() * 0.8)
        return jax.tree_util.tree_leaves(self.data)[0].shape[0]

    def get_val_size(self):
        # tsize = self.get_size()
        # return tsize - int(tsize * 0.8)
        return jax.tree_util.tree_leaves(self.val_data)[0].shape[0]

    def get_dir_size(self):
        return len(self.dataset_dir)
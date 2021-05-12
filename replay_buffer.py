import numpy as np
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)


class ReplayBuffer(object):
    def __init__(self, size, frame_height=84, frame_width=84):
        """ Args:
            size: Integer, Number of stored transitions
            frame_height: Integer, Height of a frame of an Atari game
            frame_width: Integer, Width of a frame of an Atari game
            agent_history_length: Integer, Number of frames stacked together to create a state
        """
        self._maxsize = size
        self._next_idx = 0        # 向经验池添加样本应该添加到的位置
        self._storage = 0         # 当前容量
        self.batch_size = 32

        # Pre-allocate memory
        self.states = np.empty((size, frame_height, frame_width), dtype=np.uint8)
        self.actions = np.empty(size, dtype=np.int64)
        self.rewards = np.empty(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.bool)

    def add(self, obs_t, action, reward, done):
        self.states[self._next_idx] = obs_t[:, :, -1]
        self.actions[self._next_idx] = action
        self.rewards[self._next_idx] = reward
        self.dones[self._next_idx] = done

        # update index and storage
        self._next_idx = (self._next_idx + 1) % self._maxsize
        self._storage += 1
        self._storage = min(self._maxsize, self._storage)

    def find_done_index(self):
        # 找到 done 所在的位置，注意要去除与 self._next_idx 靠近的后方的位置，因为该位置指示的前方周期已经不完整.
        terminal_array = np.where(self.dones == True)[0]
        if self._storage == self._maxsize:
            for j in range(self._next_idx, self._maxsize):
                if self.dones[j] == True:
                    delete_index = j
                    # print("terminal_array:", terminal_array)
                    # print("delete_index:", delete_index)
                    return np.delete(terminal_array, np.where(terminal_array == delete_index))
                else:
                    return terminal_array
        else:
            return terminal_array
        return terminal_array

    def sample(self):
        terminal_array = self.find_done_index()

        # batchnum = 0
        # while batchnum == 0:
        # exclude some early and final episodes from sampling due to indexing issues,
        # sample two episodes (ind1 for main, and ind2 for the remaining steps to make multiple of 32)
        ind = np.random.choice(range(5, len(terminal_array)-3), 2, replace=False)
        ind1 = ind[0]
        ind2 = ind[1]

        indice_array = range(terminal_array[ind1], terminal_array[ind1-1], -1)    # 逆序
        epi_len = len(indice_array)
        batchnum = int(np.ceil(epi_len/float(self.batch_size)))                   # 取上界
        assert batchnum > 0

        remainindex = int(batchnum * self.batch_size + 3 - epi_len)
        # print("remainindex:", remainindex, ", first episode length:", epi_len, ", ind1:", ind1, ", ind2:", ind2)

        # Normally an episode does not have steps=multiple of 32.
        # Fill last minibatch with redundant steps from another episode
        indice_array = np.append(indice_array, range(terminal_array[ind2], terminal_array[ind2]-remainindex, -1))
        indice_array = indice_array.astype(int)
        # print("sample index:", indice_array, ", length:", indice_array.shape)

        # SAMPLE
        dones = self.dones[indice_array]
        states = self.states[indice_array].copy()                 # (None,84,84)
        # print(dones.shape, dones.astype(np.int))
        # print(states.shape)
        # states
        states_stack_list = []
        for s_idx in range(0, states.shape[0]-3):
            if dones[s_idx + 1] == 1:
                s_stack = states[np.array([s_idx, s_idx, s_idx, s_idx])]
            elif dones[s_idx + 2] == 1:
                s_stack = states[np.array([s_idx+1, s_idx+1, s_idx+1, s_idx])]
            elif dones[s_idx + 3] == 1:
                s_stack = states[np.array([s_idx+2, s_idx+2, s_idx+1, s_idx])]
            else:
                s_stack = states[np.array([s_idx+3, s_idx+2, s_idx+1, s_idx])]
            states_stack_list.append(s_stack)
        states_stack = np.stack(states_stack_list, axis=0).transpose((0, 2, 3, 1))  # (None,84,84,4)
        # print(states_stack.shape)

        rewards = self.rewards[indice_array]
        actions = self.actions[indice_array]
        return states_stack, actions[:-3], rewards[:-3], batchnum, dones[:-3]

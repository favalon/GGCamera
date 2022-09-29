class lr_dummy_scheduler():
    def update(self, lr):
        return lr

    def epoch_end(self, lr):
        return lr


class lr_step_scheduler():
    def __init__(self, steps, alfa, min_lr):
        self.steps = steps
        self.alfa = alfa
        self.min_lr = min_lr
        self.update_idx = 0

    def update(self, lr):
        self.update_idx += 1

        if self.update_idx % self.steps == 0:
            return max(lr * self.alfa, self.min_lr)

        return lr

    def epoch_end(self, lr):
        return lr


class lr_step_epoch_scheduler():
    def __init__(self, steps, lrs):
        self.steps = steps
        self.steps_idx = 0
        self.lrs = lrs
        self.update_idx = 0

    def update(self, lr):
        return lr

    def epoch_end(self, lr):
        self.update_idx += 1

        if self.update_idx - self.steps[self.steps_idx] == 0:
            self.steps_idx = min(self.steps_idx + 1, len(self.steps) - 1)

        return self.lrs[self.steps_idx]


'''
This is to keep track of the number of gradient updates done, for tensorboard. 
'''


class count_updates():
    def __init__(self, num_updates):
        self.num_updates = num_updates

    ''' after gradient update, make sure you add this new update to the counter'''

    def update(self):
        self.num_updates = self.num_updates + 1

    '''at the beginning of a new epoch, we want to know where we are in terms of update'''

    def get_number_updates(self):
        return self.num_updates

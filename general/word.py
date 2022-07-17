import math
import time

class UserWordDict:
    def __init__(self, user_id, current_time, word_dict):
        self.user_id = user_id
        self.current_time = current_time
        self._initial_word_dict(word_dict)

    def add_word_dict(self, word_dict):
        return

    def _initial_word_dict(self, word_dict):
        memory_word_list = []
        for word in word_dict:
            new_word = Word(word)
            memory_word_list.append(new_word)

        return memory_word_list


class Word:
    memory_score = 0  # 越大则记得越清楚
    pre_time = 0
    visit_times = 0

    def __init__(self, user_s, word, root, ety_explain, frequency, memory_score=None, time=None):
        '''
        :param user_s: 用户学习能力
        :param word:
        :param root:
        :param ety_explain:
        :param frequency:
        :param memory_score:
        :param time:
        '''

        self.user_s = user_s
        self.word = word
        self.root = root
        self.ety_explain = ety_explain
        if memory_score:
            self.memory_score = memory_score
        else:
            self.memory_score = self._initial_memory_score(frequency)
        if time:
            self.pre_time = time
        else:
            self.pre_time = time.time()

    def _initial_memory_score(self, cur_time, frequency):
        self.pre_time = cur_time
        r_t = 0
        # frequency default score
        r1 = frequency * self._forget_value(cur_time)

        r_t += r1

        return r_t

    def _forget_value(self, cur_time):
        # 计算forget_value 基于遗忘函数 R = e^(-t/s), user_s是用户的记忆水品
        dt = cur_time - self.pre_time
        r = math.exp(-dt/self.user_s)
        r = r * self.visit_times
        return r

    def update_memory_score(self, cur_time, react_time, question_type):

        self.memory_score = self._forget_value(cur_time)
        self.visit_times += 1
        return

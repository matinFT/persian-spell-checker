from gensim.models import Word2Vec, FastText
import parsivar
import pickle
import numpy as np
import pandas as pd
from polyleven import levenshtein
import time
import multiprocessing
import functools

normalizer = parsivar.Normalizer()
tokenizer = parsivar.Tokenizer()


class spellChecker:
    def __init__(self, model, word_counts, biword_counts, threeword_counts, fourword_counts,
                 candidate_count_tresh=50, candidate_sim_tresh = 0.6, ngram_weights=[1, 1, 1],
                 max_editdistance=2, close_extra_char_penalty = 0.1, far_extra_char_penalty = 0.001,
                 missed_char_penalty = 0.01, close_replace_char_penalty = 0.1, far_replace_char_penalty = 0.001):
        self.model = model
        self.word_counts = word_counts
        self.biword_counts = biword_counts
        self.threeword_counts = threeword_counts
        self.fourword_counts = fourword_counts
        self.candidate_count_tresh = candidate_count_tresh
        self.candidate_sim_tresh = candidate_sim_tresh

        self.ngram_weights = np.array(ngram_weights)
        self.max_editdistance = max_editdistance

        self.close_extra_char_penalty = close_extra_char_penalty
        self.far_extra_char_penalty = far_extra_char_penalty
        self.missed_char_penalty = missed_char_penalty
        #         self.close_extra_char_penalty = 0.1
        self.close_replace_char_penalty = close_replace_char_penalty
        self.far_replace_char_penalty = far_replace_char_penalty

    def cal_proposed_word_penalty(self, word, proposed_word, m=0, n=0):
        distance = levenshtein(word, proposed_word)

        return 0.01**distance

        if m == len(word):
            if n == len(proposed_word):
                return 1
            x = [self.close_extra_char_penalty if proposed_word[i] in  character_neighbors.get(proposed_word[ i -1], []) else
                 self.far_extra_char_penalty for i in range(n, len(proposed_word))]
            return np.prod(x)

        if n == len(proposed_word):
            #             if m == len(word):
            #                 return 1
            return self.missed_char_penalty**(len(word )-m)

        if word[m] == proposed_word[n]:
            return self.cal_proposed_word_penalty(word, proposed_word, m+ 1, n + 1)

        neighbors = character_neighbors.get(proposed_word[n], [])

        if (n >= 1 and proposed_word[n - 1] in neighbors) or (
                n <= len(proposed_word) - 2 and proposed_word[n + 1] in neighbors):
            addition_cost = self.close_extra_char_penalty
        else:
            addition_cost = self.far_extra_char_penalty

        if word[m] in neighbors:
            replace_cost = self.close_replace_char_penalty
        else:
            replace_cost = self.far_replace_char_penalty

        return (max(addition_cost * self.cal_proposed_word_penalty(word, proposed_word, m, n + 1),  # delete letter
                    self.missed_char_penalty * self.cal_proposed_word_penalty(word, proposed_word, m + 1, n),
                    # add letter
                    replace_cost * self.cal_proposed_word_penalty(word, proposed_word, m + 1, n + 1)))  # replace letter

    def check_if_need_replacement(self, query_words, context):
        res = []
        count_threshold = 2
        word_in_vocab = [x in self.model.wv.key_to_index for x in query_words]
        if len(query_words) == 1:
            return [not word_in_vocab[0]]

        for i in range(len(query_words)):

            if not word_in_vocab[i] or self.model.wv.cosine_similarities(
                    self.model.wv[query_words[i]], context) < self.candidate_sim_tresh:
                res.append(True)
                continue
            #             if not word_in_vocab:
            #                 res.append(True)
            #             elif i > 0 and i < len(query_words)-1 and (
            #                 self.biword_counts.get((self.model.wv.key_to_index.get(query_words[i-1], None),
            #                                         self.model.wv.key_to_index.get(query_words[i], None), 0)) <= count_threshold and
            #                 self.biword_counts.get((self.model.wv.key_to_index.get(query_words[i], None),
            #                                         self.model.wv.key_to_index.get(query_words[i+1], None), 0)) <= count_threshold):
            #                 res.append(True)
            #             elif i == 0 and self.biword_counts.get((self.model.wv.key_to_index.get(query_words[i], None),
            #                                                     self.model.wv.key_to_index.get(query_words[i+1], None)), 0) <= count_threshold:
            #                 res.append(True)
            #             elif self.biword_counts.get((self.model.wv.key_to_index.get(query_words[i-1], None),
            #                                          self.model.wv.key_to_index.get(query_words[i], None)), 0) <= count_threshold:
            #                 res.append(True)
            else:
                res.append(False)
        return res

    def cal_phrase_candidates(self, word_candidates, query_words, save_time=False):
        t = []
        l = []
        t.append(time.time())
        if len(word_candidates) == 1:
            return [[x] for x in word_candidates[0]]
        res = [[x] for x in word_candidates[0]]

        t.append(time.time())
        l.append(len(res))

        res = [x + [y] for x in res for y in word_candidates[1] if
               self.biword_counts.get((self.model.wv.key_to_index.get(x[-1], None),
                                       self.model.wv.key_to_index.get(y, None)), 0) >= 1]
        if save_time == True:
            if len(res) > 50000:
                return None
        t.append(time.time())
        l.append(len(res))

        for i in range(2, len(word_candidates)):

            res = [x + [y] for x in res for y in word_candidates[i] if
                   self.threeword_counts.get((self.model.wv.key_to_index.get(x[-2], None),
                                              self.model.wv.key_to_index.get(x[-1], None),
                                              self.model.wv.key_to_index.get(y, None)), 0) >= 1]
            if save_time == True:
                if len(res) > 50000:
                    return None
            t.append(time.time())
            l.append(len(res))

        #         for i in range(1, len(word_candidates)):
        #             res = [x+[y] for x in res for y in word_candidates[i] if
        #                    self.biword_counts.get((self.model.wv.key_to_index.get(x[-1], None),
        #                                            self.model.wv.key_to_index.get(y, None)), 0) >= 1]
        #             if save_time == True:
        #                 if len(res) > 1000000:
        #                     return None

        res.append(query_words)
        for i in range(1, len(t)):
            print(f'{i}: candidates: {len(word_candidates[i - 1]):0>5}, res length = {l[i - 1]:0>5}, time = {t[i] - t[i - 1]}')
        return res

    def merge_candidates(self, res, index):
        a = res.pop(index)
        b = res.pop(index)
        if not a or not b:
            res.insert(index, [])
        elif len(a[0]) == 1 and len(b[0]) == 1:
            res.insert(index, [x + y for x in a for y in b if
                               self.biword_counts.get((self.model.wv.key_to_index.get(x[0], None),
                                                       self.model.wv.key_to_index.get(y[0], None)), 0) >= 1])
        elif len(a[0]) == 1:
            res.insert(index, [x + y for x in a for y in b if
                               self.threeword_counts.get((self.model.wv.key_to_index.get(x[0], None),
                                                          self.model.wv.key_to_index.get(y[0], None),
                                                          self.model.wv.key_to_index.get(y[1], None)), 0) >= 1])
        elif len(b[0]) == 1:
            res.insert(index, [x + y for x in a for y in b if
                               self.threeword_counts.get((self.model.wv.key_to_index.get(x[-2], None),
                                                          self.model.wv.key_to_index.get(x[-1], None),
                                                          self.model.wv.key_to_index.get(y[0], None)), 0) >= 1])
        else:
            res.insert(index, [x + y for x in a for y in b if
                               self.threeword_counts.get((self.model.wv.key_to_index.get(x[-2], None),
                                                          self.model.wv.key_to_index.get(x[-1], None),
                                                          self.model.wv.key_to_index.get(y[0], None)), 0) >= 1 and
                               self.threeword_counts.get((self.model.wv.key_to_index.get(x[-1], None),
                                                          self.model.wv.key_to_index.get(y[0], None),
                                                          self.model.wv.key_to_index.get(y[1], None)), 0) >= 1])

    def cal_phrase_candidates2(self, word_candidates, query_words, save_time=False):
        if len(word_candidates) == 1:
            return [[x] for x in word_candidates[0]]
        res = [[[x] for x in y] for y in word_candidates]
        while len(res) > 1:
            multiply_cost = [len(word_candidates[i - 1]) * len(word_candidates[i]) for i in range(1, len(res))]
            index = np.argmin(multiply_cost)
            self.merge_candidates(res, index)
            if save_time == True:
                if len(res[index]) > 50000:
                    return None
        res[0].append(query_words)
        return res[0]

    def cal_word_candidates(self, query_words):
        res = []
        context = np.average(
            np.array([self.model.wv[word] for word in query_words if word in self.model.wv.key_to_index]), axis=0)
        #         context = np.average(np.array([self.model.wv[word] for word in query_words]), axis=0)
        context = np.expand_dims(context, axis=0)
        word_need_replacement = self.check_if_need_replacement(query_words, context)
        for i in range(len(query_words)):
            if word_need_replacement[i]:
                res.append([x for x in self.model.wv.key_to_index if
                            levenshtein(x, query_words[i], self.max_editdistance) <= self.max_editdistance and
                            self.word_counts.get((self.model.wv.key_to_index[x],), 0) > self.candidate_count_tresh])

            #                 res.append(sorted([x for x in self.model.wv.key_to_index if
            #                             levenshtein(x, query_words[i], self.max_editdistance) <= self.max_editdistance and
            #                             self.word_counts.get((self.model.wv.key_to_index[x], ), 1 ) > self.candidate_count_tresh],
            #                                  key=lambda x: self.model.wv.cosine_similarities(self.model.wv[x], context),  reverse=True)[:100])
            else:
                res.append([query_words[i]])

        return res

    def cal_candidates(self, query_words, save_time=False):
        #         t1 = time.time()
        word_candidates = self.cal_word_candidates(query_words)
        #         t2 = time.time()
        #         final_candidates = self.cal_phrase_candidates(word_candidates, query_words, save_time=save_time)
        final_candidates = self.cal_phrase_candidates2(word_candidates, query_words, save_time=save_time)
        #         t3 = time.time()
        #         print("time for generating word candidates:", t2-t1)
        #         print("time for generating phrase candidates:", t3-t2)
        return final_candidates

    def cal_score(self, query_words, proposed_words):
        S0_scores = []
        S1_scores = []
        S2_scores = []
        result_penalties = [self.cal_proposed_word_penalty(query_words[i], proposed_words[i]) for i in
                            range(len(query_words))]
        for i in range(len(proposed_words)):
            # OOV penalty?
            counts = self.word_counts.get((self.model.wv.key_to_index.get(proposed_words[i], None),), None)
            if counts is None:
                S0_scores.append(1 / self.word_counts["all"] * result_penalties[i])
            else:
                S0_scores.append(counts / self.word_counts["all"] * result_penalties[i])
            if i >= 1:
                key = (self.model.wv.key_to_index.get(proposed_words[i - 1], None),
                       self.model.wv.key_to_index.get(proposed_words[i], None))
                counts = self.biword_counts.get(key, 0)
                if counts == 0:
                    S1_scores.append(1 / self.biword_counts["all"] * result_penalties[i])
                else:
                    key = (self.model.wv.key_to_index.get(proposed_words[i - 1], None),)
                    S1_scores.append(counts / self.word_counts[key] * result_penalties[i])
            else:
                S1_scores.append(1 / self.biword_counts["all"] * result_penalties[i])

            if i >= 2:
                key = (self.model.wv.key_to_index.get(proposed_words[i - 2], None),
                       self.model.wv.key_to_index.get(proposed_words[i - 1], None),
                       self.model.wv.key_to_index.get(proposed_words[i], None))
                counts = self.threeword_counts.get(key, None)
                if counts is None:
                    S2_scores.append(1 / self.threeword_counts["all"] * result_penalties[i])
                else:
                    key2 = (self.model.wv.key_to_index.get(proposed_words[i - 2], None),
                            self.model.wv.key_to_index.get(proposed_words[i - 1], None))
                    S2_scores.append(counts / self.biword_counts[key2] * result_penalties[i])
            else:
                S2_scores.append(1 / self.threeword_counts["all"] * result_penalties[i])

        score = [
            np.dot(self.ngram_weights, np.array([S0_scores[i], S1_scores[i], S2_scores[i]])) / sum(self.ngram_weights)
            for i in range(len(S0_scores))]
        score = np.prod(score)
        return score

    @staticmethod
    def cal_scores(self, candidates, proposed_words, result):
        result.extend([self.cal_score(x, proposed_words) for x in candidates])

    def cal_scores_with_processes(self, candidates, proposed_words):
        rang = len(candidates) // 5 + 1
        # results = []
        results = manager.list()
        processes = [multiprocessing.Process(target=self.cal_scores,
                                             args=(self, candidates[rang * i:min(len(candidates), rang * (i + 1))],
                                                   proposed_words, results)) for i in range(len(results))]
        for process in processes:
            process.start()
        for process in processes:
            process.join()

        return functools.reduce(lambda a, b: a+b, results)

    def correct_answere(self, query, rank=-1, save_time=False, return_times=False):
        #         query_words = tokenizer.tokenize_words(normalizer.normalize(query))
        t1 = time.time()
        query_words = tokenizer.tokenize_words(query)
        candidates = self.cal_candidates(query_words, save_time=save_time)
        t2 = time.time()
        candidate_generation_time = t2 - t1
        if candidates is None:
            res = ""
            if return_times:
                return res, (candidate_generation_time, 0)
            return res

        scores = self.cal_scores_with_processes(candidates, query_words)
        results_and_scores = [(candidates[i], scores[i]) for i in range(len(scores))]
        results_and_scores = sorted(results_and_scores, key=lambda x: x[1], reverse=True)

        # results_and_scores = [(candidate, self.cal_score(query_words, candidate)) for candidate in candidates]
        # results_and_scores = sorted(results_and_scores, key=lambda x: x[1], reverse=True)

        t3 = time.time()
        scoring_candidates_time = t3 - t2

        #         print("time for generating candidates:", candidate_generation_time)
        #         print("time for scoring candidates:", scoring_candidates_time)
        #         print("number of candidates:", len(candidates))

        #         if len(results_and_scores) == 0:
        #             res = ""
        #             if return_times:
        #                 return res, (candidate_generation_time, scoring_candidates_time)
        #             return res

        if rank == -1:
            res = " ".join(results_and_scores[0][0])
            if return_times:
                return res, (candidate_generation_time, scoring_candidates_time)
            return res
        else:
            res = [(" ".join(results_and_scores[i][0]), results_and_scores[i][1]) for i in
                   range(min(rank, len(results_and_scores)))]
            if return_times:
                return res, (candidate_generation_time, scoring_candidates_time)
            return res


def test_spell_checker(spell_checker, df, save_time, return_times=False):
    corrects = 0
    empties = 1
    answeres = []
    times = []
    for i in range(len(df)):
        correct_word, misspelled_word = df.loc[i, ["correct word", "misspelled word"]]
        try:
            #             print(correct_word, "---", misspelled_word)
            corrected_word, check_time = spell_checker.correct_answere(misspelled_word, save_time=save_time,
                                                                       return_times=True)
            answeres.append(corrected_word)
            if corrected_word == correct_word:
                corrects += 1
            times.append(check_time)
            if corrected_word == "":
                empties += 1
        except Exception as e:
            print(correct_word, misspelled_word)
            print(misspelled_word)
            # print(len(correct_word), len(misspelled_word))
            # raise Exception()

    #         print("*****")

    if return_times:
        return corrects / len(df), corrects / (len(df) - empties), answeres, times
    return corrects / len(df), corrects / (len(df) - empties)


if __name__ == '__main__':
	manager = multiprocessing.Manager()
	t1 = time.time()
	word2vec_model = Word2Vec.load("Word2VecModel")
	with open('word counts.json', 'rb') as f:
		word_counts = pickle.load(f)
		f.close()
	with open('bi-grams.json', 'rb') as f:
		biword_counts = pickle.load(f)
		f.close()
	with open('three-grams.json', 'rb') as f:
		threeword_counts = pickle.load(f)
		f.close()
	with open('four-grams.json', 'rb') as f:
		fourword_counts = pickle.load(f)
		f.close()
	t2 = time.time()
	print("time to load model: ", t2-t1)
	SP = spellChecker(word2vec_model, word_counts, biword_counts, threeword_counts, fourword_counts,
                      candidate_count_tresh=100, candidate_sim_tresh=0.7, ngram_weights=[1, 10, 100],
                      max_editdistance=3, close_extra_char_penalty=0.1, far_extra_char_penalty=0.001,
                      missed_char_penalty=0.001, close_replace_char_penalty=0.1, far_replace_char_penalty=0.001)
	threegram_random_error_df = pd.read_csv("random threegram.csv", index_col=[0])
	threegram_random_error_sample_df = threegram_random_error_df.query("`total edit distance`==1").reset_index(drop=True)
	accuracy, non_empty_accuracy, answeres, times = test_spell_checker(SP, threegram_random_error_sample_df,
                                                                       save_time=True, return_times=True)
	print(accuracy, non_empty_accuracy)
	print("time to evaluate model: ", time.time()-t2)

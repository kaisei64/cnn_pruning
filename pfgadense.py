import numpy as np
import copy


class PfgaDense:
    def __init__(self, gene_len1, gene_len2, dense_num, evaluate_func=None, better_high=True, mutate_rate=0.05):
        self.family = []
        self.gene_len1 = gene_len1
        self.gene_len2 = gene_len2
        self.evaluate_func = evaluate_func
        self.better_high = better_high
        self.mutate_rate = mutate_rate
        self.generation_num = 0
        self.dense_num = dense_num

    def add_new_population(self):
        new_gene = []
        a = np.random.rand(self.gene_len1, self.gene_len2)
        a_max = np.max(a)
        a_min = np.min(a)
        y = 2 * (a - a_min) / (a_max - a_min) - 1
        new_gene.append(y)
        new_gene.append(None)
        self.family.append(new_gene)

    def pop_num(self):
        return len(self.family)

    def copy_gene(self, g):
        return copy.deepcopy(g)

    def best_gene(self):
        idx = self.family[0]
        for i in self.family:
            if i[1] is not None:
                if idx[1] is not None:
                    if (self.better_high is True and idx[1] < i[1]) or (self.better_high is False and idx[1] > i[1]):
                        idx = i
                else:
                    idx = i
        return idx

    def select_and_delete_gene(self):
        return self.copy_gene(self.family.pop(np.random.randint(0, len(self.family))))

    def crossover(self, p1, p2):
        c1 = self.copy_gene(p1)
        c2 = self.copy_gene(p2)
        for i in range(len(c1[0])):
            if np.random.rand() < 0.5:
                c1[0][i], c2[0][i] = c2[0][i], c1[0][i]
        c1[1] = None
        c2[1] = None
        return c1, c2

    def mutate(self, g):
        for i in range(len(g[0])):
            if np.random.rand() < self.mutate_rate:
                g[0][i] = np.random.rand()
        return g

    def next_generation(self):
        while len(self.family) < 2:
            self.add_new_population()

        p1 = self.select_and_delete_gene()
        p2 = self.select_and_delete_gene()

        c1, c2 = self.crossover(p1, p2)

        if np.random.rand() < 0.5:
            c1 = self.mutate(c1)
        else:
            c2 = self.mutate(c2)

        self.generation_num += 1

        genelist = p1, p2, c1, c2
        count = 0
        for i in genelist:
            # if i[1] is None:
            i[1] = self.evaluate_func(i[0], count, self.dense_num)
            count += 1

        # rule-1:both child is better than both parent, remain both child and better 1 parent
        if (self.better_high is True and min(c1[1], c2[1]) > max(p1[1], p2[1])) or (
                self.better_high is False and max(c1[1], c2[1]) < min(p1[1], p2[1])):
            self.family.append(c1)
            self.family.append(c2)
            if (self.better_high is True and p1[1] > p2[1]) or (self.better_high is False and p1[1] < p2[1]):
                self.family.append(p1)
            else:
                self.family.append(p2)

        # rule-2:both parent is better than both child, remain better 1 parent
        elif (self.better_high is True and max(c1[1], c2[1]) < min(p1[1], p2[1])) or (
                self.better_high is False and min(c1[1], c2[1]) > max(p1[1], p2[1])):
            if (self.better_high is True and p1[1] > p2[1]) or (self.better_high is False and p1[1] < p2[1]):
                self.family.append(p1)
            else:
                self.family.append(p2)

        # rule-3:better 1 parent is better than both child, remain better 1 parent and better 1 child
        elif (self.better_high is True and max(c1[1], c2[1]) < max(p1[1], p2[1])) or (
                self.better_high is False and min(c1[1], c2[1]) > min(p1[1], p2[1])):
            if (self.better_high is True and p1[1] > p2[1]) or (self.better_high is False and p1[1] < p2[1]):
                self.family.append(p1)
            else:
                self.family.append(p2)

            if (self.better_high is True and c1[1] > c2[1]) or (self.better_high is False and c1[1] < c2[1]):
                self.family.append(c1)
            else:
                self.family.append(c2)

        # rule-4:better 1 child is better than both parent, remain better 1 child and append 1 new gene
        else:
            if (self.better_high is True and c1[1] > c2[1]) or (self.better_high is False and c1[1] < c2[1]):
                self.family.append(c1)
            else:
                self.family.append(c2)
            self.add_new_population()

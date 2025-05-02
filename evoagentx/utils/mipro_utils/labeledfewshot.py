import random


class LabeledFewShot():
    def __init__(self, k=16):
        self.k = k

    def optimize(self, student, *, trainset, sample=True):
        self.student = student.reset_copy()
        self.trainset = trainset

        if len(self.trainset) == 0:
            return self.student

        rng = random.Random(0)

        for agent in self.student.agents():
            if sample:
                agent['demos'] = rng.sample(self.trainset, min(self.k, len(self.trainset)))
            else:
                agent['demos'] = self.trainset[: min(self.k, len(self.trainset))]

        return self.student


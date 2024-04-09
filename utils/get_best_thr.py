import math

def get_best_threshold(pos_prop, pos_scores, thr=0.5, tolerance=0.01):
    min = 0.0
    max = 1.0
    max_iteration = math.ceil(math.log2(len(pos_scores))) * 2 + 10
    for _ in range(max_iteration):
        new_proportion = sum(1 for score in pos_scores if score > thr) / len(pos_scores)
        if abs(new_proportion - pos_prop) < tolerance:
            return thr

        elif new_proportion > pos_prop:
            min = thr
            thr = (thr + max) / 2

        else:
            max = thr
            thr = (thr + min) / 2

    return thr
def precision(actual,recommended,K):
    if K<=0:
        raise ValueError("Invalid K Value")
        
    length = K
    count = 0
    for value in range(length):
        if recommended[value] in actual:
            count += 1

    return round(count/K,3)

def recall(actual,recommended,K):

    total_relevant_items = len(actual)
    relevant_items = 0
        
    for value in recommended[:K]:
        if value in actual:
            relevant_items += 1

    return round(relevant_items/total_relevant_items,3)

def f1_at_k(actual, recommended, K):

    precision_p = precision(actual,recommended,K)
    recall_r = recall(actual,recommended,K)

    numerator = 2*precision_p*recall_r

    denominator = precision_p+recall_r

    if denominator <=0:
        return 0

    return round(numerator/denominator,3)

def average_precision(actual,recommended):

    if len(actual)==0:
        return "Actual recommendation values are empty"

    relevant_hits,precision_sum = 0,0

    for position,item in enumerate(recommended,start=1):

        if item in actual:
            relevant_hits += 1
            precision_sum += precision(actual,recommended,position)

    if relevant_hits == 0:
        return "Recommended items are not present in the actual."

    return round(precision_sum/relevant_hits,3)

def mean_average_precision(ap_values):
    length = len(ap_values)
    if length <= 0:
        return "Average Precision values are empty."

    return round(sum(ap_values)/length,3)

print(mean_average_precision([0.756, 0.500, 1.000]))

    # for value in recommended:

    #     if value in actual:

    #         total_calculated_preci += 1
    #         precision_val = precision(actual,recommended,K)
    #         values_of_precision.append(precision_val)

    #     K += 1

    # if total_calculated_preci == 0:
    #     return "Recommed items are not present in the actual."

    # return round(sum(values_of_precision)/total_calculated_preci,3)

# actual = {"A", "B", "C"}

# recommended = ["A", "X", "B", "Y", "C"]

# K = 5

# # print(precision(actual,recommended,K))

# # print(recall(actual,recommended,K))

# # print(f1_at_k(actual,recommended,K))

# print(average_precision(actual,recommended))
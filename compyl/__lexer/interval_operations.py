from functools import cmp_to_key

# ======================================================================================================================
# Set operations on interval
# ======================================================================================================================


def interval_cmp(x, y):
    """
    Dictionary order comparator for intervals
    """
    if x[0] > y[0]:
        return 1
    elif x[0] < y[0]:
        return -1
    elif x[1] > y[1]:
        return 1
    elif x[1] < y[1]:
        return -1
    else:
        return 0


def value_interval_cmp(value, interval):
    """
    Comparator that indicates if a value is within, higher or lower than an interval
    """

    if interval[0] <= value <= interval[1]:
        return 0
    elif value < interval[0]:
        return -1
    else:
        return 1


def binary_search_on_transitions(target, transitions):
    """
    Binary search for a transition in a sorted list of (interval, transition).
    Return the found transition, None if not found
    """
    left = 0
    right = len(transitions) - 1

    while left <= right:
        index = (left + right) // 2
        min, max = transitions[index][0]

        if target < min:
            right = index - 1
        elif target > max:
            left = index + 1
        else:
            return transitions[index]

    return None


def binary_search_value_in_intervals(target, intervals, return_closest=False):
    """
    Binary search for a value in a sorted list of intervals.
    Return the position of the found interval, None if not found
    If return_closest is set to True, the index will be returned even if the target was found in no interval, this is
    needed if we want to use binary search to find a super-interval's boundaries.
    """
    left = 0
    right = len(intervals) - 1
    index = None

    while left <= right:
        index = (left + right) // 2
        min, max = intervals[index]

        if target < min:
            right = index - 1
        elif target > max:
            left = index + 1
        else:
            return index

    return index if return_closest else None


def value_is_in_range(value, range):
    """
    :param value: an int (x)
    :param range: a tuple of int (min, max)
    :return: True if x from min to max, False otherwise
    """
    return range[0] <= value <= range[1]


def is_proper_subinterval(subinterval, interval):
    """
    Return True if subinterval (x_1, x_2) is a subset of interval (y_1, y_2), else False
    """
    return value_is_in_range(subinterval[0], interval) and value_is_in_range(subinterval[1], interval)


def merge_intervals(intervals):
    """
    Given a list of intervals, return a new list of intervals where the adjacent intervals are merged.
    Ex: [(1,3), (4,6), (10,11)] would be returned as [(1, 6), (10, 11)]
    """
    if not intervals:
        return []

    intervals.sort(key=cmp_to_key(interval_cmp))

    merged_intervals = []
    min = max = intervals[0][0]

    for interval in intervals:
        if interval[0] > max + 1:
            merged_intervals.append((min, max))
            min = interval[0]

        max = interval[1]

    merged_intervals.append((min, max))

    return merged_intervals


def set_to_intervals(ascii_set):
    """
    Given a set of int, return a list of intervals covering those ints exactly
    ex: the set {1,2,3,5,6,9} would be returned as [(1,3), (5,6), (9,9)]
    """

    set_size = len(ascii_set)

    if set_size == 0:
        return []

    else:

        ascii_list = list(ascii_set)
        ascii_list.sort()

        # Hack so the last interval in the list is added
        ascii_list.append(float('inf'))

        interval_list = []

        min = max = ascii_list[0]

        index = 1
        while index <= set_size:
            ascii = ascii_list[index]
            if ascii == max + 1:
                max += 1
            else:
                interval_list.append((min, max))
                min = max = ascii

            index += 1

        return interval_list


def get_minimal_covering_intervals(intervals):
    """
    Given a list of intervals (min_int, max_int) which might overlap, return a minimal covering set of intervals
    The Minimal Covering Set of Interval (MCSI) is the set of intervals that has the following properties:

    1) The MCSI forms a disjoint partition of the union of the initial set
       i.e. they represent the same overall values, but the MSCI doesn't allow overlaps anymore

    2) For each interval A in MCSI and each interval B in the initial set of intervals, then either
       i) A is a proper subset of B (A and B = A)
       ii) or A and B are disjoint

    3) The MCSI is the smallest set such that the two above rules are respected
       It doesn't mean it is unique, solely that any other such set has the same cardinality

    Ex: Given the set {(1,5), (3, 7), (9,34), (15,15)}, we would return
        {(1,2), (3,5), (6,7), (8,14), (15, 15), (16, 34)}

    This is a way here to partition the intervals of ascii values in the lookouts of an FDA to get the lookout values
    of a DFA
    """

    def rec(intervals):
        if not intervals:
            return []

        if len(intervals) == 1:
            return intervals

        intervals.sort(key=cmp_to_key(interval_cmp))

        left = intervals[0][0]
        right = intervals[0][1]

        for el in intervals:
            if el[0] == left:
                continue
            elif el[0] <= right:
                right = el[0] - 1
                break
            else:
                break

        # Remove intervals below (right + 1) and truncate others such that their minimum > max
        truncated_intervals = [(right + 1, el[1]) if el[0] <= right else el for el in intervals if el[1] > right]

        return [(left, right)] + rec(truncated_intervals)

    return rec(intervals)


def inverse_intervals_list(intervals):
    """
    Given a list of intervals, return the inverse of the union of the intervals
    """

    inverse = [(0, 255)]

    for interval in intervals:
        min, max = interval

        if min > max:
            raise ValueError("the lower bound of an interval must be leq than the upper bound")

        min_pos = binary_search_value_in_intervals(min, inverse, return_closest=True)

        # min_is_inside indicates if the min is contained in an interval
        # 0  => the min is contained in the interval at min_pos
        # 1  => the min is above the interval at min_pos
        # -1 => the min is below the interval at min_pos
        min_is_inside = value_interval_cmp(min, inverse[min_pos])

        max_pos = binary_search_value_in_intervals(max, inverse, return_closest=True)

        # Idem as min_is_inside, but for the max
        max_is_inside = value_interval_cmp(max, inverse[max_pos])

        # Case where we truncate the interval
        if min_is_inside == 0:
            matching_interval = inverse[min_pos]

            left = inverse[:min_pos]

            # Append the truncated interval if there is something remaining
            if matching_interval[0] < min:
                left.append((matching_interval[0], min - 1))

        # min is below min_pos, thus keep everything below min_pos, min_pos excluded
        elif min_is_inside == -1:
            left = inverse[:min_pos]

        # min is above min_pos, thus keep everything below min_pos, min_pos included
        elif min_is_inside == 1:
            try:
                _ = inverse[min_pos + 1]
            except IndexError:
                # Case where the min is above, and there is nothing above, thus inverse is not mutated
                continue
            left = inverse[:min_pos + 1]

        # Same logic as above but for the maximum
        if max_is_inside == 0:
            matching_interval = inverse[max_pos]

            right = inverse[max_pos + 1:]

            if matching_interval[1] > max:
                right.insert(0, (max + 1, matching_interval[1]))

        elif max_is_inside == -1:
            right = inverse[max_pos:]

        elif min_is_inside == 1:
            right = inverse[max_pos + 1:]

        inverse = left + right

    return inverse
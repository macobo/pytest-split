import enum
import functools
import heapq
from typing import TYPE_CHECKING, NamedTuple
from collections import defaultdict

if TYPE_CHECKING:
    from typing import Dict, List, Tuple

    from _pytest import nodes


class InputTestGroup(NamedTuple):
    tests: "List[nodes.Item]"
    duration: float


class OutputTestGroup(NamedTuple):
    selected: "List[nodes.Item]"
    deselected: "List[nodes.Item]"
    duration: float


def least_duration(
    splits: int, input_test_groups: "List[InputTestGroup]"
) -> "List[OutputTestGroup]":
    """
    Split tests into groups by runtime.
    It walks the test items, starting with the test with largest duration.
    It assigns the test with the largest runtime to the group with the smallest duration sum.

    The algorithm sorts the items by their duration. Since the sorting algorithm is stable, ties will be broken by
    maintaining the original order of items. It is therefore important that the order of items be identical on all nodes
    that use this plugin. Due to issue #25 this might not always be the case.

    :param splits: How many groups we're splitting in.
    :param items: Test items passed down by Pytest.
    :param durations: Our cached test runtimes. Assumes contains timings only of relevant tests
    :return:
        List of groups
    """

    # sort in ascending order
    sorted_items_with_durations = sorted(
        enumerate(input_test_groups), key=lambda index_and_group: index_and_group[1].duration, reverse=True
    )

    selected: "List[List[Tuple[InputTestGroup, int]]]" = [[] for i in range(splits)]
    deselected: "List[List[InputTestGroup]]" = [[] for i in range(splits)]
    duration: "List[float]" = [0 for i in range(splits)]

    # create a heap of the form (summed_durations, group_index)
    heap: "List[Tuple[float, int]]" = [(0, i) for i in range(splits)]
    heapq.heapify(heap)
    for original_index, input_test_group in sorted_items_with_durations:
        # get group with smallest sum
        summed_durations, group_idx = heapq.heappop(heap)
        new_group_durations = summed_durations + input_test_group.duration

        # store assignment
        selected[group_idx].append((input_test_group, original_index))
        duration[group_idx] = new_group_durations
        for i in range(splits):
            if i != group_idx:
                deselected[i].append(input_test_group)

        # store new duration - in case of ties it sorts by the group_idx
        heapq.heappush(heap, (new_group_durations, group_idx))

    groups = []
    for i in range(splits):
        # sort the items by their original index to maintain relative ordering
        # we don't care about the order of deselected items
        s = [
            input_test_group for input_test_group, original_index in sorted(selected[i], key=lambda tup: tup[1])
        ]
        group = OutputTestGroup(selected=_flatten(s), deselected=_flatten(deselected[i]), duration=duration[i])
        groups.append(group)
    return groups


def duration_based_chunks(
    splits: int, input_test_groups: "List[InputTestGroup]"
) -> "List[OutputTestGroup]":
    """
    Split tests into groups by runtime.
    Ensures tests are split into non-overlapping groups.
    The original list of test items is split into groups by finding boundary indices i_0, i_1, i_2
    and creating group_1 = items[0:i_0], group_2 = items[i_0, i_1], group_3 = items[i_1, i_2], ...

    :param splits: How many groups we're splitting in.
    :param items: Test items passed down by Pytest.
    :param durations: Our cached test runtimes. Assumes contains timings only of relevant tests
    :return: List of OutputTestGroup
    """
    time_per_group = sum(group.duration for group in input_test_groups) / splits

    selected: "List[List[InputTestGroup]]" = [[] for i in range(splits)]
    deselected: "List[List[InputTestGroup]]" = [[] for i in range(splits)]
    duration: "List[float]" = [0 for i in range(splits)]

    group_idx = 0
    for group in input_test_groups:
        if duration[group_idx] >= time_per_group:
            group_idx += 1

        selected[group_idx].append(group)
        for i in range(splits):
            if i != group_idx:
                deselected[i].append(group)
        duration[group_idx] += group.duration

    return [
        OutputTestGroup(selected=_flatten(selected[i]), deselected=_flatten(deselected[i]), duration=duration[i])
        for i in range(splits)
    ]

def group_tests(group_type: str, items: "List[nodes.item]", durations: "Dict[str, float]") -> "List[InputTestGroup]":
    items_with_durations = _get_items_with_durations(items, durations)
    if group_type == "test":
        return [InputTestGroup([item], duration) for item, duration in items_with_durations]
    else:
        items_by_file = defaultdict(list)
        durations_by_file = defaultdict(int)
        for item, duration in items_with_durations:
            file, _ = item.nodeid.split('::')
            items_by_file[file].append(item)
            durations_by_file[file] += duration

        return [InputTestGroup(items_by_file[file], durations_by_file[file]) for file in items_by_file.keys()]


def _get_items_with_durations(
    items: "List[nodes.Item]", durations: "Dict[str, float]"
) -> "List[Tuple[nodes.Item, float]]":
    durations = _remove_irrelevant_durations(items, durations)
    avg_duration_per_test = _get_avg_duration_per_test(durations)
    items_with_durations = [
        (item, durations.get(item.nodeid, avg_duration_per_test)) for item in items
    ]
    return items_with_durations


def _get_avg_duration_per_test(durations: "Dict[str, float]") -> float:
    if durations:
        avg_duration_per_test = sum(durations.values()) / len(durations)
    else:
        # If there are no durations, give every test the same arbitrary value
        avg_duration_per_test = 1
    return avg_duration_per_test


def _remove_irrelevant_durations(
    items: "List[nodes.Item]", durations: "Dict[str, float]"
) -> "Dict[str, float]":
    # Filtering down durations to relevant ones ensures the avg isn't skewed by irrelevant data
    test_ids = [item.nodeid for item in items]
    durations = {name: durations[name] for name in test_ids if name in durations}
    return durations

def _flatten(test_groups: "List[InputTestGroup]") -> "List[nodes.Item]":
    return [item for group in test_groups for item in group.tests]


class Algorithms(enum.Enum):
    # values have to wrapped inside functools to avoid them being considered method definitions
    duration_based_chunks = functools.partial(duration_based_chunks)
    least_duration = functools.partial(least_duration)

    @staticmethod
    def names() -> "List[str]":
        return [x.name for x in Algorithms]

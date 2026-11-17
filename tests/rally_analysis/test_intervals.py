from rally_analysis.intervals import LabeledSpan, keep_spans, states_in


def test_subtracts_excluded_and_keeps_gaps():
    # reviewed [0, 100), no-play at [10, 20) and [50, 60)
    assert keep_spans((0, 100), [(10, 20), (50, 60)], min_len=1) == ((0, 10), (20, 50), (60, 100))


def test_drops_short_pieces():
    assert keep_spans((0, 100), [(5, 50)], min_len=10) == ((50, 100),)


def test_excluded_outside_range_is_ignored_and_clamped():
    assert keep_spans((10, 50), [(0, 15), (45, 99)], min_len=1) == ((15, 45),)


def test_overlapping_excluded_spans():
    assert keep_spans((0, 100), [(10, 40), (30, 60)], min_len=1) == ((0, 10), (60, 100))


def test_states_in_orders_by_time_and_dedups():
    spans = [LabeledSpan(30, 50, "play"), LabeledSpan(0, 30, "service"), LabeledSpan(40, 45, "play"),
             LabeledSpan(90, 99, "service")]
    assert states_in((0, 60), spans) == ("service", "play")


def test_states_in_unlabelled_gap():
    assert states_in((0, 10), []) == ("unlabelled",)

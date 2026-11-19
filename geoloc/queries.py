"""Heading and scale queries: the list of (heading, scale factor) drawings that one photo goes through."""
import settings


ALL_HEADINGS = [-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0]
SETS = {
    "standard": ([-5.0, 0.0, 5.0], [1.0, 1.05]),
    "all": (ALL_HEADINGS, [1.0, 1.05]),
}


def headings_and_scales():
    if settings.QUERY_SET == "custom":
        return list(settings.HEADINGS), list(settings.SCALE_FACTORS)
    return SETS[settings.QUERY_SET]


def query_list():
    """One dict per combination. The plain query (heading 0, factor 1) comes first, so if time ever runs short it is the one that is done."""
    queries = [dict(heading=0.0, scale_factor=1.0)]
    headings, factors = headings_and_scales()
    for factor in factors:
        for heading in headings:
            if heading == 0.0 and factor == 1.0:
                continue
            queries.append(dict(heading=float(heading), scale_factor=float(factor)))
    return queries


def second_pass_queries():
    """The queries of the second pass that the first pass did not run. Frozen default: the "all" set (9 headings x scales 1.0, 1.05) minus the standard set.
    SECOND_PASS_HEADINGS / SECOND_PASS_SCALES (None = the default above) change the grid, for example wider scales."""
    if not settings.SECOND_PASS or settings.QUERY_SET != "standard":
        return []
    first = SETS["standard"]
    first_keys = set()
    for factor in first[1]:
        for heading in first[0]:
            first_keys.add((heading, factor))
    headings = settings.SECOND_PASS_HEADINGS if settings.SECOND_PASS_HEADINGS is not None else SETS["all"][0]
    factors = settings.SECOND_PASS_SCALES if settings.SECOND_PASS_SCALES is not None else SETS["all"][1]
    extra = []
    for factor in factors:
        for heading in headings:
            if (float(heading), float(factor)) not in first_keys:
                extra.append(dict(heading=float(heading), scale_factor=float(factor)))
    return extra

import re


def split_string(s):
    """Split a trailing underscore-index from a parameter name.

    Parameters
    ----------
    s : str
        Name optionally ending in ``_<integer>``, e.g. ``"f_SiIII_2"``.

    Returns
    -------
    tuple of str and str or None
        Base name and trailing index when present, e.g. ``("f_SiIII", "2")``;
        ``(s, None)`` otherwise.
    """
    match = re.match(r"^(.*)_(\d+)$", s)
    if match:
        return match.group(1), match.group(2)
    return s, None

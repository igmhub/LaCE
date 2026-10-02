def split_string(string):
    """Split a string on underscore separators.

    Parameters
    ----------
    string : str
        Input text.

    Returns
    -------
    list of str
        Underscore-delimited components, including empty components.
    """
    parts = string.split("_")
    return parts

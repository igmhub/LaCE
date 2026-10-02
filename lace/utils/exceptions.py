class ExceptionList(Exception):
    """Exception that renders a message followed by permitted values."""
    def __init__(self, msg, values):
        """Build an error message listing permitted string values.

        Parameters
        ----------
        msg : str
            Error-message prefix.
        values : sequence of str
            Values appended as a comma-separated allowed list.
        """
        self.message = msg
        for ii, val in enumerate(values):
            self.message += " " + val
            if ii < len(values) - 1:
                self.message += ","

    def __str__(self):
        """Return the rendered exception message."""
        return self.message

class ExceptionList(Exception):
    def __init__(self, msg, values):
        self.message = msg
        for ii, val in enumerate(values):
            self.message += " " + val
            if ii < len(values) - 1:
                self.message += ","

    def __str__(self):
        return self.message

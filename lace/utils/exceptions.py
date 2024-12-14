class ExceptionList(Exception):
    def __init__(self, msg, values):
        self.message = msg
        values = [str(val) for val in values]  
        self.message += " " + ", ".join(values)

    def __str__(self):
        return self.message

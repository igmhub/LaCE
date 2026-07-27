from datetime import datetime


def version_scheme(version):
    dt = version.time
    return f"{dt.year}.{dt.month}.{dt.day}"

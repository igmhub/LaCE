from setuptools import setup


def version_scheme(version):
    if version.exact and version.tag:
        return str(version.tag)
    return f"0.0.dev{version.distance or 0}"


def local_scheme(version):
    if version.exact:
        return ""
    node = (version.node or "unknown").removeprefix("g")
    return f"+g{node}" + (".dirty" if version.dirty else "")


setup(
    use_scm_version={
        "version_file": "lace/_version.py",
        "fallback_version": "0+unknown",
        "version_scheme": version_scheme,
        "local_scheme": local_scheme,
    }
)

from lace.archive.gadget_archive import GadgetArchive
def open_gadget_archive():
    cabayol23_archive = GadgetArchive(postproc='Cabayol23')
    assert len(cabayol23_archive) == 12210
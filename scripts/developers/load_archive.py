from lace.archive.gadget_archive import GadgetArchive


def main():
    print("Testing loading archive")
    cabayol23_archive = GadgetArchive(postproc="Cabayol23")
    print(cabayol23_archive.list_sim)
    print("Done!")


if __name__ == "__main__":
    main()

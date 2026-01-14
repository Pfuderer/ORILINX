import sys

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    # Delegate to the predict implementation (keeps parsing & help in one place)
    from .predict import main as predict_main
    return predict_main(argv)


if __name__ == "__main__":
    main()

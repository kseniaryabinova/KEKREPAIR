import argparse


def parse_arguments():
    """Just parse command line arguments."""
    parser = argparse.ArgumentParser(description='Apartment repair recognizer')
    parser.add_argument(
        '--c',
        '-config',
        type=str,
        required=True,
        help='Path for training configuration .yaml file.'
    )
    return parser.parse_args()
import argparse


def preload(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--ad-custom-no-huggingface",
        action="store_true",
        help="Don't use ADetailer models from huggingface",
    )

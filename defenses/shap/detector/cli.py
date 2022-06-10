import logging
import argparse

logger = logging.getLogger("attackdetect")
logger.setLevel(logging.INFO)


def main():
    """
    Main function of the program, parses the arguments from the CLI and passes them to an execution function.
    """
    args = parse_args()
    execute(args)


def parse_args() -> argparse.Namespace:
    """
    Function to parse arguments from the CLI given an input from the user.

    :return: Key-value object containing the parsed arguments.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description="AttackDetect")
    parser.add_argument("--shap_files_path")

    params = parser.parse_args()

    return params


def execute(params: argparse.Namespace):
    """
    Function to trigger the execution of the program based on input provided via the CLI.

    :param args: User input from the CLI.
    :type args: argparse.Namespace
    """
    logger.info(f"Starting AttactDetect")

    from detector import main

    main(params=params)


main()

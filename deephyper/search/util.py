import sys

bcolors = {
    "HEADER": "\033[95m",
    "OKBLUE": "\033[94m",
    "OKGREEN": "\033[92m",
    "WARNING": "\033[93m",
    "FAIL": "\033[91m",
    "ENDC": "\033[0m",
    "BOLD": "\033[1m",
    "UNDERLINE": "\033[4m",
}


def banner(message, color="HEADER"):
    """Print a banner with message

    Args:
        message (str): The message to be printed
        color (str, optional): The color of the banner in bcolors. Defaults to "HEADER".
    """

    header = "*" * (len(message) + 4)
    msg = f" {header}\n   {message}\n {header}"
    if sys.stdout.isatty():
        print(bcolors.get(color), msg, bcolors["ENDC"], sep="")
    else:
        print(msg)

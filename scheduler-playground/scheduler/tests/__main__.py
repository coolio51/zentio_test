from tests.test_1 import test_1

from rich.console import Console
from scheduler.common.console import get_console


if __name__ == "__main__":
    console = get_console()
    test_1(console)


from pathlib import Path
import time
import fcntl

class ProductionLogger:
    def __init__(
        self, log_file: str| Path, silent=False, master: bool = False, hostname: str = "", ip_address: str = ""
    ) -> None:
        self.log_file = log_file
        self.silent = silent
        self.master = master
        self.hostname = hostname
        self.ip_address = ip_address

        if not Path(self.log_file).exists():
            with open(self.log_file, "w") as f:
                if self.master:
                    f.write("timestamp, hostname, ip_address, event, status, duration, message\n")
                else:
                    f.write("timestamp, event, status, duration, message\n")
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}, INIT, SUCCESS, 0, `Logger initialized`\n")
        # if overwrite or (not Path(self.log_file).exists()):
        #     with open(self.log_file, "w") as f:
        #         f.write("timestamp, event, status, duration, message\n")

    def log(self, event: str, status: str, duration: float, message: str = "") -> None:
        with open(self.log_file, "a") as f:
            if self.master:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.write(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')}, {self.hostname}, {self.ip_address}, {event}, {status}, {duration}, `{message}`\n"
                )
            else:
                f.write(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')}, {event}, {status}, {duration}, `{message}`\n"
                )
        if not self.silent:
            print(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}: {event=} | {status=} | {duration=:.0f}s | {message}"
            )

    def message(self, message: str) -> None:
        with open(self.log_file, "a") as f:
            if self.master:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}, {self.hostname}, {self.ip_address}, MESSAGE, , , `{message}`\n")
            else:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}, MESSAGE, , , `{message}`\n")
        if not self.silent:
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: {message}")
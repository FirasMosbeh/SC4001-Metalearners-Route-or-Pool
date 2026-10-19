import time


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def format_duration(seconds):
    hours, rest = divmod(int(seconds), 3600)
    return f"{hours}h{rest // 60:02d}m{rest % 60:02d}s"


class Progress:
    """Counts finished items and extrapolates the remaining time linearly."""

    def __init__(self, total):
        self.total = total
        self.done = 0
        self.start = time.time()

    def step(self):
        self.done += 1

    def __str__(self):
        elapsed = time.time() - self.start
        if self.done == 0:
            return f"0/{self.total} images, elapsed {format_duration(elapsed)}, ETA unknown"
        percent = 100 * self.done / max(self.total, 1)
        remaining = elapsed / self.done * (self.total - self.done)
        return (f"{self.done}/{self.total} images ({percent:.1f}%), "
                f"elapsed {format_duration(elapsed)}, ETA {format_duration(remaining)}")

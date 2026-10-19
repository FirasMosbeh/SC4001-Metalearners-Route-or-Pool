import json

SHARD_LOG_FILE = "shard_log.json"  # inside the dataset folder


class ShardLog:
    """Remembers which train shards pass 1 already visited and which labels they contain.

    Saved after every shard, so a restarted run skips shards it already visited and
    the top-up still knows every shard that holds a label.
    """

    def __init__(self, dataset_dir):
        self.path = dataset_dir / SHARD_LOG_FILE
        self.labels = {}  # shard file -> list of labels in it
        if self.path.exists():
            self.labels = json.loads(self.path.read_text())

    def record(self, shard_file, labels):
        self.labels[shard_file] = sorted(labels)
        # Write to a temporary file first, so a crash mid-write can't corrupt the log.
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.labels, indent=1))
        temporary.replace(self.path)

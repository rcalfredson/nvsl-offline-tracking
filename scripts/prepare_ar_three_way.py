"""Build native analysis inputs from the four completed AR tracking batches."""

import argparse
import json
from pathlib import Path

from nvsl_offline.recording import identity, json_write


def completed_jobs(status_path):
    status = json.loads(status_path.read_text())
    jobs = status["jobs"]
    by_video = {job["video"]: job for job in jobs}
    if (not status["complete"] or not jobs or len(by_video) != len(jobs)
            or any(job["stage"] != "complete" for job in jobs)):
        raise ValueError(f"Expected a complete batch with unique recordings: {status_path}")
    return by_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    recordings = {}
    selections = {}
    for cohort in ("agarose", "flat"):
        old, new = [
            completed_jobs(args.runs / f"{cohort}-ar-control-{variant}" / "status.json")
            for variant in ("body-core", "two-rule")
        ]
        if old.keys() != new.keys():
            raise ValueError(f"Recording lists differ between batches for {cohort}")
        selection = selections[cohort] = []
        for video in old:
            canonical = str(Path(video).resolve(strict=True))
            if canonical in recordings or any(char in canonical for char in ",\r\n"):
                raise ValueError(f"Duplicate or invalid video path: {canonical}")
            variants = {}
            for name, job in (("nov17-body-core", old[video]), ("nov20-two-rule", new[video])):
                path = Path(job["companion"]).resolve(strict=True)
                if identity(path, content_hash=True) != job["companion_identity"]:
                    raise ValueError(f"Companion changed since the completed batch: {path}")
                variants[name] = str(path)
            recordings[canonical] = variants
            selection.append(canonical)

    args.output.mkdir(parents=True, exist_ok=False)
    json_write(args.output / "tracking.json", dict(schema_version=1, recordings=recordings))
    for cohort, selection in selections.items():
        (args.output / f"{cohort}-videos.txt").write_text(",".join(selection) + "\n")
        print(f"{cohort}: {len(selection)} recordings with two verified companions each")


if __name__ == "__main__":
    main()

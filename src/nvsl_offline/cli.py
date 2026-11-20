"""Command-line entry point. Reference code and template assets are never imported."""

import argparse
import json

import cv2

from .background import BackgroundConfig
from .detection import DetectionConfig
from .pipeline import audit, inspect, track


def parser():
    p = argparse.ArgumentParser(description="Independent offline fly detection baseline")
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("audit", help="Fully decode, validate alignment, and estimate backgrounds")
    a.add_argument("video")
    a.add_argument("--output", required=True, help="New audit directory (must not already exist)")
    a.add_argument("--layout", choices=["large", "large2"], required=True)
    a.add_argument("--sidewall-margin", type=int, default=24, help="Saved-image pixels beyond floor")
    a.add_argument("--samples", type=int, default=129)
    a.add_argument("--quantile", type=float, default=.8)
    a.add_argument("--noise-floor", type=float, default=3)
    a.add_argument("--noise-smoothing", type=float, default=3)
    a.add_argument("--hash-video", action="store_true", help="Read video bytes again to record SHA-256")
    a.add_argument("--trx", help="Explicit original trajectory path when sidecar discovery is ambiguous")
    a.add_argument("--data", help="Explicit original experiment metadata path")
    i = sub.add_parser("inspect", help="Review exact saved-frame candidates and export blank annotation worksheet")
    i.add_argument("run", help="Completed audit directory")
    i.add_argument("--output", required=True, help="New review directory")
    i.add_argument("--frames", required=True, help="Comma-separated zero-based saved-frame indices")
    i.add_argument("--mode", choices=["absolute", "normalized", "both"], default="both")
    t = sub.add_parser("track", help="Write experimental full-length companion; never interpolates")
    t.add_argument("run", help="Completed audit directory")
    t.add_argument("--output", required=True, help="New companion .trx file")
    t.add_argument("--mode", choices=["absolute", "normalized"], default="absolute")
    t.add_argument("--exclude-slots", default="", help="Comma-separated physical slots to leave missing")
    r = sub.add_parser("compare", help="Synchronized original/offline ellipse player or comparison export")
    r.add_argument("companion", help="Selected offline .trx; recording sources are inferred from its metadata")
    r.add_argument("--video", help="Video path (required if companion has no source metadata)")
    r.add_argument("--original", help="Original acquisition .trx path")
    r.add_argument("--data", help="Original experiment .data path")
    r.add_argument("--layout", choices=["large", "large2"], help="Layout for chamber crops if not saved in companion")
    r.add_argument("--label", help="Display label for the selected variant")
    r.add_argument("--start", default="0", help="Saved frame or MM:SS / HH:MM:SS[.fraction], inclusive")
    r.add_argument("--end", help="Saved frame or clock time, exclusive")
    r.add_argument("--fly", type=int, help="Physical fly slot for a fixed chamber crop (default: all)")
    r.add_argument("--scale", type=float, help="Magnification (default 2 for one chamber, 1 for full frame)")
    r.add_argument("--speed", type=float, default=1., help="Playback/export speed multiplier")
    r.add_argument("--autoplay", action="store_true", help="Start playing immediately (default: paused)")
    r.add_argument("--output", help="Export interval to a new .mp4 instead of opening the player")
    r.add_argument("--snapshot", help="Save start frame to a new .png instead of opening the player")
    r.add_argument("--snapshot-dir", default="runs/comparison-snapshots", help="Directory for the player's s key")
    for command in (i, t):
        command.add_argument("--segmentation", choices=["components", "body_core"], default="components",
                             help="Foreground components or experimental body-core reconstruction")
        command.add_argument("--core-radius", type=float, default=2., help="Core distance threshold in saved-image pixels")
        command.add_argument("--core-growth", type=int, default=2, help="Bounded 8-neighbor expansion steps")
        command.add_argument("--min-concentration", type=float,
                             help="Optional minimum fraction of body-mask weight in one disk (body_core)")
        command.add_argument("--concentration-radius", type=int, default=10,
                             help="Disk radius in saved-image pixels (default: 10)")
        command.add_argument("--ambiguity-darkness-ratio", type=float,
                             help="Optional lead-to-rival median residual ratio that resolves ambiguity (body_core)")
        command.add_argument("--illumination", choices=["none", "global_offset"], default="none",
                             help="Optional per-frame additive background correction (default: none)")
        command.add_argument("--threshold", type=float, default=25)
        command.add_argument("--signal-floor", type=float, default=6)
        command.add_argument("--score-threshold", type=float, default=4)
        command.add_argument("--area-min", type=float, default=40)
        command.add_argument("--area-max", type=float, default=1000)
        command.add_argument("--ambiguity-ratio", type=float, default=.5)
        command.add_argument("--largest-only", action="store_true", help="Disable ambiguity rejection for a control experiment")
        command.add_argument("--allow-truncated", action="store_true", help="Allow contours touching chamber ROI border")
        command.add_argument("--shape-fit", choices=["box", "weighted_moments"], default="weighted_moments",
                             help="Body geometry only (default: weighted_moments); centers and missingness are preserved")
    return p


def integer_list(text):
    return [int(v.strip()) for v in text.split(",") if v.strip()]


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        if args.command == "compare":
            from .replay import compare
            settings = vars(args).copy()
            settings.pop("command")
            print(json.dumps(compare(**settings), indent=2))
            return
        if args.command == "audit":
            result = audit(args.video, args.output, layout=args.layout, margin=args.sidewall_margin,
                           samples=args.samples, config=BackgroundConfig(args.quantile, args.noise_floor,
                           args.noise_smoothing), hash_video=args.hash_video, trx=args.trx, data=args.data)
            print(json.dumps(dict(output=args.output, decoded_frames=result["decoded_frames"],
                                  decode_seconds=result["decode_seconds"]), indent=2))
            return
        modes = ["absolute", "normalized"] if args.mode == "both" else [args.mode]
        configs = [DetectionConfig(mode=mode, threshold=args.threshold, signal_floor=args.signal_floor,
                   score_threshold=args.score_threshold, area_min=args.area_min, area_max=args.area_max,
                   ambiguity_ratio=args.ambiguity_ratio, reject_ambiguous=not args.largest_only,
                   reject_truncated=not args.allow_truncated, shape_fit=args.shape_fit,
                   illumination=args.illumination, segmentation=args.segmentation,
                   core_radius=args.core_radius, core_growth=args.core_growth,
                   min_concentration=args.min_concentration, concentration_radius=args.concentration_radius,
                   ambiguity_darkness_ratio=args.ambiguity_darkness_ratio) for mode in modes]
        if args.command == "inspect":
            result = inspect(args.run, args.output, indices=integer_list(args.frames), configs=configs)
        else:
            result = track(args.run, args.output, config=configs[0], exclude_slots=integer_list(args.exclude_slots))
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, cv2.error) as exc:
        p.exit(2, f"error: {exc}\n")

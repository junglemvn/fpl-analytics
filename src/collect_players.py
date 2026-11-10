"""Collect all current player season totals."""

# Shared collection functions
if __package__:
    from .collect_fpl import collect_snapshot, parse_snapshot_arguments
else:
    from collect_fpl import collect_snapshot, parse_snapshot_arguments


def main():
    """Save historical-format player tables."""
    args = parse_snapshot_arguments("players")  # Terminal settings
    result = collect_snapshot("players", args.season, args.from_batch)  # Download and export
    return result  # Exit code


if __name__ == "__main__":
    raise SystemExit(main())

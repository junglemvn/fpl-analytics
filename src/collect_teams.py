"""Collect all current FPL team records."""

# Shared collection functions
if __package__:
    from .collect_fpl import collect_snapshot, parse_snapshot_arguments
else:
    from collect_fpl import collect_snapshot, parse_snapshot_arguments


def main():
    """Save the native team table."""
    args = parse_snapshot_arguments("teams")  # Terminal settings
    result = collect_snapshot("teams", args.season, args.from_batch)  # Download and export
    return result  # Exit code


if __name__ == "__main__":
    raise SystemExit(main())

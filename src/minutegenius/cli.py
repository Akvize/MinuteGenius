"""
Unified entry point: minutegenius {check,meeting,transcribe,vtt} ...

Each subcommand delegates directly to the argparse of the matching module
(meeting.py, transcribe.py, vtt.py), which can still be used on its own.
"""
import sys

USAGE = """usage: minutegenius <command> [options]

Commands:
  check        Check configuration and connectivity (AI_API, pyannote.ai...)
  meeting      Generate smart minutes from a meeting video
  transcribe   Transcribe + diarize an audio/video file (without the full pipeline)
  vtt          Convert a WebVTT subtitle file to Markdown

Run 'minutegenius <command> --help' for command-specific options.
"""

COMMANDS = {"check", "meeting", "transcribe", "vtt"}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(USAGE)
        sys.exit(0 if len(sys.argv) >= 2 else 1)

    command, rest = sys.argv[1], sys.argv[2:]
    if command not in COMMANDS:
        print(f"Error: unknown command '{command}'.\n", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        sys.exit(1)

    # Each submodule parses its own sys.argv — rebuild it so its
    # usage/error messages stay consistent (argparse displays
    # prog=sys.argv[0]).
    sys.argv = [f"minutegenius {command}"] + rest

    if command == "check":
        from minutegenius.check import main as run
    elif command == "meeting":
        from minutegenius.meeting import main as run
    elif command == "transcribe":
        from minutegenius.transcribe import main as run
    else:
        from minutegenius.vtt import main as run

    run()


if __name__ == "__main__":
    main()

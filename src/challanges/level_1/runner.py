import argparse
from pathlib import Path

from src.challanges.base_challange import ChallengePaths, ChallengeResult
from src.challanges.level_1.agents import Level1Challenge
from src.config.config import config
from src.tracing.langfuse_helpers import generate_session_id


def build_level_1_challenge(
    input_dir: str | Path = "datasets/level_1",
    output: str | Path = "outputs/level_1_submission.txt",
    session_id: str | None = None,
) -> Level1Challenge:
    paths = ChallengePaths.from_directory(input_dir=input_dir, output=output)
    return Level1Challenge(paths=paths, session_id=session_id or generate_session_id())


def run_level_1(
    input_dir: str | Path = "datasets/level_1",
    output: str | Path = "outputs/level_1_submission.txt",
    session_id: str | None = None,
    limit: int | None = None,
) -> ChallengeResult:
    challenge = build_level_1_challenge(input_dir=input_dir, output=output, session_id=session_id)
    input_dir_str = str(input_dir).replace("\\", "/")
    default_dataset_run = input_dir_str.endswith("datasets/level_1")
    effective_limit = config.TEST_ROW_LIMIT if limit is None and default_dataset_run else limit
    return challenge.run(limit=effective_limit)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Reply Mirror level 1 fraud detection.")
    parser.add_argument("--input-dir", default="datasets/level_1")
    parser.add_argument("--output", default="outputs/level_1_submission.txt")
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    result = run_level_1(args.input_dir, args.output, args.session_id, args.limit)
    print(f"Wrote {len(result.suspected_transaction_ids)} suspected fraud IDs to {result.output_path}")


if __name__ == "__main__":
    main()

"""CLI script to generate and save synthetic SFT and DPO training data."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.logger import setup_logger
from src.data.dataset import save_dataset_splits
from src.data.generator import SyntheticDataEngine

logger = setup_logger("generate_data")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic dataset for SFT & DPO")
    parser.add_argument("--schemas", type=str, default="data/schemas", help="Path to schema directory")
    parser.add_argument("--output", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--samples-per-tool", type=int, default=150, help="Number of samples per tool")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    logger.info("Initializing Synthetic Data Engine (schemas: %s)...", args.schemas)
    engine = SyntheticDataEngine(schemas_dir=args.schemas, seed=args.seed)

    logger.info("Generating %d samples per tool across %d tools...", args.samples_per_tool, len(engine.schemas))
    dataset_splits = engine.generate_dataset(samples_per_tool=args.samples_per_tool)

    saved_paths = save_dataset_splits(dataset_splits, output_dir=args.output)
    logger.info("Datasets generated successfully:")
    for split_name, file_path in saved_paths.items():
        logger.info("  - %s: %d records -> %s", split_name, len(dataset_splits[split_name]), file_path)


if __name__ == "__main__":
    main()

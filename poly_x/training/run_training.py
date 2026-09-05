"""
CLI Orchestrator for Polymer Model Training

Usage (standalone):
    python run_training.py --property tg --data-path data/pi1m_tg.csv
    python run_training.py --property all --data-dir data/

Usage (via Django management command):
    python manage.py train_polymer_models --property tg --data-path data/pi1m_tg.csv
"""

import argparse
import logging
import sys

logger = logging.getLogger(__name__)

# Property configuration
PROPERTY_CONFIG = {
    'tg': {
        'display': 'Glass Transition Temperature',
        'unit': 'K',
        'target_column': 'Tg',
    },
    'tm': {
        'display': 'Melting Temperature',
        'unit': 'K',
        'target_column': 'Tm',
    },
    'td': {
        'display': 'Decomposition Temperature',
        'unit': 'K',
        'target_column': 'Td',
    },
    'density': {
        'display': 'Density',
        'unit': 'g/cm³',
        'target_column': 'density',
    },
    'solubility_param': {
        'display': 'Solubility Parameter',
        'unit': 'MPa^0.5',
        'target_column': 'solubility_param',
    },
}


def train_property(property_name: str, data_path: str,
                   max_samples: int = None, output_dir: str = None):
    """Train a single property model."""
    config = PROPERTY_CONFIG.get(property_name)
    if config is None:
        raise ValueError(f"Unknown property: {property_name}. "
                         f"Available: {list(PROPERTY_CONFIG.keys())}")

    if property_name == 'tg':
        from .train_tg import TgTrainer
        trainer = TgTrainer(
            data_path=data_path,
            output_dir=output_dir,
            max_samples=max_samples,
        )
    else:
        from .train_properties import PropertyTrainer
        trainer = PropertyTrainer(
            property_name=property_name,
            data_path=data_path,
            target_column=config['target_column'],
            property_display=config['display'],
            property_unit=config['unit'],
            output_dir=output_dir,
            max_samples=max_samples,
        )

    return trainer.train()


def main():
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    )

    parser = argparse.ArgumentParser(
        description='Train polymer property prediction models')
    parser.add_argument('--property', required=True,
                        choices=list(PROPERTY_CONFIG.keys()) + ['all'],
                        help='Property to train (or "all")')
    parser.add_argument('--data-path', required=True,
                        help='Path to CSV dataset')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max training samples (for quick testing)')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory for trained models')

    args = parser.parse_args()

    if args.property == 'all':
        for prop in PROPERTY_CONFIG:
            try:
                logger.info("Training %s...", prop)
                train_property(prop, args.data_path,
                               args.max_samples, args.output_dir)
            except Exception as e:
                logger.error("Failed to train %s: %s", prop, e)
    else:
        train_property(args.property, args.data_path,
                       args.max_samples, args.output_dir)


if __name__ == '__main__':
    main()

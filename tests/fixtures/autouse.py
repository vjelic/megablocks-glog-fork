# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

import gc
import logging
import os

import pytest
import torch
import torch.distributed as dist


@pytest.fixture(autouse=True)
def clear_cuda_cache(request: pytest.FixtureRequest):
    """Clear memory between GPU tests."""
    marker = request.node.get_closest_marker('gpu')
    if marker is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()  # Only gc on GPU tests as it 2x slows down CPU tests


@pytest.fixture(autouse=True)
def reset_mlflow_tracking_dir():
    """Reset MLFlow tracking dir so it doesn't persist across tests."""
    try:
        import mlflow
        mlflow.set_tracking_uri(None)  # type: ignore
    except ModuleNotFoundError:
        # MLFlow not installed
        pass


@pytest.fixture(scope='session')
def cleanup_dist():
    """Ensure all dist tests clean up resources properly."""
    yield
    # Avoid race condition where a test is still writing to a file on one rank
    # while the file system is being torn down on another rank.
    dist.barrier()


@pytest.fixture(autouse=True, scope='session')
def configure_dist(request: pytest.FixtureRequest):
    """Set up distributed processing for the entire test session."""
    # Don't override environment variables if they're already set by torchrun
    if 'RANK' not in os.environ:
        # Only set defaults if not running with torchrun
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
    
    # Get values from environment (respecting torchrun settings)
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    
    print(f"Initializing distributed with world_size={world_size}, rank={rank}")
    
    # Initialize distributed processing
    if not dist.is_initialized():
        try:
            # Try NCCL first if CUDA is available
            if torch.cuda.is_available():
                dist.init_process_group(
                    backend='nccl',
                    init_method='env://',
                    world_size=world_size,
                    rank=rank
                )
                print(f"Distributed process group initialized with NCCL backend")
            else:
                # Use gloo for CPU-only environments
                dist.init_process_group(
                    backend='gloo',
                    init_method='env://',
                    world_size=world_size,
                    rank=rank
                )
                print(f"Distributed process group initialized with Gloo backend")
        except Exception as e:
            print(f"Failed to initialize distributed process group: {e}")
            print(f"Environment: RANK={os.environ.get('RANK')}, WORLD_SIZE={os.environ.get('WORLD_SIZE')}, MASTER_ADDR={os.environ.get('MASTER_ADDR')}, MASTER_PORT={os.environ.get('MASTER_PORT')}")
            raise  # Re-raise to see the full error

    # Hold PyTest until all ranks have reached this barrier. Ensure that no rank starts
    # any test before other ranks are ready to start it, which could be a cause of random timeouts
    # (e.g. rank 1 starts the next test while rank 0 is finishing up the previous test).
    dist.barrier()

@pytest.fixture(autouse=True)
def set_log_levels():
    """Ensures all log levels are set to DEBUG."""
    logging.basicConfig()


@pytest.fixture(autouse=True)
def seed_all(rank_zero_seed: int, monkeypatch: pytest.MonkeyPatch):
    """Monkeypatch reproducibility.

    Make get_random_seed to always return the rank zero seed, and set the random seed before each test to the rank local
    seed.
    """

    def get_random_seed():
        return rank_zero_seed
    
    monkeypatch.setitem(globals(), 'get_random_seed', get_random_seed)
    rank = dist.get_rank() if dist.is_initialized() else 0
    local_seed = rank_zero_seed + rank
    torch.manual_seed(local_seed)
    torch.use_deterministic_algorithms(True)



@pytest.fixture(autouse=True)
def remove_run_name_env_var():
    # Remove environment variables for run names in unit tests
    composer_run_name = os.environ.get('COMPOSER_RUN_NAME')
    run_name = os.environ.get('RUN_NAME')

    if 'COMPOSER_RUN_NAME' in os.environ:
        del os.environ['COMPOSER_RUN_NAME']
    if 'RUN_NAME' in os.environ:
        del os.environ['RUN_NAME']

    yield

    if composer_run_name is not None:
        os.environ['COMPOSER_RUN_NAME'] = composer_run_name
    if run_name is not None:
        os.environ['RUN_NAME'] = run_name

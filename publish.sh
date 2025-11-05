#!/bin/bash
# Manual PyPI publish script using uv
# Usage: ./publish.sh [testpypi|pypi]

set -e

REPO=${1:-pypi}  # Default to PyPI, use 'testpypi' for testing

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "Cleaning previous builds..."
rm -rf dist build *.egg-info src/*.egg-info

echo "Removing __pycache__ and pytest cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + || true
find . -type d -name ".pytest_cache" -exec rm -rf {} + || true

echo "Building package with uv..."
uv build

echo "Checking package..."
uvx twine check dist/*

if [ "$REPO" = "testpypi" ]; then
    echo "Uploading to TestPyPI..."
    if [ -z "$TESTPYPI_API_TOKEN" ]; then
        echo -n "Enter TestPyPI API token (starts with pypi-): "
        stty -echo
        read -r TESTPYPI_API_TOKEN
        stty echo
        echo
    fi
    export TWINE_USERNAME="__token__"
    export TWINE_PASSWORD="$TESTPYPI_API_TOKEN"
    uvx twine upload --repository testpypi dist/*
    echo "Published to TestPyPI: https://test.pypi.org/project/ecg-k-fold/"
else
    echo "Uploading to PyPI..."
    if [ -z "$PYPI_API_TOKEN" ]; then
        echo -n "Enter PyPI API token (starts with pypi-): "
        stty -echo
        read -r PYPI_API_TOKEN
        stty echo
        echo
    fi
    export TWINE_USERNAME="__token__"
    export TWINE_PASSWORD="$PYPI_API_TOKEN"
    uvx twine upload dist/*
    echo "Published to PyPI: https://pypi.org/project/ecg-k-fold/"
fi

echo "Done!"

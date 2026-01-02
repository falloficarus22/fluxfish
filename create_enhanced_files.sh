#!/bin/bash
# create_enhanced_files.sh - Creates all necessary files for enhanced encoder

set -e

echo "Creating enhanced encoder files..."

# Create conftest.py
cat > tests/conftest.py << 'CONFTEST_EOF'
"""pytest configuration to fix import paths"""
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
python_dir = repo_root / "python"

if str(python_dir) not in sys.path:
    sys.path.insert(0, str(python_dir))
CONFTEST_EOF

echo "✓ Created tests/conftest.py"

# Ensure __init__.py files exist
touch python/liquid_chess/__init__.py
touch python/liquid_chess/models/__init__.py
touch python/liquid_chess/models/lrt/__init__.py

echo "✓ Ensured __init__.py files exist"

# Now you can run tests with:
echo ""
echo "Setup complete! Now you can:"
echo "1. Run tests: pytest tests/test_imports.py -v"
echo "2. Or with explicit path: PYTHONPATH=\$PWD/python pytest tests/ -v"
echo ""
echo "Missing files that need content from integration guide:"
echo "  - python/liquid_chess/models/lrt/feature_extraction.py"
echo "  - python/liquid_chess/models/lrt/enhanced_encoder.py"
echo "  - tests/test_feature_extraction.py"
echo "  - tests/test_enhanced_encoder.py"
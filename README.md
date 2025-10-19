# River Segmentation Under Canopy Cover

Luminance-Prioritized Deep Learning for River Segmentation Under Forest Canopy  
**CERI 2026 Submission**

## Dataset
- 415 annotated RGB drone images
- River Bride, Crookstown, County Cork
- March & June 2025 surveys
- Resolution: 5280×3956 → 512×512

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Analyze dataset
python scripts/01_analyze_dataset.py

# Train baseline
python scripts/04_train_unet.py
```

## Project Structure
See `docs/setup_guide.md` for details.

## Author
Hasitha Adikari
Nimbus|SIRIG|MTU
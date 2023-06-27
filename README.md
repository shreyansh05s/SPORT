# SPORT
Sports Object Recognition and Tracking

## Requirements
- Python 3.11 or higher

## Setup Environment (Optional but recommended)
Create a virtual environment
```bash
python -m venv .venv
```

Activate the virtual environment
For windows
```bash
.venv\Scripts\activate.bat
```

For Linux
```bash
source .venv/bin/activate
```

## Installation
```bash
pip install -e . -r requirements.txt
```

## Dataset
SportsMOT: A Large Multi-Object Tracking Dataset in Multiple Sports Scenes

link: https://codalab.lisn.upsaclay.fr/competitions/12424

## Demo
```bash
python streamlit run sport/cli/demo.py
```

## References

[1] SportsMOT: A Large Multi-Object Tracking Dataset in Multiple Sports Scenes
\[[arXiv](https://arxiv.org/abs/2109.14834)\]
<!-- TODO -->
\[[Project]()\]

[2] DeepSORT: Deep Learning to Track Custom Objects with PyTorch
\[[arXiv](https://arxiv.org/abs/1703.07402)\]

[3] TransTrack: Multiple-Object Tracking with Transformer
\[[arXiv](https://arxiv.org/pdf/2012.15460v2.pdf)\]

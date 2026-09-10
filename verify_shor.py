import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.shor_code import verify_identity_single_qubit

verify_identity_single_qubit(theta_y=0.7, theta_z=1.3)
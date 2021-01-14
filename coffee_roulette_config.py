from pathlib import Path

class crConfig:
    working_dir = '' 
    file_npy = Path(working_dir).resolve() / "crProbMatrix.npy"
    file_pkl = Path(working_dir).resolve() / "crParticipantDict.p" 

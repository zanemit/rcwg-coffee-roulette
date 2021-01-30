from pathlib import Path

class crConfig:
    working_dir = 'C:\\Users\\zanem\\Documents\\rcwg-coffee-roulette'
    file_npy = Path(working_dir).resolve() / "crProbMatrix.npy"
    file_pkl = Path(working_dir).resolve() / "crParticipantDict.p" 
    file_txt = Path(working_dir).resolve() / "crConversationStarters.txt"

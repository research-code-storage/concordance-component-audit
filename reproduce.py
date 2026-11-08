"""Entry point for reproducing the manuscript's numerical results."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent/'code'))
from reproduction import main
if __name__=='__main__':main()

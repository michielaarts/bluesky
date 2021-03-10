import pandas as pd
import numpy as np
from tkinter import filedialog, Tk
from pathlib import Path

root = Tk()
OUTPUT_DIR = Path('../output/')
filenames = filedialog.askopenfilenames(initialdir=OUTPUT_DIR)
root.destroy()

for fname in filenames:
    data = pd.read_csv(Path(fname), comment='#', skipinitialspace=True)

from tkinter import Tk, filedialog
from pathlib import Path
import pickle as pkl

RES_FOLDER = Path('../../output/RESULT/')

tk_root = Tk()
res_files = filedialog.askopenfilenames(initialdir=RES_FOLDER, title='Select results to include in total',
                                        filetypes=[('pkl', '*.pkl')])
tk_root.destroy()

n_total_ac = 0.
for res_f in res_files:
    with open(res_f, 'rb') as f:
        res_dict = pkl.load(f)
        for (run, result) in res_dict.items():
            if not run == 'name':
                n_total_ac += result['conflog']['ntotal_ac'].iloc[-1]
print('Total no. of aircraft simulated is: ', n_total_ac)

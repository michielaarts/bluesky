from tkinter import Tk, filedialog
from pathlib import Path
import pickle as pkl

RES_FOLDER = Path('../../output/RESULT/')

tk_root = Tk()
res_files = filedialog.askopenfilenames(initialdir=RES_FOLDER, title='Select results to include in total',
                                        filetypes=[('pkl', '*.pkl')])
tk_root.destroy()

n_total_ac = 0.
n_total_res = dict()
for res_f in res_files:
    fname = Path(res_f).name[:-4]
    n_total_res[fname] = 0.
    with open(res_f, 'rb') as f:
        res_dict = pkl.load(f)
        for (run, result) in res_dict.items():
            if not run == 'name':
                n_total_ac += result['conflog']['ntotal_ac'].iloc[-1]
                n_total_res[fname] += result['conflog']['ntotal_ac'].iloc[-1]
print(f'Total no. of aircraft simulated is: {n_total_ac:.0f}')
print('Total no. per experiment:', '\n'.join(f"Exp. {run}: {ntotal:.0f}" for run, ntotal in n_total_res.items()))

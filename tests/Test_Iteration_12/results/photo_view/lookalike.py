"""Do the fit diagnostics (rms of the match residual after the pose fit, size of the fitted attitude correction, inliers) separate right answers from wrong ones?
usage: python lookalike.py details.csv"""
import sys, pandas as pd, numpy as np
from score import pts
from sklearn.metrics import roc_auc_score
d = pd.read_csv(sys.argv[1]); d['p'] = d.apply(pts, axis=1)
a = d[(d.status == 'match') & (d.pose_fit_used == True)].copy()
a['wrong'] = (a.p == -20).astype(int)
a['att'] = np.hypot(a.fit_dpitch, a.fit_droll)
print('answers with a fit', len(a), 'wrong', int(a.wrong.sum()), 'of', int((d.status == 'match').sum()), 'answers')
if a.wrong.sum() >= 2:
    for col, sign in (('inliers', -1), ('fit_rms_px', 1), ('att', 1), ('fit_dyaw', 1), ('fit_dheight', 1)):
        v = a[col].abs() if col in ('fit_dyaw', 'fit_dheight') else a[col]
        print('%-12s AUC for wrong (higher = more wrong): %.2f   median right %.2f wrong %.2f' % (col, roc_auc_score(a.wrong, sign * v), v[a.wrong == 0].median(), v[a.wrong == 1].median()))

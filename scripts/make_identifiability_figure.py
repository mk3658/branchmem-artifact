from __future__ import annotations
import csv, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
# The figure belongs to the paper, the simulation table to the result tree;
# deriving either from __file__ put both next to whichever copy was run.
OUT = ROOT / "paper"
CSV_OUT = ROOT / "results" / "contextual_branchguard"
OUT.mkdir(parents=True, exist_ok=True)
CSV_OUT.mkdir(parents=True, exist_ok=True)
lam = 0.25
delta = 0.05
ms = [25, 100, 400]
rhos = np.linspace(0,1,201)

def binom_probs(m, rho):
    probs=[]
    for k in range(m+1):
        probs.append(math.comb(m,k)*(rho**k)*((1-rho)**(m-k)))
    s=sum(probs)
    return [p/s for p in probs]

def metrics(m,rho):
    eps=math.sqrt(math.log(2/delta)/(2*m))
    follow=invert=0.0
    for k,p in enumerate(binom_probs(m,rho)):
        phat=k/m
        lo=max(0.0,phat-eps); hi=min(1.0,phat+eps)
        if lo >= 1-lam:
            follow += p
        elif hi <= lam:
            invert += p
    cov=follow+invert
    wrong=follow*(1-rho)+invert*rho
    cond=wrong/cov if cov>0 else float('nan')
    return eps,follow,invert,cov,wrong,cond

rows=[]
for m in ms:
    for rho in rhos:
        eps,follow,invert,cov,wrong,cond=metrics(m,float(rho))
        rows.append([m,float(rho),eps,follow,invert,cov,wrong,cond])
with open(CSV_OUT/'branchguard_simulation.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['m','rho','epsilon','p_follow','p_invert','coverage','wrong_commit_rate','conditional_wrong_rate']); w.writerows(rows)

plt.rcParams.update({'font.family':'serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(7.0,2.25))
ax=axs[0]
# Each region needs its own colour: axvspan defaults to the same one, which
# rendered all three decision regions identically and made the panel's whole
# point -- that the Bayes action changes across rho -- invisible.
ax.axvspan(0,lam,alpha=.30,color='#d1495b',lw=0,label='invert provenance')
ax.axvspan(lam,1-lam,alpha=.30,color='#c9cbcf',lw=0,label='abstain')
ax.axvspan(1-lam,1,alpha=.30,color='#2e86ab',lw=0,label='follow provenance')
ax.axvline(lam,ls='--',lw=.8,c='black'); ax.axvline(1-lam,ls='--',lw=.8,c='black')
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_yticks([])
ax.set_xlabel(r'provenance reliability $\rho=P(Y=\arg\max z)$')
ax.set_title(r'(a) Bayes action, abstention loss $\lambda=0.25$')
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.28),ncol=3,frameon=False,fontsize=7)

ax=axs[1]
for m in ms:
    covs=[metrics(m,float(r))[3] for r in rhos]
    ax.plot(rhos,covs,label=f'm={m}')
ax.axvline(lam,ls='--',lw=.8,c='black'); ax.axvline(1-lam,ls='--',lw=.8,c='black')
ax.set_xlim(0,1); ax.set_ylim(-.02,1.02)
ax.set_xlabel(r'true provenance reliability $\rho$')
ax.set_ylabel('certified commit probability')
ax.set_title(r'(b) BranchGuard coverage ($\delta=0.05$)')
ax.grid(axis='y',alpha=.25)
# One column, centred: the band between the two lambda rules is the only part
# of this panel that is empty at every m, and a 3-column legend overran it.
ax.legend(frameon=False,loc='center',ncol=1,fontsize=7)
fig.subplots_adjust(left=.08,right=.99,top=.84,bottom=.32,wspace=.28)
fig.savefig(OUT/'identifiability_theory.png',dpi=300)
plt.close(fig)

for rho in [0.0,0.15,0.5,0.85,1.0]:
    print('rho',rho)
    for m in ms:
        eps,follow,invert,cov,wrong,cond=metrics(m,rho)
        print(m,'eps',round(eps,3),'cov',round(cov,3),'cond',None if math.isnan(cond) else round(cond,3),'follow',round(follow,3),'invert',round(invert,3))

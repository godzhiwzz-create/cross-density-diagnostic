"""Task 3 (reviewer): multi-covariate content-matched real-vs-synth gate-output probe.

Extends the beta-only matched probe (di_signal_ladder_probe.match_pairs) by ALSO
matching content covariates between RTTS(real) and Foggy-Cityscapes(synth):
  - brightness (mean luma), contrast (std luma)
  - object count, mean bbox scale (sqrt area), class histogram (shared coarse vocab)
  - compression/noise proxy (Laplacian high-frequency energy)

For each real image we keep only synth candidates whose RELATIVE beta diff <= beta_tol,
then require each requested covariate within a z-scored tolerance, then pick the
nearest synth in the joint standardized covariate space. We then recompute the
real-vs-synth gate-output MMD/KS numerator (and the cross-density denominator,
unchanged) on the new content-matched pair set, for RGB + 3 physical conditions.

Reads gate checkpoints from the EXISTING 20-epoch runs (tsp_dadg_<variant>/seedX).
Writes only to the reviewer_hardening scratch dir. No paper file touched.
"""
from __future__ import annotations
import sys, json, argparse, csv
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(".")
sys.path.insert(0, str(REPO))
import gate.analysis.di_signal_ladder_probe as P  # reuse collect_beta, gate_outputs, load_gate, mmd_rbf, mean_ks

# shared coarse class vocab across RTTS(5) and Foggy(8):
#   RTTS:   0 person 1 bicycle 2 car 3 motorcycle 4 bus
#   Foggy:  0 person 1 rider 2 car 3 truck 4 bus 5 train 6 motorcycle 7 bicycle
# coarse buckets: person(+rider), two_wheel(bicycle+motorcycle), car, large(bus+truck+train)
RTTS_MAP =  {0:"person",1:"two_wheel",2:"car",3:"two_wheel",4:"large"}
FOGGY_MAP = {0:"person",1:"person",2:"car",3:"large",4:"large",5:"large",6:"two_wheel",7:"two_wheel"}
COARSE = ["person","two_wheel","car","large"]

def label_path_for(img_path: str) -> Path:
    p = Path(img_path)
    parts = list(p.parts)
    # replace last 'images' segment with 'labels'
    for i in range(len(parts)-1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"; break
    lp = Path(*parts).with_suffix(".txt")
    return lp

def read_label(img_path: str, cmap: dict):
    lp = label_path_for(img_path)
    n = 0; areas = []; hist = {c:0 for c in COARSE}
    if lp.exists():
        for line in lp.read_text().splitlines():
            t = line.split()
            if len(t) < 5: continue
            cls = int(float(t[0])); w = float(t[3]); h = float(t[4])
            n += 1; areas.append((w*h)**0.5)
            cc = cmap.get(cls)
            if cc: hist[cc] += 1
    mean_scale = float(np.mean(areas)) if areas else 0.0
    total = sum(hist.values())
    hvec = np.array([hist[c]/total if total else 0.0 for c in COARSE], dtype=np.float64)
    return n, mean_scale, hvec

@torch.no_grad()
def photo_covariates(paths, batch, workers, imgsz, device):
    """brightness(mean luma), contrast(std luma), laplacian hi-freq energy proxy."""
    loader = P.make_loader([Path(p) for p in paths], batch=batch, workers=workers, imgsz=imgsz)
    lap_k = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32, device=device).view(1,1,3,3)
    bri=[]; con=[]; hf=[]
    for imgs,_ in loader:
        imgs = imgs.to(device)
        lum = (0.299*imgs[:,0:1]+0.587*imgs[:,1:2]+0.114*imgs[:,2:3])
        bri.append(lum.mean(dim=(1,2,3)).cpu().numpy())
        con.append(lum.std(dim=(1,2,3)).cpu().numpy())
        lp = F.conv2d(lum, lap_k, padding=1)
        hf.append(lp.abs().mean(dim=(1,2,3)).cpu().numpy())
    return np.concatenate(bri), np.concatenate(con), np.concatenate(hf)

def build_covar_matrix(paths, cmap, batch, workers, imgsz, device):
    bri, con, hf = photo_covariates(paths, batch, workers, imgsz, device)
    counts=[]; scales=[]; hists=[]
    for p in paths:
        n, sc, hv = read_label(p, cmap)
        counts.append(n); scales.append(sc); hists.append(hv)
    counts=np.array(counts,float); scales=np.array(scales,float); hists=np.vstack(hists)
    # columns: bri, con, hf, count, scale, + 4 class fractions
    M = np.column_stack([bri, con, hf, counts, scales, hists])
    names = ["brightness","contrast","hf_energy","obj_count","bbox_scale"]+["cls_"+c for c in COARSE]
    return M, names

def zmatch_pairs(real_beta, synth_beta, Rz, Sz, beta_tol, cov_tol, mode="nn"):
    """For each real i: synth candidates with rel beta diff<=beta_tol.
    mode='nn'      -> always pick nearest synth in std covariate space (content balancing,
                      keeps pair count high; this is the primary reviewer-requested match).
    mode='caliper' -> additionally require every standardized covariate within cov_tol
                      (hard content filter; fewer pairs, stricter)."""
    pairs=[]
    for i in range(len(real_beta)):
        rb = real_beta[i]
        bdiff = np.abs(synth_beta - rb)/(rb+1e-6)
        cand = np.where(bdiff <= beta_tol)[0]
        if len(cand)==0: continue
        if mode=="caliper":
            within = np.all(np.abs(Sz[cand]-Rz[i]) <= cov_tol, axis=1)
            cand = cand[within]
            if len(cand)==0: continue
        dd = np.linalg.norm(Sz[cand]-Rz[i], axis=1)
        j = int(cand[int(dd.argmin())])
        pairs.append((i,j))
    return pairs

def covariate_balance(Rz_s, Sz_s, pairs, names):
    """Standardized mean difference (SMD) of each covariate between matched real & synth."""
    if not pairs: return {}
    ri=[i for i,_ in pairs]; sj=[j for _,j in pairs]
    rm=Rz_s[ri].mean(0); sm=Sz_s[sj].mean(0)
    return {names[k]: float(abs(rm[k]-sm[k])) for k in range(len(names))}

@torch.no_grad()
def numerator_for_variant(variant, seed, real_paths, synth_paths, pairs,
                          synth_paths_by_tag, batch, workers, imgsz, device, run_suffix):
    run_dir = P.EXP_ROOT / ("tsp_dadg_%s%s"%(variant,run_suffix)) / ("seed%d"%seed)
    gate, cfg = P.load_gate(run_dir, device=device)
    real_matched = [real_paths[i] for i,_ in pairs]
    synth_matched = [synth_paths[j] for _,j in pairs]
    r = P.gate_outputs(gate,cfg,real_matched,batch=batch,workers=workers,imgsz=imgsz,device=device)
    s = P.gate_outputs(gate,cfg,synth_matched,batch=batch,workers=workers,imgsz=imgsz,device=device)
    mmd_num = P.mmd_rbf(r,s); ks_num = P.mean_ks(r,s)
    # denominator (cross-density synth-synth) on full synth sets, unchanged definition
    outs={tag:P.gate_outputs(gate,cfg,pp,batch=batch,workers=workers,imgsz=imgsz,device=device)
          for tag,pp in synth_paths_by_tag.items()}
    den_mmd=[]; den_ks=[]
    tags=P.SYNTH_TAGS
    for a in range(len(tags)):
        for b in range(a+1,len(tags)):
            den_mmd.append(P.mmd_rbf(outs[tags[a]],outs[tags[b]]))
            den_ks.append(P.mean_ks(outs[tags[a]],outs[tags[b]]))
    return {"mmd_num":mmd_num,"ks_num":ks_num,
            "mmd_den":float(np.mean(den_mmd)),"ks_den":float(np.mean(den_ks)),
            "mmd_ratio":mmd_num/max(np.mean(den_mmd),1e-9),
            "ks_ratio":ks_num/max(np.mean(den_ks),1e-9)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--variants",nargs="+",default=["rgb","raw_dark_channel","raw_transmission","tsp_grad_mag"])
    ap.add_argument("--seeds",nargs="+",type=int,default=[42,123,456,789,2024])
    ap.add_argument("--max-images",type=int,default=600)
    ap.add_argument("--batch",type=int,default=32)
    ap.add_argument("--workers",type=int,default=8)
    ap.add_argument("--imgsz",type=int,default=640)
    ap.add_argument("--beta-tol",type=float,default=0.05)
    ap.add_argument("--cov-tol",type=float,default=0.75,help="z-score tolerance per covariate")
    ap.add_argument("--covariates",nargs="+",default=["brightness","contrast","hf_energy","obj_count","bbox_scale","cls"])
    ap.add_argument("--device",default="0")
    ap.add_argument("--run-suffix",default="")
    ap.add_argument("--out",default=str(REPO/"gate/experiments_dlhost/reviewer_hardening_20260617/task3_covariate/task3_covariate.json"))
    args=ap.parse_args()
    device="cuda:%s"%args.device if torch.cuda.is_available() and args.device!="cpu" else "cpu"

    real_paths, real_beta = P.collect_beta("rtts",max_images=args.max_images,batch=args.batch,
                                           workers=args.workers,imgsz=args.imgsz,device=device)
    synth_paths=[]; synth_paths_by_tag={}; bparts=[]
    for tag in P.SYNTH_TAGS:
        pp,bb=P.collect_beta(tag,max_images=args.max_images,batch=args.batch,workers=args.workers,
                             imgsz=args.imgsz,device=device)
        synth_paths.extend(pp); synth_paths_by_tag[tag]=pp; bparts.append(bb)
    synth_beta=np.concatenate(bparts)

    # covariate matrices
    Rm, names = build_covar_matrix(real_paths, RTTS_MAP, args.batch,args.workers,args.imgsz,device)
    Sm, _     = build_covar_matrix(synth_paths, FOGGY_MAP, args.batch,args.workers,args.imgsz,device)
    # standardize on POOLED stats so z-tolerances are comparable
    pooled=np.vstack([Rm,Sm]); mu=pooled.mean(0); sd=pooled.std(0)+1e-9
    Rz=(Rm-mu)/sd; Sz=(Sm-mu)/sd
    # select covariate columns
    sel=[]
    for i,nm in enumerate(names):
        base = nm.split("_")[0] if nm.startswith("cls") else nm
        key = "cls" if nm.startswith("cls") else nm
        if key in args.covariates or (nm in args.covariates):
            sel.append(i)
    Rz_s=Rz[:,sel]; Sz_s=Sz[:,sel]; sel_names=[names[i] for i in sel]

    # baseline beta-only pairs (reproduce original) and multi-covariate pairs (nn + caliper)
    pairs_beta = P.match_pairs(real_beta, synth_beta, tol=args.beta_tol)
    pairs_nn   = zmatch_pairs(real_beta, synth_beta, Rz_s, Sz_s, args.beta_tol, args.cov_tol, mode="nn")
    pairs_cal  = zmatch_pairs(real_beta, synth_beta, Rz_s, Sz_s, args.beta_tol, args.cov_tol, mode="caliper")
    # covariate balance (mean |SMD| over selected covariates) before vs after matching
    bal_beta=covariate_balance(Rz_s,Sz_s,[(i,j) for i,j in pairs_beta],sel_names)
    bal_nn  =covariate_balance(Rz_s,Sz_s,pairs_nn,sel_names)
    bal_cal =covariate_balance(Rz_s,Sz_s,pairs_cal,sel_names)

    out={"config":{"max_images":args.max_images,"beta_tol":args.beta_tol,"cov_tol":args.cov_tol,
                   "covariates_used":sel_names,"run_suffix":args.run_suffix,
                   "n_real":len(real_paths),"n_synth":len(synth_paths)},
         "n_pairs_beta_only":len(pairs_beta),"n_pairs_nn":len(pairs_nn),"n_pairs_caliper":len(pairs_cal),
         "covariate_balance":{"beta_only_meanSMD":float(np.mean(list(bal_beta.values()))) if bal_beta else None,
                              "nn_meanSMD":float(np.mean(list(bal_nn.values()))) if bal_nn else None,
                              "caliper_meanSMD":float(np.mean(list(bal_cal.values()))) if bal_cal else None,
                              "per_covariate_beta_only":bal_beta,"per_covariate_nn":bal_nn,"per_covariate_caliper":bal_cal},
         "beta_only":{}, "multicovar_nn":{}, "multicovar_caliper":{}}
    print("pairs: beta_only=%d  nn=%d  caliper=%d  (covariates=%s)"%(len(pairs_beta),len(pairs_nn),len(pairs_cal),sel_names),flush=True)
    print("mean|SMD| balance: beta_only=%.3f  nn=%.3f  caliper=%.3f"%(
        out["covariate_balance"]["beta_only_meanSMD"] or -1,
        out["covariate_balance"]["nn_meanSMD"] or -1,
        out["covariate_balance"]["caliper_meanSMD"] or -1),flush=True)

    for label,pairs in [("beta_only",pairs_beta),("multicovar_nn",pairs_nn),("multicovar_caliper",pairs_cal)]:
        if not pairs:
            print("  [%s] no pairs, skipping"%label,flush=True); continue
        for variant in args.variants:
            vals={"mmd_num":[],"ks_num":[],"mmd_den":[],"ks_den":[],"mmd_ratio":[],"ks_ratio":[]}
            for seed in args.seeds:
                try:
                    r=numerator_for_variant(variant,seed,real_paths,synth_paths,pairs,
                                            synth_paths_by_tag,args.batch,args.workers,args.imgsz,device,args.run_suffix)
                    for k in vals: vals[k].append(r[k])
                except FileNotFoundError as e:
                    print("[missing] %s seed%d: %s"%(variant,seed,e),flush=True)
            agg={k:(float(np.mean(v)),float(np.std(v)),len(v)) for k,v in vals.items()}
            out[label][variant]={k:{"mean":m,"std":s,"n":n} for k,(m,s,n) in agg.items()}
            print("  [%s] %-18s mmd_num=%.4f±%.4f  mmd_ratio=%.3f  ks_num=%.4f"
                  %(label,variant,agg["mmd_num"][0],agg["mmd_num"][1],agg["mmd_ratio"][0],agg["ks_num"][0]),flush=True)

    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out).write_text(json.dumps(out,indent=2,ensure_ascii=False))
    print("\nSaved -> %s"%args.out)

if __name__=="__main__":
    main()

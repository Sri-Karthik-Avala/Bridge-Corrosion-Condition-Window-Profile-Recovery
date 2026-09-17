# made by - Karthik
import os, sys, json, time, math, random
import numpy as np
import pandas as pd

T0 = time.time()
BUDGET = float(os.environ.get("ERIS_BUDGET", "4500"))
SEED = int(os.environ.get("ERIS_SEED", "17"))
IMG = int(os.environ.get("ERIS_IMG", "256"))
BS = int(os.environ.get("ERIS_BS", "16"))
VAL_FRAC = float(os.environ.get("ERIS_VAL", "0.15"))
MAX_EPOCH = int(os.environ.get("ERIS_MAX_EPOCH", "40"))
ARCH = os.environ.get("ERIS_ARCH", "resnet34")

random.seed(SEED)
np.random.seed(SEED)

STATES = ["fair", "poor", "severe"]
G = 8


def find_data_root(argv_root):
    cands = []
    if argv_root:
        cands.append(argv_root)
    cands += [".", "dataset/public", "public", "../dataset/public", "/kaggle/input"]
    for c in cands:
        try:
            if c and os.path.isfile(os.path.join(c, "train.csv")) and os.path.isfile(os.path.join(c, "test.csv")):
                return c
        except Exception:
            pass
    for base in [".", "..", "/kaggle/input"]:
        try:
            for dp, dn, fn in os.walk(base):
                if "train.csv" in fn and "test.csv" in fn:
                    return dp
        except Exception:
            pass
    return "."


ARGV_ROOT = sys.argv[1] if len(sys.argv) > 1 else None
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join("working", "submission.csv")
ROOT = find_data_root(ARGV_ROOT)

train = pd.read_csv(os.path.join(ROOT, "train.csv"))
test = pd.read_csv(os.path.join(ROOT, "test.csv"))


def tok(r, c):
    return "r%02d_c%02d" % (r, c)


TOKS = [tok(r, c) for r in range(G) for c in range(G)]
TIDX = {t: i for i, t in enumerate(TOKS)}


def parse_answer(s):
    a = json.loads(s)
    m = np.zeros((3, G, G), dtype=np.float32)
    for si, st in enumerate(STATES):
        for t in a.get(st + "_cells", []):
            i = TIDX.get(t)
            if i is not None:
                m[si, i // G, i % G] = 1.0
    b = a.get("area_bins", {})
    bins = np.array([int(b.get(st, 0)) for st in STATES], dtype=np.int64)
    bins = np.clip(bins, 0, 9)
    return m, bins


YM = np.zeros((len(train), 3, G, G), dtype=np.float32)
YB = np.zeros((len(train), 3), dtype=np.int64)
for i, s in enumerate(train.answer_json.values):
    m, b = parse_answer(s)
    YM[i] = m
    YB[i] = b

EMPTY_BIN = np.zeros(3, dtype=np.int64)
NONEMPTY_MIN = np.ones(3, dtype=np.int64)
for si in range(3):
    cnt = YM[:, si].reshape(len(train), -1).sum(1)
    z = YB[cnt == 0, si]
    EMPTY_BIN[si] = int(np.bincount(z, minlength=10).argmax()) if len(z) else 0
    nz = YB[cnt > 0, si]
    NONEMPTY_MIN[si] = int(max(0, np.percentile(nz, 1))) if len(nz) else 0

from PIL import Image


def load_all(df):
    n = len(df)
    arr = np.zeros((n, IMG, IMG, 3), dtype=np.uint8)
    for i, p in enumerate(df.image_path.values):
        fp = os.path.join(ROOT, str(p).replace("\\", "/"))
        if not os.path.isfile(fp):
            fp = os.path.join(ROOT, "images", os.path.basename(str(p)))
        try:
            im = Image.open(fp).convert("RGB")
            if im.size != (IMG, IMG):
                im = im.resize((IMG, IMG), Image.BILINEAR)
            arr[i] = np.asarray(im, dtype=np.uint8)
        except Exception:
            pass
    return arr


XTR = load_all(train)
XTE = load_all(test)
print("loaded", XTR.shape, XTE.shape, "t=%.0f" % (time.time() - T0), flush=True)


def f1(pred, true):
    ps = pred.sum()
    ts = true.sum()
    if ps == 0 and ts == 0:
        return 1.0
    if ps == 0 or ts == 0:
        return 0.0
    inter = float((pred * true).sum())
    if inter == 0:
        return 0.0
    return 2.0 * inter / (ps + ts)


def profile_from(mask):
    prof = np.zeros(G, dtype=np.int64)
    for r in range(G):
        st = [si + 1 for si in range(3) if mask[si, r].sum() > 0]
        prof[r] = 0 if len(st) == 0 else (st[0] if len(st) == 1 else 4)
    return prof


def row_score(pm, pb, tm, tb):
    pa = (pm.sum(0) > 0).astype(np.float32)
    ta = (tm.sum(0) > 0).astype(np.float32)
    aff = f1(pa, ta)
    stt = np.mean([f1(pm[si], tm[si]) for si in range(3)])
    bn = np.mean([max(0.0, 1.0 - abs(int(pb[si]) - int(tb[si])) / 4.0) for si in range(3)])
    pp = profile_from(pm)
    tp = profile_from(tm)
    pr = float((pp == tp).mean())
    raw = 0.34 * aff + 0.30 * stt + 0.20 * bn + 0.16 * pr
    return raw ** 4.0


import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(SEED)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NCPU = max(1, (os.cpu_count() or 4))
try:
    torch.set_num_threads(min(NCPU, 10))
except Exception:
    pass

import torchvision


def backbone():
    fn = getattr(torchvision.models, ARCH)
    m = None
    for w in ("IMAGENET1K_V1", None):
        try:
            m = fn(weights=w)
            break
        except Exception:
            continue
    if m is None:
        m = fn(weights=None)
    ch = m.fc.in_features
    m.fc = nn.Identity()
    m.avgpool = nn.Identity()
    return m, ch


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        b, ch = backbone()
        self.stem = nn.Sequential(b.conv1, b.bn1, b.relu, b.maxpool, b.layer1, b.layer2, b.layer3, b.layer4)
        self.dec = nn.Sequential(nn.Conv2d(ch, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.cell = nn.Conv2d(256, 3, 1)
        self.binh = nn.Sequential(nn.Linear(ch + 256, 256), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.Linear(256, 30))

    def forward(self, x):
        f = self.stem(x)
        if f.shape[-1] != G:
            f = F.adaptive_avg_pool2d(f, G)
        d = self.dec(f)
        c = self.cell(d)
        g = torch.cat([F.adaptive_avg_pool2d(f, 1).flatten(1), F.adaptive_avg_pool2d(d, 1).flatten(1)], 1)
        return c, self.binh(g).view(-1, 3, 10)


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_tensor(a):
    t = torch.from_numpy(a).permute(0, 3, 1, 2).float().div_(255.0)
    return (t - MEAN) / STD


idx = np.arange(len(train))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
nv = max(60, int(len(train) * VAL_FRAC))
vidx = idx[:nv]
tidx = idx[nv:]

pos = YM.reshape(-1, 3, G * G).mean((0, 2))
pw = torch.tensor(np.clip((1 - pos) / np.maximum(pos, 1e-6), 1.0, 6.0), dtype=torch.float32).view(1, 3, 1, 1).to(DEV)

model = Net().to(DEV)
head_p, back_p = [], []
for n_, p_ in model.named_parameters():
    (head_p if n_.startswith(("dec", "cell", "binh")) else back_p).append(p_)
opt = torch.optim.AdamW([{"params": back_p, "lr": 3e-4}, {"params": head_p, "lr": 1e-3}], weight_decay=1e-4)
scaler = torch.cuda.amp.GradScaler(enabled=(DEV.type == "cuda"))

ntr = len(tidx)
steps = max(1, ntr // BS)


def run_epoch(ep, total_ep):
    model.train()
    order = np.random.permutation(tidx)
    tot = 0.0
    for s in range(steps):
        bi = order[s * BS:(s + 1) * BS]
        if len(bi) == 0:
            continue
        xb = XTR[bi].copy()
        mb = YM[bi].copy()
        fl = np.random.rand(len(bi)) < 0.5
        if fl.any():
            xb[fl] = xb[fl][:, :, ::-1, :]
            mb[fl] = mb[fl][:, :, :, ::-1]
        x = to_tensor(xb).to(DEV, non_blocking=True)
        if np.random.rand() < 0.5:
            x = x * float(np.random.uniform(0.85, 1.15))
        y = torch.from_numpy(mb).to(DEV)
        yb = torch.from_numpy(YB[bi]).to(DEV)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(DEV.type == "cuda")):
            c, b = model(x)
            loss = F.binary_cross_entropy_with_logits(c, y, pos_weight=pw)
            loss = loss + 0.3 * F.cross_entropy(b.reshape(-1, 10), yb.reshape(-1))
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        tot += float(loss.item())
    for gi, g_ in enumerate(opt.param_groups):
        base = 3e-4 if gi == 0 else 1e-3
        g_["lr"] = base * 0.5 * (1 + math.cos(math.pi * min(1.0, (ep + 1) / total_ep)))
    return tot / max(1, steps)


@torch.no_grad()
def predict(X):
    model.eval()
    P = np.zeros((len(X), 3, G, G), dtype=np.float32)
    B = np.zeros((len(X), 3, 10), dtype=np.float32)
    bs = max(8, BS)
    for s in range(0, len(X), bs):
        xb = X[s:s + bs]
        x = to_tensor(xb).to(DEV)
        with torch.cuda.amp.autocast(enabled=(DEV.type == "cuda")):
            c1, b1 = model(x)
            c2, b2 = model(torch.flip(x, dims=[3]))
        c = (torch.sigmoid(c1.float()) + torch.flip(torch.sigmoid(c2.float()), dims=[3])) / 2
        b = (F.softmax(b1.float(), -1) + F.softmax(b2.float(), -1)) / 2
        P[s:s + bs] = c.cpu().numpy()
        B[s:s + bs] = b.cpu().numpy()
    return P, B


t_load = time.time() - T0
epoch_budget = BUDGET - t_load - 60
ep = 0
tprev = time.time()
est = None
total_ep = MAX_EPOCH
while ep < MAX_EPOCH:
    el = time.time() - T0
    if est is not None and el + est * 1.15 > BUDGET - 90:
        break
    l = run_epoch(ep, total_ep)
    dt = time.time() - tprev
    tprev = time.time()
    est = dt if est is None else 0.5 * est + 0.5 * dt
    if ep == 0:
        rem = (BUDGET - 90) - (time.time() - T0)
        total_ep = int(max(3, min(MAX_EPOCH, 1 + rem / max(1e-6, dt))))
        print("planned epochs", total_ep, "sec/ep %.1f" % dt, flush=True)
    ep += 1
    print("ep %d loss %.4f dt %.1f elapsed %.0f" % (ep, l, dt, time.time() - T0), flush=True)
    if ep >= total_ep:
        break

PV, BV = predict(XTR[vidx])
TMV = YM[vidx]
TBV = YB[vidx]

BW = np.array([[max(0.0, 1.0 - abs(a - b) / 4.0) for b in range(10)] for a in range(10)], dtype=np.float32)


def decode_bins(bp, masks):
    out = np.zeros((len(bp), 3), dtype=np.int64)
    for i in range(len(bp)):
        for si in range(3):
            if masks[i, si].sum() == 0:
                out[i, si] = EMPTY_BIN[si]
            else:
                u = BW @ bp[i, si]
                lo = int(NONEMPTY_MIN[si])
                u = u.copy()
                u[:lo] = -1e9
                out[i, si] = int(u.argmax())
    return out


def eval_thr(thr, P, B, TM, TB):
    masks = (P > thr.reshape(1, 3, 1, 1)).astype(np.float32)
    pb = decode_bins(B, masks)
    return float(np.mean([row_score(masks[i], pb[i], TM[i], TB[i]) for i in range(len(P))]))


thr = np.array([0.5, 0.5, 0.5], dtype=np.float32)
best = eval_thr(thr, PV, BV, TMV, TBV)
grid = np.arange(0.10, 0.91, 0.025)
for it in range(3):
    for si in range(3):
        cur = thr[si]
        for g_ in grid:
            t2 = thr.copy()
            t2[si] = g_
            s = eval_thr(t2, PV, BV, TMV, TBV)
            if s > best + 1e-9:
                best = s
                cur = g_
        thr[si] = cur
    print("iter", it, "thr", thr.round(3), "val %.4f" % best, flush=True)

raw_parts = None
masks = (PV > thr.reshape(1, 3, 1, 1)).astype(np.float32)
pbv = decode_bins(BV, masks)
aff = np.mean([f1((masks[i].sum(0) > 0).astype(np.float32), (TMV[i].sum(0) > 0).astype(np.float32)) for i in range(len(PV))])
stt = np.mean([np.mean([f1(masks[i, s], TMV[i, s]) for s in range(3)]) for i in range(len(PV))])
bnn = np.mean([np.mean([max(0.0, 1 - abs(int(pbv[i, s]) - int(TBV[i, s])) / 4.0) for s in range(3)]) for i in range(len(PV))])
prf = np.mean([(profile_from(masks[i]) == profile_from(TMV[i])).mean() for i in range(len(PV))])
print("VAL aff %.4f state %.4f bin %.4f prof %.4f  raw %.4f  FINAL %.4f" % (
    aff, stt, bnn, prf, 0.34 * aff + 0.30 * stt + 0.20 * bnn + 0.16 * prf, best), flush=True)

PT, BT = predict(XTE)
mt = (PT > thr.reshape(1, 3, 1, 1)).astype(np.float32)
bt = decode_bins(BT, mt)

rows = []
for i, rid in enumerate(test.id.values):
    m = mt[i]
    cells = {st: [] for st in STATES}
    affc = []
    for r in range(G):
        for c in range(G):
            any_ = False
            for si, st in enumerate(STATES):
                if m[si, r, c] > 0:
                    cells[st].append(tok(r, c))
                    any_ = True
            if any_:
                affc.append(tok(r, c))
    prof = profile_from(m).tolist()
    ans = {
        "affected_cells": affc,
        "fair_cells": cells["fair"],
        "poor_cells": cells["poor"],
        "severe_cells": cells["severe"],
        "area_bins": {st: int(bt[i, si]) for si, st in enumerate(STATES)},
        "row_condition_profile": [int(v) for v in prof],
    }
    rows.append({"id": rid, "answer_json": json.dumps(ans, separators=(",", ":"))})

sub = pd.DataFrame(rows, columns=["id", "answer_json"])
d = os.path.dirname(os.path.abspath(OUT))
if d:
    os.makedirs(d, exist_ok=True)
sub.to_csv(OUT, index=False)
print("wrote", OUT, sub.shape, "total %.0fs" % (time.time() - T0), flush=True)

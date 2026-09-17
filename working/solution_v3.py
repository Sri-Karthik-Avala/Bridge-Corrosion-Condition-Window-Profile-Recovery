# made by - Karthik
import os, sys, json, time, math, random
import numpy as np
import pandas as pd

T0 = time.time()
BUDGET = float(os.environ.get("ERIS_BUDGET", "3300"))
SEED = int(os.environ.get("ERIS_SEED", "17"))
IMG = int(os.environ.get("ERIS_IMG", "256"))
BS = int(os.environ.get("ERIS_BS", "32"))
NFOLD = int(os.environ.get("ERIS_NFOLD", "3"))
MIN_EPOCH = int(os.environ.get("ERIS_MIN_EPOCH", "25"))
MAX_EPOCH = int(os.environ.get("ERIS_MAX_EPOCH", "110"))
WIDTH = int(os.environ.get("ERIS_WIDTH", "64"))
RESERVE = float(os.environ.get("ERIS_RESERVE", "150"))

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
    bins = np.clip(np.array([int(b.get(st, 0)) for st in STATES], dtype=np.int64), 0, 9)
    return m, bins


YM = np.zeros((len(train), 3, G, G), dtype=np.float32)
YB = np.zeros((len(train), 3), dtype=np.int64)
for i, s in enumerate(train.answer_json.values):
    m, b = parse_answer(s)
    YM[i] = m
    YB[i] = b

EMPTY_BIN = np.zeros(3, dtype=np.int64)
NONEMPTY_MIN = np.zeros(3, dtype=np.int64)
for si in range(3):
    cnt = YM[:, si].reshape(len(train), -1).sum(1)
    z = YB[cnt == 0, si]
    EMPTY_BIN[si] = int(np.bincount(z, minlength=10).argmax()) if len(z) else 0
    nz = YB[cnt > 0, si]
    NONEMPTY_MIN[si] = int(max(0, np.percentile(nz, 1))) if len(nz) else 0

from PIL import Image

NCH = 6


def derive(a):
    f = a.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    s = r + g + b + 1.0
    mx = f.max(-1)
    mn = f.min(-1)
    d0 = (r - g) / s
    d1 = (r - b) / s
    d2 = (mx - mn) / (mx + 1.0)
    out = np.stack([d0, d1, d2], -1)
    return np.clip(out * 255.0 + 128.0, 0, 255).astype(np.uint8)


def load_all(df):
    n = len(df)
    arr = np.zeros((n, IMG, IMG, NCH), dtype=np.uint8)
    for i, p in enumerate(df.image_path.values):
        fp = os.path.join(ROOT, str(p).replace("\\", "/"))
        if not os.path.isfile(fp):
            fp = os.path.join(ROOT, "images", os.path.basename(str(p)))
        try:
            im = Image.open(fp).convert("RGB")
            if im.size != (IMG, IMG):
                im = im.resize((IMG, IMG), Image.BILINEAR)
            a = np.asarray(im, dtype=np.uint8)
            arr[i, :, :, :3] = a
            arr[i, :, :, 3:] = derive(a)
        except Exception:
            pass
    return arr


XTR = load_all(train)
XTE = load_all(test)
print("loaded", XTR.shape, XTE.shape, "t=%.0f" % (time.time() - T0), flush=True)

_s = XTR[:: max(1, len(XTR) // 300)].astype(np.float32) / 255.0
CH_MEAN = _s.mean((0, 1, 2)).astype(np.float32)
CH_STD = np.maximum(_s.std((0, 1, 2)), 1e-3).astype(np.float32)
del _s


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
    aff = f1((pm.sum(0) > 0).astype(np.float32), (tm.sum(0) > 0).astype(np.float32))
    stt = np.mean([f1(pm[si], tm[si]) for si in range(3)])
    bn = np.mean([max(0.0, 1.0 - abs(int(pb[si]) - int(tb[si])) / 4.0) for si in range(3)])
    pr = float((profile_from(pm) == profile_from(tm)).mean())
    return (0.34 * aff + 0.30 * stt + 0.20 * bn + 0.16 * pr) ** 4.0


import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(SEED)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    torch.set_num_threads(min(max(1, os.cpu_count() or 4), 10))
    torch.backends.cudnn.benchmark = True
except Exception:
    pass


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.sc = None
        if stride != 1 or cin != cout:
            self.sc = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        r = x if self.sc is None else self.sc(x)
        x = F.relu(self.b1(self.c1(x)), inplace=True)
        x = self.b2(self.c2(x))
        return F.relu(x + r, inplace=True)


class Net(nn.Module):
    def __init__(self, w=WIDTH):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(NCH, w // 2, 5, 2, 2, bias=False), nn.BatchNorm2d(w // 2), nn.ReLU(inplace=True),
            nn.Conv2d(w // 2, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        ch = [w, w * 2, w * 4, w * 6]
        self.s1 = nn.Sequential(Block(w, ch[0], 1), Block(ch[0], ch[0], 1))
        self.s2 = nn.Sequential(Block(ch[0], ch[1], 2), Block(ch[1], ch[1], 1))
        self.s3 = nn.Sequential(Block(ch[1], ch[2], 2), Block(ch[2], ch[2], 1))
        self.s4 = nn.Sequential(Block(ch[2], ch[3], 2), Block(ch[3], ch[3], 1))
        self.dec = nn.Sequential(nn.Conv2d(ch[3], 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.drop = nn.Dropout2d(0.1)
        self.cell = nn.Conv2d(256, 3, 1)
        self.binh = nn.Sequential(nn.Linear(ch[3] + 256, 256), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(256, 30))

    def forward(self, x):
        x = self.stem(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        f = self.s4(x)
        if f.shape[-1] != G:
            f = F.adaptive_avg_pool2d(f, G)
        d = self.dec(f)
        c = self.cell(self.drop(d))
        g = torch.cat([F.adaptive_avg_pool2d(f, 1).flatten(1), F.adaptive_avg_pool2d(d, 1).flatten(1)], 1)
        return c, self.binh(g).view(-1, 3, 10)


MEAN = torch.tensor(CH_MEAN).view(1, NCH, 1, 1)
STD = torch.tensor(CH_STD).view(1, NCH, 1, 1)


def to_tensor(a):
    t = torch.from_numpy(np.ascontiguousarray(a)).permute(0, 3, 1, 2).float().div_(255.0)
    return (t - MEAN) / STD


pos = YM.reshape(-1, 3, G * G).mean((0, 2))
PW = torch.tensor(np.clip((1 - pos) / np.maximum(pos, 1e-6), 1.0, 6.0), dtype=torch.float32).view(1, 3, 1, 1).to(DEV)


def soft_dice(logit, y):
    p = torch.sigmoid(logit).flatten(2)
    t = y.flatten(2)
    num = 2.0 * (p * t).sum(-1) + 1.0
    den = p.sum(-1) + t.sum(-1) + 1.0
    return (1.0 - num / den).mean()


def augment(xb, mb):
    h = np.random.rand(len(xb)) < 0.5
    if h.any():
        xb[h] = xb[h][:, :, ::-1, :]
        mb[h] = mb[h][:, :, :, ::-1]
    return xb, mb


def train_fold(tr_idx, epochs, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Net().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=5e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEV.type == "cuda"))
    steps = max(1, len(tr_idx) // BS)
    total = epochs * steps
    it = 0
    for ep in range(epochs):
        model.train()
        order = np.random.permutation(tr_idx)
        tot = 0.0
        for s in range(steps):
            bi = order[s * BS:(s + 1) * BS]
            if len(bi) < 2:
                continue
            xb, mb = augment(XTR[bi].copy(), YM[bi].copy())
            x = to_tensor(xb).to(DEV, non_blocking=True)
            gain = float(np.random.uniform(0.8, 1.2))
            bias = float(np.random.uniform(-0.15, 0.15))
            x = x * gain + bias
            y = torch.from_numpy(np.ascontiguousarray(mb)).to(DEV)
            yb = torch.from_numpy(YB[bi]).to(DEV)
            it += 1
            lr = 3e-3 * min(1.0, it / max(1, int(0.05 * total))) * 0.5 * (1 + math.cos(math.pi * it / max(1, total)))
            for g_ in opt.param_groups:
                g_["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(DEV.type == "cuda")):
                c, b = model(x)
                loss = F.binary_cross_entropy_with_logits(c, y, pos_weight=PW)
                loss = loss + 0.5 * soft_dice(c.float(), y) + 0.3 * F.cross_entropy(b.reshape(-1, 10), yb.reshape(-1))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss.item())
        if ep == epochs - 1 or ep % 5 == 0:
            print("   ep %d/%d loss %.4f t=%.0f" % (ep + 1, epochs, tot / max(1, steps), time.time() - T0), flush=True)
    return model


@torch.no_grad()
def predict(model, X, rows=None):
    model.eval()
    n = len(X) if rows is None else len(rows)
    P = np.zeros((n, 3, G, G), dtype=np.float32)
    B = np.zeros((n, 3, 10), dtype=np.float32)
    bs = 32
    for s in range(0, n, bs):
        sel = slice(s, s + bs) if rows is None else rows[s:s + bs]
        xb = X[sel]
        x = to_tensor(xb).to(DEV)
        acc_c = 0
        acc_b = 0
        for fh in (False, True):
            xx = torch.flip(x, dims=[3]) if fh else x
            with torch.cuda.amp.autocast(enabled=(DEV.type == "cuda")):
                c, b = model(xx)
            c = torch.sigmoid(c.float())
            if fh:
                c = torch.flip(c, dims=[3])
            acc_c = acc_c + c
            acc_b = acc_b + F.softmax(b.float(), -1)
        P[s:s + len(xb)] = (acc_c / 2).cpu().numpy()
        B[s:s + len(xb)] = (acc_b / 2).cpu().numpy()
    return P, B


rng = np.random.RandomState(SEED)
order = rng.permutation(len(train))
folds = np.array_split(order, NFOLD)

probe_idx = folds[0]
tr0 = np.concatenate([folds[i] for i in range(1, NFOLD)])
t_probe = time.time()
_ = train_fold(tr0[:BS * 8], 1, SEED)
sec_ep_unit = (time.time() - t_probe) / max(1, (len(tr0[:BS * 8]) // BS))
sec_ep = sec_ep_unit * max(1, len(tr0) // BS)
avail = BUDGET - (time.time() - T0) - RESERVE
tot_ep = int(max(MIN_EPOCH, avail / max(1e-6, sec_ep)))
nf = int(min(NFOLD, max(1, tot_ep // MIN_EPOCH)))
epochs = int(min(MAX_EPOCH, max(MIN_EPOCH, tot_ep // nf)))
print("sec/ep %.1f avail %.0f -> folds %d epochs %d" % (sec_ep, avail, nf, epochs), flush=True)

models = []
val_of = []
for f in range(nf):
    el = time.time() - T0
    left = BUDGET - el - RESERVE - (nf - f - 1) * 0
    if f > 0 and left < sec_ep * MIN_EPOCH:
        print("stop folds early at", f, flush=True)
        break
    ep_f = epochs
    rem_folds = nf - f
    aff_ep = int((BUDGET - el - RESERVE) / max(1e-6, sec_ep * rem_folds))
    ep_f = int(min(MAX_EPOCH, max(3, min(epochs, aff_ep))))
    tr_idx = np.concatenate([folds[i] for i in range(NFOLD) if i != f])
    print("fold %d epochs %d n=%d t=%.0f" % (f, ep_f, len(tr_idx), time.time() - T0), flush=True)
    models.append(train_fold(tr_idx, ep_f, SEED + 100 * f))
    val_of.append(folds[f])

K = len(models)
OP = np.zeros((len(train), 3, G, G), dtype=np.float32)
OB = np.zeros((len(train), 3, 10), dtype=np.float32)
cov = np.zeros(len(train), dtype=np.int64)
for f in range(K):
    vi = val_of[f]
    acc_p = np.zeros((len(vi), 3, G, G), dtype=np.float32)
    acc_b = np.zeros((len(vi), 3, 10), dtype=np.float32)
    m_used = 0
    for g in range(K):
        if g == f:
            continue
        p, b = predict(models[g], XTR, vi)
        acc_p += p
        acc_b += b
        m_used += 1
    if m_used == 0:
        p, b = predict(models[f], XTR, vi)
        acc_p, acc_b, m_used = p, b, 1
    OP[vi] = acc_p / m_used
    OB[vi] = acc_b / m_used
    cov[vi] = 1
oof = np.where(cov > 0)[0]
print("oof rows", len(oof), "t=%.0f" % (time.time() - T0), flush=True)

BW = np.array([[max(0.0, 1.0 - abs(a - b) / 4.0) for b in range(10)] for a in range(10)], dtype=np.float32)


def decode_bins(bp, masks):
    out = np.zeros((len(bp), 3), dtype=np.int64)
    for i in range(len(bp)):
        for si in range(3):
            if masks[i, si].sum() == 0:
                out[i, si] = EMPTY_BIN[si]
            else:
                u = (BW @ bp[i, si]).copy()
                u[:int(NONEMPTY_MIN[si])] = -1e9
                out[i, si] = int(u.argmax())
    return out


def eval_thr(thr, P, B, TM, TB):
    masks = (P > thr.reshape(1, 3, 1, 1)).astype(np.float32)
    pb = decode_bins(B, masks)
    return float(np.mean([row_score(masks[i], pb[i], TM[i], TB[i]) for i in range(len(P))]))


PV, BV, TMV, TBV = OP[oof], OB[oof], YM[oof], YB[oof]
thr = np.array([0.5, 0.5, 0.5], dtype=np.float32)
best = eval_thr(thr, PV, BV, TMV, TBV)
grid = np.arange(0.10, 0.91, 0.025)
for it in range(3):
    improved = False
    for si in range(3):
        cur = thr[si]
        for g_ in grid:
            t2 = thr.copy()
            t2[si] = g_
            s = eval_thr(t2, PV, BV, TMV, TBV)
            if s > best + 1e-9:
                best = s
                cur = g_
                improved = True
        thr[si] = cur
    print("iter", it, "thr", thr.round(3), "oof %.4f" % best, flush=True)
    if not improved:
        break

masks = (PV > thr.reshape(1, 3, 1, 1)).astype(np.float32)
pbv = decode_bins(BV, masks)
n = len(PV)
aff = np.mean([f1((masks[i].sum(0) > 0).astype(np.float32), (TMV[i].sum(0) > 0).astype(np.float32)) for i in range(n)])
stt = np.mean([np.mean([f1(masks[i, s], TMV[i, s]) for s in range(3)]) for i in range(n)])
bnn = np.mean([np.mean([max(0.0, 1 - abs(int(pbv[i, s]) - int(TBV[i, s])) / 4.0) for s in range(3)]) for i in range(n)])
prf = np.mean([(profile_from(masks[i]) == profile_from(TMV[i])).mean() for i in range(n)])
print("OOF aff %.4f state %.4f bin %.4f prof %.4f raw %.4f FINAL %.4f" % (
    aff, stt, bnn, prf, 0.34 * aff + 0.30 * stt + 0.20 * bnn + 0.16 * prf, best), flush=True)

PT = np.zeros((len(test), 3, G, G), dtype=np.float32)
BT = np.zeros((len(test), 3, 10), dtype=np.float32)
for m in models:
    p, b = predict(m, XTE)
    PT += p
    BT += b
PT /= max(1, len(models))
BT /= max(1, len(models))

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
    ans = {
        "affected_cells": affc,
        "fair_cells": cells["fair"],
        "poor_cells": cells["poor"],
        "severe_cells": cells["severe"],
        "area_bins": {st: int(bt[i, si]) for si, st in enumerate(STATES)},
        "row_condition_profile": [int(v) for v in profile_from(m)],
    }
    rows.append({"id": rid, "answer_json": json.dumps(ans, separators=(",", ":"))})

sub = pd.DataFrame(rows, columns=["id", "answer_json"])
d = os.path.dirname(os.path.abspath(OUT))
if d:
    os.makedirs(d, exist_ok=True)
sub.to_csv(OUT, index=False)
print("wrote", OUT, sub.shape, "models", len(models), "total %.0fs" % (time.time() - T0), flush=True)

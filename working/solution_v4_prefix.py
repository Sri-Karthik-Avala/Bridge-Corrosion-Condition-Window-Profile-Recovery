# made by - Karthik
import os, sys, json, time, math, random
import numpy as np
import pandas as pd

T0 = time.time()
BUDGET = float(os.environ.get("ERIS_BUDGET", "3300"))
SEED = int(os.environ.get("ERIS_SEED", "17"))
IMG = int(os.environ.get("ERIS_IMG", "256"))
BS = int(os.environ.get("ERIS_BS", "24"))
NFOLD = int(os.environ.get("ERIS_NFOLD", "4"))
MIN_EPOCH = int(os.environ.get("ERIS_MIN_EPOCH", "20"))
MAX_EPOCH = int(os.environ.get("ERIS_MAX_EPOCH", "60"))
WIDTH = int(os.environ.get("ERIS_WIDTH", "48"))
RESERVE = float(os.environ.get("ERIS_RESERVE", "180"))

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
YM = np.zeros((len(train), 3, G, G), dtype=np.float32)
YB = np.zeros((len(train), 3), dtype=np.int64)
for i, s in enumerate(train.answer_json.values):
    a = json.loads(s)
    for si, st in enumerate(STATES):
        for t in a.get(st + "_cells", []):
            j = TIDX.get(t)
            if j is not None:
                YM[i, si, j // G, j % G] = 1.0
    b = a.get("area_bins", {})
    YB[i] = np.clip([int(b.get(st, 0)) for st in STATES], 0, 9)

CNT = YM.reshape(len(train), 3, -1).sum(-1).astype(int)
CMAP = np.zeros((3, G * G + 1), dtype=np.int64)
for k in range(3):
    last = 0
    for c in range(G * G + 1):
        m = CNT[:, k] == c
        if m.sum() >= 3:
            last = int(np.median(YB[m, k]))
        CMAP[k, c] = last
EMPTY_BIN = np.zeros(3, dtype=np.int64)
NONEMPTY_MIN = np.zeros(3, dtype=np.int64)
for k in range(3):
    z = YB[CNT[:, k] == 0, k]
    EMPTY_BIN[k] = int(np.bincount(z, minlength=10).argmax()) if len(z) else 0
    nz = YB[CNT[:, k] > 0, k]
    NONEMPTY_MIN[k] = int(max(0, np.percentile(nz, 1))) if len(nz) else 0

from PIL import Image

NCH = 6
NFEAT = 16


def derive(a):
    f = a.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    s = r + g + b + 1.0
    d = np.stack([(r - g) / s, (r - b) / s, (f.max(-1) - f.min(-1)) / (f.max(-1) + 1.0)], -1)
    return np.clip(d * 255.0 + 128.0, 0, 255).astype(np.uint8)


def cell_feats(a):
    h, w = a.shape[:2]
    ch, cw = max(1, h // G), max(1, w // G)
    out = np.zeros((NFEAT, G, G), np.float32)
    f = a.astype(np.float32)
    R, Gc, B = f[..., 0], f[..., 1], f[..., 2]
    s = R + Gc + B + 1.0
    rg = (R - Gc) / s
    rb = (R - B) / s
    sat = f.max(-1) - f.min(-1)
    gray = f.mean(-1)
    ed = np.zeros_like(gray)
    ed[1:] = np.abs(np.diff(gray, axis=0))
    for r in range(G):
        for c in range(G):
            sl = (slice(r * ch, (r + 1) * ch), slice(c * cw, (c + 1) * cw))
            bl = gray[sl]
            q = rg[sl]
            if q.size == 0:
                continue
            out[:, r, c] = [bl.mean(), bl.std(), R[sl].mean(), Gc[sl].mean(), B[sl].mean(),
                            q.mean(), rb[sl].mean(), q.std(), sat[sl].mean(),
                            np.percentile(q, 90), np.percentile(q, 99), np.percentile(rb[sl], 90),
                            ed[sl].mean(), float((q > 0.05).mean()), float((q > 0.10).mean()), sat[sl].std()]
    return out


def load_all(df):
    n = len(df)
    arr = np.zeros((n, IMG, IMG, 3), dtype=np.uint8)
    ft = np.zeros((n, NFEAT, G, G), dtype=np.float32)
    for i, p in enumerate(df.image_path.values):
        fp = os.path.join(ROOT, str(p).replace("\\", "/"))
        if not os.path.isfile(fp):
            fp = os.path.join(ROOT, "images", os.path.basename(str(p)))
        try:
            im = Image.open(fp).convert("RGB")
            ft[i] = cell_feats(np.asarray(im, dtype=np.uint8))
            if im.size != (IMG, IMG):
                im = im.resize((IMG, IMG), Image.BILINEAR)
            arr[i] = np.asarray(im, dtype=np.uint8)
        except Exception:
            pass
    return arr, ft


XTR, FTR = load_all(train)
XTE, FTE = load_all(test)
print("loaded", XTR.shape, FTR.shape, "t=%.0f" % (time.time() - T0), flush=True)

FM = FTR.mean((0, 2, 3), keepdims=True)
FS = np.maximum(FTR.std((0, 2, 3), keepdims=True), 1e-3)
FTR = (FTR - FM) / FS
FTE = (FTE - FM) / FS
_s = XTR[:: max(1, len(XTR) // 200)].astype(np.float32) / 255.0
_r, _g, _b = _s[..., 0], _s[..., 1], _s[..., 2]
_sum = _r + _g + _b + 1e-3
_mx = _s.max(-1); _mn = _s.min(-1)
_d = np.stack([(_r - _g) / _sum, (_r - _b) / _sum, (_mx - _mn) / (_mx + 1e-3)], -1)
_a6 = np.concatenate([_s, _d], -1)
CH_MEAN = _a6.mean((0, 1, 2)).astype(np.float32)
CH_STD = np.maximum(_a6.std((0, 1, 2)), 1e-3).astype(np.float32)
del _s, _d, _a6


def f1(pred, true):
    ps, ts = pred.sum(), true.sum()
    if ps == 0 and ts == 0:
        return 1.0
    if ps == 0 or ts == 0:
        return 0.0
    return 2.0 * float((pred * true).sum()) / (ps + ts)


def profile_from(mask):
    o = np.zeros(G, dtype=np.int64)
    for r in range(G):
        st = [k + 1 for k in range(3) if mask[k, r].sum() > 0]
        o[r] = 0 if not st else (st[0] if len(st) == 1 else 4)
    return o


def row_score(pm, pb, tm, tb):
    aff = f1((pm.sum(0) > 0).astype(np.float32), (tm.sum(0) > 0).astype(np.float32))
    stt = np.mean([f1(pm[k], tm[k]) for k in range(3)])
    bn = np.mean([max(0.0, 1.0 - abs(int(pb[k]) - int(tb[k])) / 4.0) for k in range(3)])
    pr = float((profile_from(pm) == profile_from(tm)).mean())
    return (0.34 * aff + 0.30 * stt + 0.20 * bn + 0.16 * pr) ** 4.0


import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(SEED)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_frac = float(os.environ.get("ERIS_GPU_FRAC", "0"))
if DEV.type == "cuda" and _frac > 0:
    try:
        torch.cuda.set_per_process_memory_fraction(_frac)
    except Exception:
        pass
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
        return F.relu(self.b2(self.c2(x)) + r, inplace=True)


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
        self.u3 = nn.Sequential(nn.Conv2d(ch[3] + ch[2], ch[2], 3, padding=1, bias=False), nn.BatchNorm2d(ch[2]), nn.ReLU(inplace=True))
        self.u2 = nn.Sequential(nn.Conv2d(ch[2] + ch[1], ch[1], 3, padding=1, bias=False), nn.BatchNorm2d(ch[1]), nn.ReLU(inplace=True))
        self.drop = nn.Dropout(0.1)
        self.fine = nn.Conv2d(ch[1], 3, 1)
        self.fuse = nn.Sequential(nn.Conv2d(ch[1] + NFEAT + 3, 96, 1), nn.ReLU(inplace=True), nn.Conv2d(96, 3, 1))
        self.binh = nn.Sequential(nn.Linear(ch[3] + NFEAT + 3, 256), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(256, 30))

    def forward(self, x, cf):
        x0 = self.stem(x)
        a1 = self.s1(x0)
        a2 = self.s2(a1)
        a3 = self.s3(a2)
        a4 = self.s4(a3)
        u = self.u3(torch.cat([F.interpolate(a4, size=a3.shape[-2:], mode="nearest"), a3], 1))
        u = self.u2(torch.cat([F.interpolate(u, size=a2.shape[-2:], mode="nearest"), a2], 1))
        fine = self.fine(self.drop(u))
        B_, C_, H_, W_ = fine.shape
        if H_ % G or W_ % G:
            fine = F.interpolate(fine, size=(G * max(1, H_ // G), G * max(1, W_ // G)), mode="bilinear", align_corners=False)
            B_, C_, H_, W_ = fine.shape
        ph, pw = H_ // G, W_ // G
        blk = fine.view(B_, C_, G, ph, G, pw).permute(0, 1, 2, 4, 3, 5).reshape(B_, C_, G, G, ph * pw)
        rr = 5.0
        mil = torch.logsumexp(blk * rr, -1) / rr - math.log(ph * pw) / rr
        pooled = F.adaptive_avg_pool2d(u, G)
        cell = mil + self.fuse(torch.cat([pooled, cf, mil], 1))
        gf = torch.cat([F.adaptive_avg_pool2d(a4, 1).flatten(1), cf.mean((2, 3)), torch.sigmoid(cell).mean((2, 3))], 1)
        return cell, self.binh(gf).view(-1, 3, 10)


MEAN = torch.tensor(CH_MEAN).view(1, NCH, 1, 1)
STD = torch.tensor(CH_STD).view(1, NCH, 1, 1)


def to_tensor(a):
    t = torch.from_numpy(np.ascontiguousarray(a)).permute(0, 3, 1, 2).float().div_(255.0)
    r, g, b = t[:, 0:1], t[:, 1:2], t[:, 2:3]
    sm = r + g + b + 1e-3
    mx = t.max(1, keepdim=True)[0]
    mn = t.min(1, keepdim=True)[0]
    t = torch.cat([t, (r - g) / sm, (r - b) / sm, (mx - mn) / (mx + 1e-3)], 1)
    return (t - MEAN) / STD


pos = YM.reshape(-1, 3, G * G).mean((0, 2))
PW = torch.tensor(np.clip((1 - pos) / np.maximum(pos, 1e-6), 1.0, 6.0), dtype=torch.float32).view(1, 3, 1, 1).to(DEV)


def soft_dice(logit, y):
    p = torch.sigmoid(logit).flatten(2)
    t = y.flatten(2)
    return (1.0 - (2.0 * (p * t).sum(-1) + 1.0) / (p.sum(-1) + t.sum(-1) + 1.0)).mean()


SCALE = [1.0]


def prep(x):
    if SCALE[0] != 1.0:
        n = int(round(IMG * SCALE[0] / 8.0)) * 8
        x = F.interpolate(x, size=(n, n), mode="bilinear", align_corners=False)
    return x


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
            xb = XTR[bi].copy()
            mb = YM[bi].copy()
            fb = FTR[bi].copy()
            h = np.random.rand(len(bi)) < 0.5
            if h.any():
                xb[h] = xb[h][:, :, ::-1, :]
                mb[h] = mb[h][:, :, :, ::-1]
                fb[h] = fb[h][:, :, :, ::-1]
            x = prep(to_tensor(xb).to(DEV, non_blocking=True))
            x = x * float(np.random.uniform(0.85, 1.15)) + float(np.random.uniform(-0.12, 0.12))
            cf = torch.from_numpy(np.ascontiguousarray(fb)).to(DEV)
            y = torch.from_numpy(np.ascontiguousarray(mb)).to(DEV)
            yb = torch.from_numpy(YB[bi]).to(DEV)
            it += 1
            for g_ in opt.param_groups:
                g_["lr"] = 3e-3 * min(1.0, it / max(1, int(0.05 * total))) * 0.5 * (1 + math.cos(math.pi * it / max(1, total)))
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(DEV.type == "cuda")):
                c, b = model(x, cf)
                loss = F.binary_cross_entropy_with_logits(c, y, pos_weight=PW) + 0.5 * soft_dice(c.float(), y) \
                    + 0.3 * F.cross_entropy(b.reshape(-1, 10), yb.reshape(-1))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss.item())
        if ep % 8 == 0 or ep == epochs - 1:
            print("   ep %d/%d loss %.4f t=%.0f" % (ep + 1, epochs, tot / max(1, steps), time.time() - T0), flush=True)
    return model


@torch.no_grad()
def predict(model, X, FE, rows=None):
    model.eval()
    n = len(X) if rows is None else len(rows)
    P = np.zeros((n, 3, G, G), np.float32)
    B = np.zeros((n, 3, 10), np.float32)
    bs = 24
    for s in range(0, n, bs):
        sel = slice(s, s + bs) if rows is None else rows[s:s + bs]
        xb, fb = X[sel], FE[sel]
        x = prep(to_tensor(xb).to(DEV))
        cf = torch.from_numpy(np.ascontiguousarray(fb)).to(DEV)
        ac = 0
        ab = 0
        for fh in (False, True):
            xx = torch.flip(x, dims=[3]) if fh else x
            ff = torch.flip(cf, dims=[3]) if fh else cf
            with torch.cuda.amp.autocast(enabled=(DEV.type == "cuda")):
                c, b = model(xx, ff)
            c = torch.sigmoid(c.float())
            if fh:
                c = torch.flip(c, dims=[3])
            ac = ac + c
            ab = ab + F.softmax(b.float(), -1)
        P[s:s + len(xb)] = (ac / 2).cpu().numpy()
        B[s:s + len(xb)] = (ab / 2).cpu().numpy()
    return P, B


rng = np.random.RandomState(SEED)
order = rng.permutation(len(train))
folds = np.array_split(order, NFOLD)
tr0 = np.concatenate([folds[i] for i in range(1, NFOLD)])
tp = time.time()
_probe = train_fold(tr0[:BS * 6], 1, SEED)
del _probe
unit = (time.time() - tp) / 6.0
sec_ep = unit * max(1, len(tr0) // BS)
avail = BUDGET - (time.time() - T0) - RESERVE
if avail / max(1e-6, sec_ep) < MIN_EPOCH:
    SCALE[0] = 0.7778
    sec_ep *= 0.62
    print("slow host -> train scale 224", flush=True)
if avail / max(1e-6, sec_ep) < MIN_EPOCH:
    SCALE[0] = 0.5556
    sec_ep *= 0.52
    print("very slow host -> train scale 160", flush=True)
tot_ep = int(max(MIN_EPOCH, avail / max(1e-6, sec_ep)))
nf = int(min(NFOLD, max(1, tot_ep // MIN_EPOCH)))
epochs = int(min(MAX_EPOCH, max(MIN_EPOCH, tot_ep // nf)))
print("sec/ep %.1f avail %.0f -> folds %d epochs %d scale %.3f" % (sec_ep, avail, nf, epochs, SCALE[0]), flush=True)

models, val_of = [], []
for f in range(nf):
    el = time.time() - T0
    rem = nf - f
    ep_f = int(min(MAX_EPOCH, max(3, min(epochs, (BUDGET - el - RESERVE) / max(1e-6, sec_ep * rem)))))
    if f > 0 and (BUDGET - el - RESERVE) < sec_ep * 3:
        print("stop folds early at", f, flush=True)
        break
    tri = np.concatenate([folds[i] for i in range(NFOLD) if i != f])
    print("fold %d epochs %d n=%d t=%.0f" % (f, ep_f, len(tri), time.time() - T0), flush=True)
    models.append(train_fold(tri, ep_f, SEED + 100 * f))
    val_of.append(folds[f])

K = len(models)
OP = np.zeros((len(train), 3, G, G), np.float32)
OB = np.zeros((len(train), 3, 10), np.float32)
cov = np.zeros(len(train), np.int64)
for f in range(K):
    vi = val_of[f]
    ap = np.zeros((len(vi), 3, G, G), np.float32)
    ab = np.zeros((len(vi), 3, 10), np.float32)
    u = 0
    for g in range(K):
        if g == f:
            continue
        p, b = predict(models[g], XTR, FTR, vi)
        ap += p
        ab += b
        u += 1
    if u == 0:
        ap, ab = predict(models[f], XTR, FTR, vi)
        u = 1
    OP[vi] = ap / u
    OB[vi] = ab / u
    cov[vi] = 1
oof = np.where(cov > 0)[0]
print("oof rows", len(oof), "t=%.0f" % (time.time() - T0), flush=True)

BW = np.array([[max(0.0, 1.0 - abs(a - b) / 4.0) for b in range(10)] for a in range(10)], np.float32)


def decode_bins(bp, masks, w):
    out = np.zeros((len(bp), 3), np.int64)
    for i in range(len(bp)):
        for k in range(3):
            n = int(masks[i, k].sum())
            if n == 0:
                out[i, k] = EMPTY_BIN[k]
                continue
            u = ((1.0 - w) * (BW @ bp[i, k]) + w * BW[:, CMAP[k, min(G * G, n)]]).copy()
            u[:int(NONEMPTY_MIN[k])] = -1e9
            out[i, k] = int(u.argmax())
    return out


def eval_cfg(thr, w, P, B, TM, TB):
    m = (P > thr.reshape(1, 3, 1, 1)).astype(np.float32)
    pb = decode_bins(B, m, w)
    return float(np.mean([row_score(m[i], pb[i], TM[i], TB[i]) for i in range(len(P))]))


PV, BV, TMV, TBV = OP[oof], OB[oof], YM[oof], YB[oof]
thr = np.array([0.5, 0.5, 0.5], np.float32)
wbin = 0.0
best = eval_cfg(thr, wbin, PV, BV, TMV, TBV)
for it in range(3):
    ch = False
    for k in range(3):
        for g_ in np.arange(0.10, 0.91, 0.025):
            t2 = thr.copy()
            t2[k] = g_
            v = eval_cfg(t2, wbin, PV, BV, TMV, TBV)
            if v > best + 1e-9:
                best, thr[k], ch = v, g_, True
    for w_ in np.arange(0.0, 1.01, 0.1):
        v = eval_cfg(thr, w_, PV, BV, TMV, TBV)
        if v > best + 1e-9:
            best, wbin, ch = v, float(w_), True
    print("iter", it, "thr", thr.round(3), "wbin %.2f" % wbin, "oof %.4f" % best, flush=True)
    if not ch:
        break

m = (PV > thr.reshape(1, 3, 1, 1)).astype(np.float32)
pbv = decode_bins(BV, m, wbin)
n = len(PV)
aff = np.mean([f1((m[i].sum(0) > 0).astype(np.float32), (TMV[i].sum(0) > 0).astype(np.float32)) for i in range(n)])
stt = np.mean([np.mean([f1(m[i, k], TMV[i, k]) for k in range(3)]) for i in range(n)])
bnn = np.mean([np.mean([max(0.0, 1 - abs(int(pbv[i, k]) - int(TBV[i, k])) / 4.0) for k in range(3)]) for i in range(n)])
prf = np.mean([(profile_from(m[i]) == profile_from(TMV[i])).mean() for i in range(n)])
print("OOF aff %.4f state %.4f bin %.4f prof %.4f raw %.4f FINAL %.4f" % (
    aff, stt, bnn, prf, 0.34 * aff + 0.30 * stt + 0.20 * bnn + 0.16 * prf, best), flush=True)

PT = np.zeros((len(test), 3, G, G), np.float32)
BT = np.zeros((len(test), 3, 10), np.float32)
for mo in models:
    p, b = predict(mo, XTE, FTE)
    PT += p
    BT += b
PT /= max(1, len(models))
BT /= max(1, len(models))
mt = (PT > thr.reshape(1, 3, 1, 1)).astype(np.float32)
bt = decode_bins(BT, mt, wbin)

rows = []
for i, rid in enumerate(test.id.values):
    mm = mt[i]
    cells = {st: [] for st in STATES}
    affc = []
    for r in range(G):
        for c in range(G):
            hit = False
            for si, st in enumerate(STATES):
                if mm[si, r, c] > 0:
                    cells[st].append(tok(r, c))
                    hit = True
            if hit:
                affc.append(tok(r, c))
    rows.append({"id": rid, "answer_json": json.dumps({
        "affected_cells": affc, "fair_cells": cells["fair"], "poor_cells": cells["poor"],
        "severe_cells": cells["severe"],
        "area_bins": {st: int(bt[i, si]) for si, st in enumerate(STATES)},
        "row_condition_profile": [int(v) for v in profile_from(mm)]}, separators=(",", ":"))})

sub = pd.DataFrame(rows, columns=["id", "answer_json"])
d = os.path.dirname(os.path.abspath(OUT))
if d:
    os.makedirs(d, exist_ok=True)
sub.to_csv(OUT, index=False)
print("wrote", OUT, sub.shape, "models", len(models), "total %.0fs" % (time.time() - T0), flush=True)

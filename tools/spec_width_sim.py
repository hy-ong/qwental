#!/usr/bin/env python3
# Offline trajectory simulator for the DFlash2 verify-width policy (src/engine.c sw_pick, QW_SPEC_POLICY).
# Replays qw_engine_generate_ex's round loop (suffix lookup, copy / drafter skip, DFlash2 window) over a greedy
# sequence, taking the drafter's path / confidence at every anchor from an offline QW_DF_TEST dump, and reports
# tokens per ms for conf vs cost. Rounds are not independent (a shorter window moves every later anchor), so
# per-round "E[tokens] / cost" simulations over a recorded trajectory are biased; this one follows the trajectory.
#
# Per prompt P, in DIR:
#   P_ids.txt   the prompt token ids (comma-separated, as build/run takes them)
#   P.dump      QW_GEN=1 QW_TEMP=0 QW_SPEC_POLICY=conf QW_SPEC_DUMP=DIR/P.dump build/run @P_ids.txt 600
#               (only its last "S np tokens…" line is used: the greedy sequence)
#   P.df        DFlash2 path / prob at every position along that sequence:
#               QW_TAPS=t.bin build/run @<prompt + generated ids> 0, then
#               QW_DF_TEST=t.bin QW_DF_DUMP=DIR/P.df build/run @P_ids.txt 0
# usage: tools/spec_width_sim.py DIR P [P …]   (CAL=DIR2:P,P,… calibrates the prior q table; default: the prompts themselves)
# env: COST=ms for widths 2..8 (measured round GPU time incl. the drafter), OVH=ms per round outside the GPU
import os, struct, sys

EDGE = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
COST = [0, 0] + [float(x) for x in os.environ.get("COST", "15.85,18.1,20.2,23.6,26.3,28.6,31.9").split(",")]
OVH = float(os.environ.get("OVH", "0.25"))
LMAX = float(os.environ.get("LMAX", "2.2"))  # tokens per width-2 round above which the policy defers to conf (SW_LMAX)
SKIPSAVE, LAZY = 1.27, 2.0  # ms: drafter skipped while copying / drafted lazily at the start of a round


def bin_(p):
    k = 0
    while k < 8 and p >= EDGE[k]:
        k += 1
    return k


def load(d, p):
    for line in open(f"{d}/{p}.dump"):
        if line.startswith("S "):
            f = line.split()
            np_, gen = int(f[1]), list(map(int, f[2:]))
    prompt = list(map(int, open(f"{d}/{p}_ids.txt").read().strip().split(",")))
    assert len(prompt) == np_, (p, len(prompt), np_)
    off, raw, R = {}, open(f"{d}/{p}.df", "rb").read(), 952  # path[7] i32, prob[7] f32, top-16 values / ids per row
    for k in range(len(raw) // R):
        rec = raw[k * R:(k + 1) * R]
        off[np_ + k] = (struct.unpack("7i", rec[:28]), struct.unpack("7f", rec[28:56]))
    return prompt + gen, np_, off


def conf_nd(prob, t=0.8):
    nd = 1
    while nd < 7 and prob[nd] >= t:
        nd += 1
    return nd


class Policy:  # mirrors sw_state / sw_pick / sw_note_df / sw_note_round
    def __init__(self, cost, prior=None, K=8, lam0=3.0 / 15.85, lamw=160.0):  # SW_L0, SW_W
        self.cost, self.prior = cost, prior
        self.n = [K] * 9
        self.a = [(prior[k] if prior else 0.5) * K for k in range(9)]
        self.tk, self.ms, self.full_prev = lam0 * lamw, lamw, False

    def pick(self, prob):
        c0 = conf_nd(prob)
        if not self.cost or (c0 == 7 and self.full_prev):
            return c0
        L, best, bn, e, pc = self.tk / self.ms, -1e30, 1, 1.0, 1.0
        if L > LMAX / COST[2]:  # high-throughput content (code, json, edits): conf is already near optimal
            return c0
        for nd in range(1, 8):
            b = bin_(prob[nd - 1])
            pc *= self.a[b] / self.n[b]
            e += pc
            v = e - L * (COST[nd + 1] + OVH)
            if v > best:
                best, bn = v, nd
        return bn

    def note_df(self, prob, nd, j):
        for i in range(nd):
            b = bin_(prob[i])
            self.n[b] += 1
            if j > i:
                self.a[b] += 1
            else:
                break


def run(seq, np_, off, pol, suf_m=8, suf_long=20):
    ns, df_have, rounds, ms = np_ + 1, True, 0, 0.0
    cut = len(seq) - 9
    while ns - 1 in off:
        best_i, best_m, sf = -1, 0, []
        i = ns - 2
        while i >= 2 and best_m < 32:
            if seq[i] == seq[ns - 1] and seq[i - 1] == seq[ns - 2] and seq[i - 2] == seq[ns - 3]:
                m = 3
                while m < 32 and i - m >= 0 and seq[i - m] == seq[ns - 1 - m]:
                    m += 1
                if m > best_m:
                    best_m, best_i = m, i
                if best_m >= suf_m:
                    break
            i -= 1
        if best_i >= 0:
            while len(sf) < 7 and best_i + 1 + len(sf) < ns:
                sf.append(seq[best_i + 1 + len(sf)])
        ks, cost = len(sf), 0.0
        copy = ks == 7 and best_m >= suf_long
        lazy = not df_have and not copy
        if lazy:
            cost, df_have = LAZY, True
        nd, dr, src = 0, [], 0
        path, prob = off[ns - 1]
        if df_have:
            nd = pol.pick(prob)
            dr = list(path[:nd])
        dfr = df_have
        copy_ok = copy and (nd < 7 or all(sf[k] == dr[k] for k in range(nd)))
        if ks >= 2 and (copy_ok or (ks > nd and (best_m >= suf_m or nd < 2 or sf[0] == dr[0]))):
            dr, nd, src = sf, ks, 1
        skip = src == 1 and copy
        j = 0
        while j < nd and dr[j] == seq[ns + j]:
            j += 1
        if src == 0 and dfr:
            pol.note_df(prob, nd, j)
        cost += COST[nd + 1] + OVH - (SKIPSAVE if skip else 0)
        pol.tk += j + 1
        pol.ms += cost
        pol.full_prev = j == nd and nd == 7
        ms += cost
        rounds += 1
        if ns + j + 1 >= cut:  # stop exactly at the cutoff
            ms -= cost * (1 - (cut - ns) / (j + 1))
            return cut - (np_ + 1), rounds, ms
        ns += j + 1
        df_have = not skip
    return ns - (np_ + 1), rounds, ms


def calibrate(d, prompts):
    n, a = [0] * 9, [0] * 9

    class Rec(Policy):
        def note_df(self, prob, nd, j):
            for i in range(nd):
                b = bin_(prob[i])
                n[b] += 1
                if j > i:
                    a[b] += 1
                else:
                    break

    for p in prompts:
        run(*load(d, p), Rec(False))
    return [(a[k] + 1) / (n[k] + 2) for k in range(9)]


if __name__ == "__main__":
    d, prompts = sys.argv[1], sys.argv[2:]
    cd, cp = (os.environ["CAL"].split(":") + [""])[:2] if os.environ.get("CAL") else (d, ",".join(prompts))
    prior = calibrate(cd, cp.split(","))
    print("prior q per confidence bin", [0] + EDGE, ":", [round(x, 3) for x in prior])
    for p in prompts:
        g = load(d, p)
        t0, r0, m0 = run(*g, Policy(False))
        t1, r1, m1 = run(*g, Policy(True, prior))
        print(f"{p:8s} conf {t0 / m0 * 1e3:6.1f} tok/s ({r0} rounds)  cost {t1 / m1 * 1e3:6.1f} tok/s ({r1} rounds)  {t1 / m1 / (t0 / m0) * 100 - 100:+.1f}%")

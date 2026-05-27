#!/bin/env python3

import numpy as np
import pandas as pd
import heapq
import itertools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import trange  # optional; comment out if not installed

# ---------------------------
# Configurable baseline params
# ---------------------------
SEED_BASE = 12345
SIM_DAYS = 5                    # days per replication
SECONDS_PER_DAY = 24*3600
SIM_TIME = SIM_DAYS * SECONDS_PER_DAY

# Arrival rate piecewise (per minute) -> convert to per second
RATE_OFF = 0.2/60.0
RATE_BASE = 0.5/60.0
RATE_PEAK = 1.2/60.0

# Define time windows (seconds from midnight)
def lambda_t_of_time(t):
    # t is seconds since simulation start; map to time-of-day
    tod = int(t % SECONDS_PER_DAY)
    # business hours 08:00-18:00; peaks 09-11 and 14-16
    if 9*3600 <= tod < 11*3600 or 14*3600 <= tod < 16*3600:
        return RATE_PEAK
    if 8*3600 <= tod < 18*3600:
        return RATE_BASE
    return RATE_OFF

# Service time distributions (lognormal via mean & sigma on original scale)
L1_MEAN_MIN = 8.0     # minutes
L1_SIGMA = 0.8
L2_MEAN_MIN = 40.0
L2_SIGMA = 1.0

def lognormal_from_mean_sigma(mean_minutes, sigma):
    # returns (mu, sigma) for np.random.lognormal where mu is log-mean
    var = (sigma**2)
    mu = np.log(mean_minutes) - 0.5*var
    return mu, sigma

L1_MU, L1_S = lognormal_from_mean_sigma(L1_MEAN_MIN*60.0, L1_SIGMA)  # seconds
L2_MU, L2_S = lognormal_from_mean_sigma(L2_MEAN_MIN*60.0, L2_SIGMA)

# Probabilities & other params
P_FCR_BASE = 0.55
P_REOPEN = 0.08
# Patience (Weibull) median ~15 minutes
PATIENT_MEDIAN = 15.0*60.0
WEIBULL_K = 1.2
# compute scale lambda_w so median = scale*(ln 2)^(1/k)
WEIBULL_SCALE = PATIENT_MEDIAN / (np.log(2)**(1.0/WEIBULL_K))

# VOC baseline and penalties
VOC_BASE_MEAN = 4.2
VOC_SD = 0.4
TRANSFER_PENALTY = 0.6
REOPEN_PENALTY = 0.8
WAIT_PENALITY_PER_30MIN = 1.0  # per 30 minutes wait

# Staffing schedule function (returns number of servers at time t)
def staffing_L1(t, peak_level):
    # simple day/night: peak_level during 8-18, half during off
    tod = int(t % SECONDS_PER_DAY)
    if 8*3600 <= tod < 18*3600:
        return peak_level
    return max(1, peak_level//2)

def staffing_L2(t, peak_level):
    tod = int(t % SECONDS_PER_DAY)
    if 8*3600 <= tod < 18*3600:
        return peak_level
    return max(1, peak_level//2)

# ---------------------------
# DES implementation
# ---------------------------
EVENT_ARRIVAL = "arrival"
EVENT_DEPARTURE = "departure"

class Simulation:
    def __init__(self, rng, l1_peak_staff=12, l2_peak_staff=6, p_fcr=P_FCR_BASE):
        self.rng = rng
        self.l1_peak = l1_peak_staff
        self.l2_peak = l2_peak_staff
        self.p_fcr = p_fcr

        # state
        self.clock = 0.0
        self.event_q = []
        self.event_counter = itertools.count()
        self.l1_queue = []
        self.l2_queue = []
        self.l1_busy = 0
        self.l2_busy = 0

        # stats collection (per-customer records)
        self.records = []  # list of dicts per customer

        # schedule first arrival using nonhomogeneous Poisson thinning
        t0 = self._next_arrival_thinning(self.clock)
        heapq.heappush(self.event_q, (t0, next(self.event_counter), EVENT_ARRIVAL, {}))

    def _max_rate(self):
        return RATE_PEAK

    def _next_arrival_thinning(self, now):
        # thinning: sample exponential with max rate, accept with prob lambda(t)/max
        max_r = self._max_rate()
        while True:
            u = self.rng.random()
            wait = -np.log(u)/max_r
            cand = now + wait
            # accept with prob lambda(cand)/max_r
            if self.rng.random() <= (lambda_t_of_time(cand) / max_r):
                return cand
            now = cand

    def _sample_service_l1(self):
        return self.rng.lognormal(L1_MU, L1_S)

    def _sample_service_l2(self):
        return self.rng.lognormal(L2_MU, L2_S)

    def _sample_patience(self):
        return WEIBULL_SCALE * (-np.log(1 - self.rng.random()))**(1.0/WEIBULL_K)

    def run(self, until):
        while self.event_q:
            time, _, evtype, evdata = heapq.heappop(self.event_q)
            if time > until:
                break
            self.clock = time
            if evtype == EVENT_ARRIVAL:
                self._handle_arrival()
            elif evtype == EVENT_DEPARTURE:
                self._handle_departure(evdata)
        return pd.DataFrame(self.records)

    def _handle_arrival(self):
        arrival_time = self.clock
        record = {
            "arrival_time": arrival_time,
            "l1_enter_time": None,
            "l1_start_time": None,
            "l1_end_time": None,
            "transferred": False,
            "l2_start_time": None,
            "l2_end_time": None,
            "resolved_time": None,
            "resolved_at_level": None,
            "abandoned": False,
            "reopened": False,
            "voc": None,
        }

        # schedule next arrival
        tnext = self._next_arrival_thinning(self.clock)
        heapq.heappush(self.event_q, (tnext, next(self.event_counter), EVENT_ARRIVAL, {}))

        # Join L1 queue
        record["l1_enter_time"] = arrival_time
        patience = self._sample_patience()
        record["patience_deadline"] = arrival_time + patience
        self.l1_queue.append((arrival_time, record))

        # try to start service if capacity
        self._try_start_l1()

    def _try_start_l1(self):
        # get current staffed servers
        staffed = staffing_L1(self.clock, self.l1_peak)
        while self.l1_queue and self.l1_busy < staffed:
            arrival_time, record = self.l1_queue.pop(0)
            # check if already abandoned
            if record["patience_deadline"] <= self.clock:
                record["abandoned"] = True
                record["resolved_time"] = record["patience_deadline"]
                record["resolved_at_level"] = None
                record["l1_start_time"] = None
                record["l1_end_time"] = record["patience_deadline"]
                self.records.append(record)
                continue
            # start service
            record["l1_start_time"] = self.clock
            service = self._sample_service_l1()
            record["l1_end_time"] = self.clock + service
            self.l1_busy += 1
            heapq.heappush(self.event_q, (record["l1_end_time"], next(self.event_counter), EVENT_DEPARTURE, {"level":"l1", "record":record}))

    def _try_start_l2(self):
        staffed = staffing_L2(self.clock, self.l2_peak)
        while self.l2_queue and self.l2_busy < staffed:
            arrival_time, record = self.l2_queue.pop(0)
            # start service
            record["l2_start_time"] = self.clock
            service = self._sample_service_l2()
            record["l2_end_time"] = self.clock + service
            self.l2_busy += 1
            heapq.heappush(self.event_q, (record["l2_end_time"], next(self.event_counter), EVENT_DEPARTURE, {"level":"l2", "record":record}))

    def _handle_departure(self, evdata):
        level = evdata["level"]
        record = evdata["record"]
        if level == "l1":
            self.l1_busy = max(0, self.l1_busy - 1)
            # decide FCR
            if self.rng.random() <= self.p_fcr:
                # resolved at L1
                record["resolved_time"] = self.clock
                record["resolved_at_level"] = 1
                # assign VOC
                self._assign_voc(record)
                self.records.append(record)
            else:
                # transfer to L2
                record["transferred"] = True
                # place in L2 queue (handoff delay negligible here)
                self.l2_queue.append((self.clock, record))
                self._try_start_l2()
            # after any departure, try to start more L1 from queue
            self._try_start_l1()

        elif level == "l2":
            self.l2_busy = max(0, self.l2_busy - 1)
            record["resolved_time"] = self.clock
            record["resolved_at_level"] = 2
            # reopen probability
            if self.rng.random() < P_REOPEN:
                record["reopened"] = True
            self._assign_voc(record)
            self.records.append(record)
            self._try_start_l2()

    def _assign_voc(self, record):
        # wait time = time until resolution minus arrival
        wait_secs = (record["resolved_time"] - record["arrival_time"]) if record["resolved_time"] else 0.0
        wait_pen = min((wait_secs / (30.0*60.0)) * WAIT_PENALITY_PER_30MIN, 1.5)
        transfer_pen = TRANSFER_PENALTY if record.get("transferred", False) else 0.0
        reopen_pen = REOPEN_PENALTY if record.get("reopened", False) else 0.0
        mean = VOC_BASE_MEAN - wait_pen - transfer_pen - reopen_pen
        voc_raw = self.rng.normal(mean, VOC_SD)
        voc = min(5.0, max(1.0, voc_raw))
        record["voc"] = voc

# ---------------------------
# Experiment sweep utilities
# ---------------------------
def run_replication(seed, l1_staff, l2_staff, p_fcr=P_FCR_BASE):
    rng = np.random.default_rng(seed)
    sim = Simulation(rng, l1_peak_staff=l1_staff, l2_peak_staff=l2_staff, p_fcr=p_fcr)
    df = sim.run(SIM_TIME)
    # derive metrics
    served = df[ df["resolved_time"].notna() & (df["abandoned"]==False) ]
    arrivals = len(df)
    served_count = len(served)
    avg_wait = ((served["resolved_time"] - served["arrival_time"]).mean())/60.0 if served_count else np.nan
    p95_wait = np.percentile(((served["resolved_time"] - served["arrival_time"]).dropna())/60.0, 95) if served_count else np.nan
    util_l1 = None  # approximate: total L1 busy seconds / (staff_hours*3600)
    # rough utilization: sum of L1 service durations / (avg staffed * total seconds)
    l1_service_total = df["l1_end_time"].fillna(df["l1_end_time"]).dropna() - df["l1_start_time"].fillna(df["l1_start_time"]).dropna()
    total_l1_service = l1_service_total.sum() if not l1_service_total.empty else 0.0
    # approximate average staffed L1 over sim: average staffing at sample points
    sample_times = np.arange(0, SIM_TIME, 3600)  # hourly
    staffed_samples = [staffing_L1(t, l1_staff) for t in sample_times]
    avg_staffed_l1 = np.mean(staffed_samples)
    util_l1 = total_l1_service / (avg_staffed_l1 * SIM_TIME) if avg_staffed_l1>0 else np.nan

    fcr_observed = df["resolved_at_level"].fillna(0).eq(1).sum() / served_count if served_count else np.nan
    reopen_rate = df["reopened"].sum() / served_count if served_count else np.nan
    voc_mean = df["voc"].dropna().mean() if not df["voc"].dropna().empty else np.nan
    sla_2min = ((served["resolved_time"] - served["arrival_time"]) <= 2*60).sum() / served_count if served_count else np.nan

    return {
        "seed": seed,
        "l1_staff": l1_staff,
        "l2_staff": l2_staff,
        "arrivals": arrivals,
        "served": served_count,
        "avg_wait_min": avg_wait,
        "p95_wait_min": p95_wait,
        "util_l1": util_l1,
        "fcr": fcr_observed,
        "reopen": reopen_rate,
        "voc_mean": voc_mean,
        "sla_2min": sla_2min
    }

def sweep_staffing(l1_options, l2_options, reps=30, p_fcr=P_FCR_BASE):
    results = []
    seeds = [SEED_BASE + i for i in range(reps)]
    total_runs = reps * len(l1_options) * len(l2_options)
    it = range(total_runs)
    pbar = trange(total_runs) if 'trange' in globals() else None
    idx = 0
    for l1 in l1_options:
        for l2 in l2_options:
            for r in range(reps):
                seed = seeds[r]
                res = run_replication(seed, l1, l2, p_fcr=p_fcr)
                results.append(res)
                if pbar:
                    pbar.update(1)
                idx += 1
    if pbar:
        pbar.close()
    return pd.DataFrame(results)

# ---------------------------
# Main: run sweep and plot
# ---------------------------
if __name__ == "__main__":
    l1_options = [8, 10, 12, 14]
    l2_options = [4, 6, 8]
    reps = 30

    df_results = sweep_staffing(l1_options, l2_options, reps=reps)
    summary = df_results.groupby(["l1_staff","l2_staff"]).agg(
        arrivals=("arrivals","mean"),
        served=("served","mean"),
        avg_wait_min=("avg_wait_min","mean"),
        p95_wait_min=("p95_wait_min","mean"),
        util_l1=("util_l1","mean"),
        fcr=("fcr","mean"),
        voc_mean=("voc_mean","mean")
    ).reset_index()

    print(summary)

    # plot cost proxy (FTEs) vs VOC_mean and avg_wait
    summary["fte_hours_per_day"] = (summary["l1_staff"] + summary["l2_staff"]) * 8  # proxy
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    for l2 in l2_options:
        sub = summary[summary["l2_staff"]==l2]
        plt.plot(sub["fte_hours_per_day"], sub["voc_mean"], marker='o', label=f"L2={l2}")
    plt.xlabel("FTE hours/day (proxy)")
    plt.ylabel("Mean VOC")
    plt.legend()
    plt.title("Cost proxy vs VOC")

    plt.subplot(1,2,2)
    for l2 in l2_options:
        sub = summary[summary["l2_staff"]==l2]
        plt.plot(sub["fte_hours_per_day"], sub["avg_wait_min"], marker='o', label=f"L2={l2}")
    plt.xlabel("FTE hours/day (proxy)")
    plt.ylabel("Avg wait (min)")
    plt.legend()
    plt.title("Cost proxy vs Avg Wait")

    plt.tight_layout()
    plt.savefig("staffing_plots.png", dpi=150)

    # save outputs
    summary.to_csv("staffing_sweep_summary.csv", index=False)
    df_results.to_csv("staffing_sweep_raw.csv", index=False)
    print("Saved staffing_sweep_summary.csv and staffing_sweep_raw.csv")


import sys
import os as os
import glob as glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as m

columnNames = ["scaling", "p", "mpi_p", "omp_p", "kernel", "tfread", "tpreproc", "tkernel", "tblur", "tpostproc", "tfwrite", "twall", "telapsed", "tuser", "tsys"]
df_dict = dict()
for k in columnNames:
    df_dict[k] = list()
measureNames = ["tfread", "tpreproc", "tkernel", "tblur", "tpostproc", "tfwrite", "twall", "telapsed", "tuser", "tsys"]

def balanced_divisors(n):
    a = n
    diff = n - 1
    for b in range(1, n + 1):
        if n % b == 0:
            d = abs(n // b - b)
            if d < diff:
                a = b
                diff = d
    b = n // a
    return max(a, b), min(a, b)

def blur_workload(rows, columns, kernel_diameter):
    q = kernel_diameter * (kernel_diameter - 1) - (kernel_diameter // 2 + 2) * (kernel_diameter // 2 + 1)
    r = rows - (kernel_diameter - 1)
    c = columns - (kernel_diameter - 1)

    return r * kernel_diameter * q + c * kernel_diameter * q + r * c * kernel_diameter**2 + q**2

def append(df_dict, scaling, mpi_p, omp_p, kernel, telapsed, tuser, tsys, tfread, tpreproc, tkernel, tblur, tpostproc, tfwrite, twall):
    df_dict["scaling"].append(scaling)

    df_dict["mpi_p"].append(mpi_p)
    df_dict["omp_p"].append(omp_p)
    df_dict["p"].append(mpi_p * omp_p)
    df_dict["kernel"].append(kernel)

    df_dict["telapsed"].append(telapsed)
    df_dict["tuser"].append(tuser)
    df_dict["tsys"].append(tsys)

    df_dict["tfread"].append(tfread)
    df_dict["tpreproc"].append(tpreproc)
    df_dict["tkernel"].append(tkernel)
    df_dict["tblur"].append(tblur)
    df_dict["tpostproc"].append(tpostproc)
    df_dict["tfwrite"].append(tfwrite)
    df_dict["twall"].append(twall)

def processDir(d, df_dict):
    for fn in sorted(glob.glob(os.path.join(d, "*.o[0-9]*"))):
        print(f"reading {fn}")
        with open(fn, "r") as f:
            scaling = ""

            mpi_p = 0
            omp_p = 0
            kernel = 0

            telapsed = 0
            tuser = 0
            tsys = 0

            tfread = 0
            tpreproc = 0
            tkernel = 0
            tblur = 0
            tpostproc = 0
            tfwrite = 0
            twall = 0

            for l in f.readlines():
                tokens = l[:-1].split(" ")

                if tokens[0] == "run":
                    scaling = tokens[-1]
                    mpi_p = int(tokens[2])
                    omp_p = int(tokens[4])
                    kernel = int(tokens[6])
                elif tokens[0] == "rank":
                    if tokens[4] == "timing_file_read:":
                        tfread = max(tfread, float(tokens[5]))
                    elif tokens[4] == "timing_preprocess:":
                        tpreproc = max(tpreproc, float(tokens[5]))
                    elif tokens[4] == "timing_kernel_init:":
                        tkernel = max(tkernel, float(tokens[5]))
                    elif tokens[4] == "timing_blur:":
                        tblur = max(tblur, float(tokens[5]))
                    elif tokens[4] == "timing_postprocess:":
                        tpostproc = max(tpostproc, float(tokens[5]))
                    elif tokens[4] == "timing_file_write:":
                        tfwrite = max(tfwrite, float(tokens[5]))
                    elif tokens[4] == "timing_wall:":
                        twall = max(twall, float(tokens[5]))
                elif tokens[0] == "elapsed:":
                    telapsed = float(tokens[1])
                elif tokens[0] == "user:":
                    tuser = float(tokens[1])
                elif tokens[0] == "system:":
                    tsys = float(tokens[1])

                    #this is the last line
                    if mpi_p and omp_p and twall > 0:
                        #valid entry

                        if mpi_p == 1 and omp_p == 1:
                            #this counts for multiple scaling combinations
                            if "strong" in fn:
                                xscaling = "mpi_strong"
                                append(df_dict, xscaling, mpi_p, omp_p, kernel, telapsed, tuser, tsys, tfread, tpreproc, tkernel, tblur, tpostproc, tfwrite, twall)

                                xscaling = "omp_strong"
                                append(df_dict, xscaling, mpi_p, omp_p, kernel, telapsed, tuser, tsys, tfread, tpreproc, tkernel, tblur, tpostproc, tfwrite, twall)

                            if "weak" in fn:
                                xscaling = "mpi_weak"
                                append(df_dict, xscaling, mpi_p, omp_p, kernel, telapsed, tuser, tsys, tfread, tpreproc, tkernel, tblur, tpostproc, tfwrite, twall)

                                xscaling = "omp_weak"
                                append(df_dict, xscaling, mpi_p, omp_p, kernel, telapsed, tuser, tsys, tfread, tpreproc, tkernel, tblur, tpostproc, tfwrite, twall)
                        elif mpi_p > 1:
                            scaling = "mpi_" + scaling
                            append(df_dict, scaling, mpi_p, omp_p, kernel, telapsed, tuser, tsys, tfread, tpreproc, tkernel, tblur, tpostproc, tfwrite, twall)
                        elif omp_p > 1:
                            scaling = "omp_" + scaling
                            append(df_dict, scaling, mpi_p, omp_p, kernel, telapsed, tuser, tsys, tfread, tpreproc, tkernel, tblur, tpostproc, tfwrite, twall)

                    #reset
                    scaling = ""

                    mpi_p = 0
                    omp_p = 0
                    kernel = 0

                    telapsed = 0
                    tuser = 0
                    tsys = 0

                    tfread = 0
                    tkernel = 0
                    tblur = 0
                    tfwrite = 0
                    twall = 0

def plotExperiment(df, title, column, ylabel, yscale="linear", drawBisect=False, drawFlat1=False, save="", show=False):
    plt.close()

    kernels = df["kernel"].unique()
    #print(f"kernels: {kernels}")

    reset_x = False
    for k in sorted(kernels):
        df_tmp = df.loc[df["kernel"] == k]
        if not reset_x:
            plt.plot(df_tmp["p"], df_tmp[column], label=f"kernel: {k}")

    if drawBisect:
        plt.plot(df_tmp["p"], df_tmp["p"], label=f"perfect scaling", linestyle=":", color="grey")
        plt.ylim(0, df_tmp["p"].max())
    elif drawFlat1:
        plt.plot(df_tmp["p"], np.ones_like(df_tmp["p"]), label=f"perfect scaling", linestyle=":", color="grey")
        plt.ylim(0, 1.1)

    #locs, labels = plt.xticks()
    #print(f"xlocs: {locs}")
    #print(f"xlabels: {labels}")

    plt.title(title)

    plt.xlabel("P")
    #plt.xscale("log")
    plt.ylabel(ylabel)
    plt.yscale(yscale)
    plt.legend()

    if save:
        plt.savefig(save, bbox_inches="tight")
    elif show:
        plt.show()

if len(sys.argv) < 2:
    print(f"Usage: python {sys.argv[0]} input_folder [output_folder]")

dOut = "csvs"
dOut_suffix = sys.argv[1][len(dOut):]

if len(sys.argv) >= 3:
    dOut = sys.argv[2]

processDir(sys.argv[1], df_dict)

df = pd.DataFrame(df_dict, columns=columnNames)
df.sort_values(["scaling", "kernel", "p", "mpi_p", "omp_p"], inplace=True)
print(df)

df.to_csv(os.path.join(dOut + dOut_suffix, "master.csv"), index=False, float_format="%.6f")

plt.close()
groupk = ["scaling", "kernel", "p", "mpi_p", "omp_p"]
for fid in ["omp_weak", "mpi_weak"]:
    print(f"scaling: {fid}")
    base_path = os.path.join(dOut + dOut_suffix, fid)

    df_mean = df.loc[(df["scaling"] == fid) & (df["kernel"] == 31)].groupby(groupk).mean().copy()
    df_mean.reset_index(inplace=True)

    df_baseline = df_mean.loc[df_mean["p"] == 1].copy()
    df_baseline.drop(["p", "mpi_p", "omp_p"], axis=1, inplace=True)
    joink = ["scaling", "kernel"]
    df_ratio = pd.merge(df_mean, df_baseline, how="inner", on=joink, left_on=None, right_on=None, left_index=False, right_index=False, sort=True, suffixes=("", "_baseline"), copy=True, indicator=False, validate=None,)
    df_ratio.reset_index(inplace=True)

    ratio_suffix = "_ratio"
    for c in ["telapsed"]:
        df_ratio[c + ratio_suffix] = df_ratio[c] / df_ratio[c + "_baseline"]

    columns = df_ratio.columns.tolist()
    keys = ["scaling", "kernel", "p", "mpi_p", "omp_p"]
    columns = sorted(list(set(columns) - set(keys)))

    df_ratio = df_ratio[keys + columns]
    print(f"df_ratio {df_ratio}")

    df_ratio.drop("index", axis=1, inplace=True)
    df_ratio.to_csv(base_path + ratio_suffix + ".csv", index=False, float_format="%.6f")

    kernels = df_ratio["kernel"].unique()
    for k in sorted(kernels):
        df_tmp = df_ratio.loc[df_ratio["kernel"] == k]
        lab = f"OpenMP k {k}"
        if fid == "mpi_weak":
            lab = f"MPI k {k}"
        plt.plot(df_tmp["p"], df_tmp["telapsed_ratio"], label=lab)

plt.title("Weak scaling")

plt.xlabel("P")
#plt.xscale("log")
plt.ylabel("normalized elapsed time")
plt.yscale("linear")
plt.legend()

plt.savefig("weakscaling.png", bbox_inches="tight")

plt.close()
for fid in ["omp_strong", "mpi_strong"]:
    print(f"scaling: {fid}")
    base_path = os.path.join(dOut + dOut_suffix, fid)

    df_mean = df.loc[df["scaling"] == fid].groupby(groupk).mean().copy()
    df_mean.reset_index(inplace=True)

    df_baseline = df_mean.loc[df_mean["p"] == 1].copy()
    df_baseline.drop(["p", "mpi_p", "omp_p"], axis=1, inplace=True)
    joink = ["scaling", "kernel"]
    df_ratio = pd.merge(df_mean, df_baseline, how="inner", on=joink, left_on=None, right_on=None, left_index=False, right_index=False, sort=True, suffixes=("", "_baseline"), copy=True, indicator=False, validate=None,)
    df_ratio.reset_index(inplace=True)

    ratio_suffix = "_ratio"
    for c in ["telapsed"]:
        df_ratio[c + ratio_suffix] = df_ratio[c] / df_ratio[c + "_baseline"]

    columns = df_ratio.columns.tolist()
    keys = ["scaling", "kernel", "p", "mpi_p", "omp_p"]
    columns = sorted(list(set(columns) - set(keys)))

    df_ratio = df_ratio[keys + columns]
    print(f"df_ratio {df_ratio}")

    df_ratio.drop("index", axis=1, inplace=True)
    df_ratio.to_csv(base_path + ratio_suffix + ".csv", index=False, float_format="%.6f")

    kernels = df_ratio["kernel"].unique()
    for k in sorted(kernels):
        df_tmp = df_ratio.loc[df_ratio["kernel"] == k]
        lab = f"OpenMP k {k}"
        if fid == "mpi_strong":
            lab = f"MPI k {k}"
        plt.plot(df_tmp["p"], df_tmp["telapsed_ratio"], label=lab)

plt.title("Strong scaling")

plt.xlabel("P")
#plt.xscale("log")
plt.ylabel("normalized elapsed time")
plt.yscale("linear")
plt.legend()

plt.savefig("strongscaling.png", bbox_inches="tight")

"""
for fid in ["omp_strong", "omp_weak", "mpi_strong", "mpi_weak"]:
    print(f"scaling: {fid}")
    base_path = os.path.join(dOut + dOut_suffix, fid)

    df_mean = df.loc[df["scaling"] == fid].groupby(groupk).mean().copy()
    df_min = df.loc[df["scaling"] == fid].groupby(groupk).min().copy()
    df_max = df.loc[df["scaling"] == fid].groupby(groupk).max().copy()

    df_grouped = pd.merge(df_mean, df_min, how="inner", on=groupk, left_on=None, right_on=None, left_index=False, right_index=False, sort=True, suffixes=("", "_min"), copy=True, indicator=False, validate=None,)
    df_grouped = pd.merge(df_grouped, df_max, how="inner", on=groupk, left_on=None, right_on=None, left_index=False, right_index=False, sort=True, suffixes=("", "_max"), copy=True, indicator=False, validate=None,)

    df_mean.reset_index(inplace=True)
    df_min.reset_index(inplace=True)
    df_max.reset_index(inplace=True)

    df_grouped.reset_index(inplace=True)
    df_grouped.sort_values(["scaling", "kernel", "p", "mpi_p", "omp_p"], inplace=True)
    #print(f"df_grouped {df_grouped}")
    #df_grouped.to_csv(base_path + ".csv", index=False, float_format="%.6f")

    df_baseline = df_mean.loc[df_mean["p"] == 1].copy()
    df_baseline.drop(["p", "mpi_p", "omp_p"], axis=1, inplace=True)
    joink = ["scaling", "kernel"]
    df_ratio = pd.merge(df_mean, df_baseline, how="inner", on=joink, left_on=None, right_on=None, left_index=False, right_index=False, sort=True, suffixes=("", "_baseline"), copy=True, indicator=False, validate=None,)
    df_ratio.reset_index(inplace=True)

    ratio_suffix = ""
    if fid.endswith("strong"):
        ratio_suffix = "_speedup"
    else:
        ratio_suffix = "_efficiency"

    for c in measureNames:
        df_ratio[c + ratio_suffix] = df_ratio[c + "_baseline"] / df_ratio[c]

        df_ratio[c + ratio_suffix + "_naive"] = 1
        if fid.endswith("strong"):
            df_ratio[c + ratio_suffix + "_naive"] *= df_ratio["p"]

    #corrected blur ratio for weak scaling
    if fid.endswith("weak"):
        df_ratio["tblur" + ratio_suffix + "_corrected"] = df_ratio["tblur" + ratio_suffix]

        for i in df_ratio.index:
            blur_workload_baseline = blur_workload(img_rows, img_columns, df_ratio.loc[i, "kernel"])
            rows_ratio, columns_ratio = balanced_divisors(df_ratio.loc[i, "p"])
            df_ratio.loc[i, "tblur" + ratio_suffix + "_corrected"] *= blur_workload(img_rows * rows_ratio, img_columns * columns_ratio, df_ratio.loc[i, "kernel"]) / (df_ratio.loc[i, "p"] * blur_workload_baseline)

    columns = df_ratio.columns.tolist()
    keys = ["scaling", "kernel", "p", "mpi_p", "omp_p"]
    columns = sorted(list(set(columns) - set(keys)))

    print(f"df_ratio {df_ratio}")
    df_ratio = df_ratio[keys + columns]
    print(f"df_ratio {df_ratio}")

    df_ratio.drop("index", axis=1, inplace=True)
    df_ratio.to_csv(base_path + ratio_suffix + ".csv", index=False, float_format="%.6f")

    #plotExperiment(df, title, column, ylabel, yscale="linear", drawBisect=False, drawFlat1=False, show=False)
    #elapsed times
    plotExperiment(df_mean, fid + "_elapsed", "telapsed", "elapsed time (s)", yscale="linear", show=False)
    #process times
    plotExperiment(df_mean, fid + "_wall", "twall", "process time (s)", yscale="linear", show=False)
    #blur times
    plotExperiment(df_mean, fid + "_blur", "tblur", "blur time (s)", yscale="linear", show=False)

    if "mpi" in fid:
        name = "MPI"
    else:
        name = "OpenMP"

    if fid.endswith("strong"):
        name_suffix = " speedup"
    else:
        name_suffix = " efficiency"

    if fid.endswith("strong"):
        #elapsed speedup
        plotExperiment(df_ratio, name + " elapsed time" + name_suffix, "telapsed" + ratio_suffix, ratio_suffix[1:], yscale="linear", drawBisect=True, save=base_path + "_elapsed" + ratio_suffix + ".png", show=True)
        #process speedup
        plotExperiment(df_ratio, name + " wall " + name_suffix, "twall" + ratio_suffix, ratio_suffix[1:], yscale="linear", drawBisect=True, save=base_path + "_wall" + ratio_suffix + ".png", show=True)
        #blur speedup
        plotExperiment(df_ratio, name + " blur " + name_suffix, "tblur" + ratio_suffix, ratio_suffix[1:], yscale="linear", drawBisect=True, save=base_path + "_blur" + ratio_suffix + ".png", show=True)
    else:
        pass
        #elapsed efficiency
        plotExperiment(df_ratio, name + " elapsed time" + name_suffix, "telapsed" + ratio_suffix, ratio_suffix[1:], yscale="linear", drawFlat1=True, save=base_path + "_elapsed" + ratio_suffix + ".png", show=True)
        #process efficiency
        plotExperiment(df_ratio, name + " wall " + name_suffix, "twall" + ratio_suffix, ratio_suffix[1:], yscale="linear", drawFlat1=True, save=base_path + "_wall" + ratio_suffix + ".png", show=True)
        #blur efficiency
        plotExperiment(df_ratio, name + " blur " + name_suffix, "tblur" + ratio_suffix, ratio_suffix[1:], yscale="linear", drawFlat1=True, save=base_path + "_blur" + ratio_suffix + ".png", show=True)
        #blur efficiency corrected
        plotExperiment(df_ratio, name + " blur corrected " + name_suffix, "tblur" + ratio_suffix + "_corrected", ratio_suffix[1:], yscale="linear", drawFlat1=True, save=base_path + "_blur_corrected" + ratio_suffix + ".png", show=True)
"""
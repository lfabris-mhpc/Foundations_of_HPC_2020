import sys
import os as os
import glob as glob
import pandas as pd
import math as m

columnNames = ["scaling", "p", "mpi_p", "omp_p", "kernel", "tfread", "tkernel", "tblur", "tfwrite", "twall", "telapsed", "tuser", "tsys"]
df_dict = dict()
for k in columnNames:
    df_dict[k] = list()

def append(df_dict, scaling, mpi_p, omp_p, kernel, telapsed, tuser, tsys, tfread, tkernel, tblur, tfwrite, twall):
    df_dict["scaling"].append(scaling)

    df_dict["mpi_p"].append(mpi_p)
    df_dict["omp_p"].append(omp_p)
    df_dict["p"].append(mpi_p * omp_p)
    df_dict["kernel"].append(kernel)

    df_dict["telapsed"].append(telapsed)
    df_dict["tuser"].append(tuser)
    df_dict["tsys"].append(tsys)

    df_dict["tfread"].append(tfread)
    df_dict["tkernel"].append(tkernel)
    df_dict["tblur"].append(tblur)
    df_dict["tfwrite"].append(tfwrite)
    df_dict["twall"].append(twall)

def processDir(d, df_dict, dOut="csvs"):
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
            tkernel = 0
            tblur = 0
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
                    elif tokens[4] == "timing_kernel_init:":
                        tkernel = max(tkernel, float(tokens[5]))
                    elif tokens[4] == "timing_blur:":
                        tblur = max(tblur, float(tokens[5]))
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
                            #this counts for all scaling combinations
                            xscaling = "mpi_strong"
                            append(df_dict, xscaling, mpi_p, omp_p, kernel, telapsed, tuser, tsys, tfread, tkernel, tblur, tfwrite, twall)

                            xscaling = "mpi_weak"
                            append(df_dict, xscaling, mpi_p, omp_p, kernel, telapsed, tuser, tsys, tfread, tkernel, tblur, tfwrite, twall)

                            xscaling = "omp_strong"
                            append(df_dict, xscaling, mpi_p, omp_p, kernel, telapsed, tuser, tsys, tfread, tkernel, tblur, tfwrite, twall)

                            xscaling = "omp_weak"
                            append(df_dict, xscaling, mpi_p, omp_p, kernel, telapsed, tuser, tsys, tfread, tkernel, tblur, tfwrite, twall)
                        elif mpi_p > 1:
                            scaling = "mpi_" + scaling
                            append(df_dict, scaling, mpi_p, omp_p, kernel, telapsed, tuser, tsys, tfread, tkernel, tblur, tfwrite, twall)
                        elif omp_p > 1:
                            scaling = "omp_" + scaling
                            append(df_dict, scaling, mpi_p, omp_p, kernel, telapsed, tuser, tsys, tfread, tkernel, tblur, tfwrite, twall)

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

if len(sys.argv) < 2:
    print(f"Usage: python {sys.argv[0]} input_folder [output_folder]")

dOut = "csvs"
suffix = sys.argv[1][len(dOut):]

if len(sys.argv) >= 3:
    dOut = sys.argv[2]

processDir(sys.argv[1], df_dict, dOut)

df = pd.DataFrame(df_dict, columns=columnNames)
df.sort_values(["scaling", "kernel", "p", "mpi_p", "omp_p"], inplace=True)
print(df)

df.to_csv(os.path.join(dOut, "master.csv"), index=False, float_format="%.6f")

groupk = ["scaling", "p", "mpi_p", "omp_p", "kernel"]
for fid in ["omp_strong", "omp_weak", "mpi_strong", "mpi_weak"]:
    print(f"scaling: {fid}")
    df_mean = df.loc[df["scaling"] == fid].groupby(groupk).mean()
    df_min = df.loc[df["scaling"] == fid].groupby(groupk).min()
    df_max = df.loc[df["scaling"] == fid].groupby(groupk).max()

    #print(df_mean)
    #print(df_min)
    #print(df_max)

    #df_grouped = pd.concat([df_mean, df_min, df_max], axis=1, join="inner", ignore_index=True, keys=groupk, levels=None, names=None, verify_integrity=False, copy=True,)

    df_grouped = pd.merge(df_mean, df_min, how="inner", on=groupk, left_on=None, right_on=None, left_index=False, right_index=False, sort=True, suffixes=("", "_min"), copy=True, indicator=False, validate=None,)
    df_grouped = pd.merge(df_grouped, df_max, how="inner", on=groupk, left_on=None, right_on=None, left_index=False, right_index=False, sort=True, suffixes=("", "_max"), copy=True, indicator=False, validate=None,)

    df_grouped.sort_values(["scaling", "kernel", "p", "mpi_p", "omp_p"], inplace=True)
    print(df_grouped)
    df_grouped.to_csv(os.path.join(dOut, fid + ".csv"), index=False, float_format="%.6f")

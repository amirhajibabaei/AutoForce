# +
from theforce.io.sgprio import SgprIO


def truncate_sgpr(infile, outfile, ndat, nlce):
    m = SgprIO(infile)
    mm = SgprIO(outfile)
    c_ndat = 0
    c_nlce = 0
    for a, b in m.read():
        if a == "params":
            mm.write_params(**b)
        elif a == "atoms":
            mm.write(b)
            c_ndat += 1
        elif a == "local":
            mm.write(b)
            c_nlce += 1
        if c_ndat >= ndat and c_nlce >= nlce:
            break
    print(f"truncated to {c_ndat} data and {c_nlce} inducing")


if __name__ == "__main__":
    import sys

    infile = sys.argv[1]
    outfile = sys.argv[2]
    ndat = int(sys.argv[3])
    nlce = int(sys.argv[4])
    truncate_sgpr(infile, outfile, ndat, nlce)

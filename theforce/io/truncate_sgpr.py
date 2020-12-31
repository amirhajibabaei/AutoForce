# +
from theforce.io.sgprio import SgprIO


def truncate_sgpr(infile, outfile, ndat, ncle):
    m = SgprIO(infile)
    mm = SgprIO(outfile)
    c_ndat = 0
    c_nlce = 0
    for a, b in m.read():
        if a == 'params':
            mm.write_params(**b)
        elif a == 'atoms':
            mm.write(b)
            c_ndat += 1
        elif a == 'local':
            mm.write(b)
            c_nlce += 1
        if c_ndat == ndat and c_ncle == ncle:
            break


if __name__ == '__main__':
    import sys
    infile = sys.argv[1]
    outfile = sys.argv[2]
    ndat = sys.argv[3]
    ncle = sys.argv[4]
    truncate_sgpr(infile, outfile, ndat, ncle)

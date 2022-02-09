# +
from theforce.regression.gppotential import PosteriorPotentialFromFolder
from theforce.regression.scores import cd
import torch
from theforce.util.parallel import mpi_init
import theforce.distributed as dist


def least_important(A, y, rank, size):
    i_glob = torch.zeros(size, dtype=torch.int)
    r_glob = torch.zeros(size, dtype=torch.float)
    i_local = []
    r_local = []
    for i in range(rank, A.size(1), size):
        B = torch.cat([A[:, :i], A[:, i+1:]], dim=1)
        mu = torch.linalg.lstsq(B, y).solution
        i_local.append(i)
        r_local.append(cd(B@mu, y))
    i = torch.argmax(torch.stack(r_local))
    i_glob[rank] = i_local[i]
    r_glob[rank] = r_local[i]
    dist.all_reduce(i_glob)
    dist.all_reduce(r_glob)
    j = torch.argmax(r_glob)
    return i_glob[j], r_glob[j]


def main(pckl, inducing, R2, out):
    mpi_init()
    rank = dist.get_rank()
    size = dist.get_world_size()
    model = PosteriorPotentialFromFolder(pckl,
                                         load_data=True,
                                         update_data=False)
    A = model.Kf.clone()
    y = torch.cat([atoms.target_forces.view(-1) for atoms in model.data])
    m = A.size(1)
    indices = list(range(m))
    dumped = []
    while True:
        i, score = least_important(A, y, rank, size)
        if score < R2:
            break
        A = torch.cat([A[:, :i], A[:, i+1:]], dim=1)
        dumped.append(indices.pop(i))
        m -= 1
        if rank == 0:
            print(m, float(score))
        if m <= inducing:
            break

    model.select_inducing(indices)
    # assert model.Kf.allclose(A)
    model.optimize_model_parameters()
    if rank == 0:
        if out is None:
            out = pckl
        model.to_folder(out)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Compression of SGPR models'
    )

    parser.add_argument('-p', '--pckl',
                        default='model.pckl',
                        type=str,
                        help="path to a .pckl forlder")

    parser.add_argument('-o', '--out',
                        default=None,
                        type=str,
                        help="output .pckl folder "
                             "(if not given, the input will be overwritten)"
                        )

    parser.add_argument('-i', '--inducing',
                        default=100000,
                        type=int,
                        help=f'minimum number of inducing points')

    parser.add_argument('-r', '--r2',
                        default=1.,
                        type=float,
                        help='minimum force R2')

    args = parser.parse_args()

    main(args.pckl, args.inducing, args.r2, args.out)

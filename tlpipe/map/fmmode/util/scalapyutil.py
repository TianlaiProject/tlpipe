import numpy as np
import mpiutil
from mpi4py import MPI

from scalapy import core
import scalapy.routines as rt


def eigh_gen(A, B, lower=True, overwrite_a=True,  overwrite_b=True):
    """Solve the generalised eigenvalue problem. :math:`\mathbf{A} \mathbf{v} = \lambda \mathbf{B} \mathbf{v}` of two symmetric/hermitian matrix.

    Use Scalapack to compute the eigenvalues and eigenvectors of two distributed matrix.
    The returned eigenvectors is somewhate different from that returned by `scipy.linalg.eigh`. Here evecs.T.conj() corresponds to the eigenvectors returned by `scipy.linalg.eigh`, but usally they are not the same, they still may differ from a diagonal unitary matrix.

    Parameters
    ----------
    A, B : DistributedMatrix
        The matrix to decompose.
    lower : boolean, optional
        Scalapack uses only half of the matrix, by default the lower
        triangle will be used. Set to False to use the upper triangle.
    overwrite_a, overwrite_b : boolean, optional
        By default the input matrix is destroyed, if set to False a
        copy is taken and operated on.
    eigvals : tuple (lo, hi), optional
        Indices of the lowest and highest eigenvalues you would like to
        calculate. Indexed from zero.

    Returns
    -------
    evals : np.ndarray
        The eigenvalues of the matrix, they are returned as a global
        numpy array of all values.
    evecs : DistributedMatrix
        The eigenvectors as a DistributedMatrix.
    """
    Lambda1, R1H = rt.eigh(B, lower=lower, overwrite_a=overwrite_b)

    # L1 = rt.dot(rt.dot(R1H, B, transA='C', transB='N'), R1H, transA='N', transB='N')
    # L1 = L1.to_global_array(rank=0)
    # if MPI.COMM_WORLD.rank == 0:
    #     assert np.allclose(Lambda1, np.diag(L1).real)

    if not (Lambda1 > 0.0).all():
        add_const = - (np.min(Lambda1) * (1.0 + 1e-12) + 1e-60)
        Lambda1 += add_const
        if B.context.mpi_comm.Get_rank() == 0:
            print "Second matrix probably not positive definite due to numerical issues. \
Add a minimum constant %f to all of its eigenvalues to make it positive definite...." % add_const

    ihalfL = np.diag(Lambda1**(-0.5)).astype(R1H.dtype)

    # if MPI.COMM_WORLD.rank == 0:
    #     print np.dot(np.dot(ihalfL, np.diag(Lambda1)), ihalfL.T.conj())

    ihalfL = np.asfortranarray(ihalfL)
    ihalfL = core.DistributedMatrix.from_global_array(ihalfL, rank=0, block_shape=A.block_shape, context=A.context)
    # R2R1 = rt.dot(ihalfL, R1H, transA='N', transB='H')
    R2R1 = rt.dot(ihalfL, R1H, transA='N', transB='C')

    # I = rt.dot(rt.dot(R2R1, B, transA='N', transB='N'), R2R1, transA='N', transB='C')
    # I = I.to_global_array(rank=0)
    # if MPI.COMM_WORLD.rank == 0:
    #     print 'I ='
    #     print I

    # A1 = rt.dot(rt.dot(R2R1, A, transA='N', transB='N'), R2R1, transA='N', transB='H')
    if overwrite_a:
        A = rt.dot(rt.dot(R2R1, A, transA='N', transB='N'), R2R1, transA='N', transB='C')
        Lambda3, R3H = rt.eigh(A, lower=lower, overwrite_a=overwrite_a)
    else:
        A1 = rt.dot(rt.dot(R2R1, A, transA='N', transB='N'), R2R1, transA='N', transB='C')
        Lambda3, R3H = rt.eigh(A1, lower=lower, overwrite_a=True)
    # R = rt.dot(R3H, R2R1, transA='H', transB='N')
    R = rt.dot(R3H, R2R1, transA='C', transB='N')
    return Lambda3, R
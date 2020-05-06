from pathlib import Path
from PIL import Image
import numpy as np
import math
from timeit import default_timer as timer

from numba import jit, njit, prange, vectorize, cuda


MOVIE_TITLE = "10sec4k"


# Numba Parallel 
@njit(parallel=True)
def filter3d_core_parallel(image, filt, result):
    M, N, D = image.shape
    Mf, Nf = filt.shape
    Mf2 = Mf // 2
    Nf2 = Nf // 2
    for d in prange(D):
        for i in prange(M):
            for j in prange(N):
                num = 0
                cnt = 0
                for ii in prange(Mf):
                    for jj in prange(Nf):
                        if 0 <= i-Mf2+ii < M and 0 <= j-Nf2+jj < N:
                            cnt += 1
                            num += (filt[Mf-1-ii, Nf-1-jj] * image[i-Mf2+ii,j-Nf2+jj,d])

                result[i, j, d] = num / cnt


@njit
def filter3d_parallel(image, filt):
    result = np.zeros_like(image)
    filter3d_core_parallel(image, filt, result)
    return result


# Numba JIT only 
@njit
def filter3d_core_jit(image, filt, result):
    M, N, D = image.shape
    Mf, Nf = filt.shape
    Mf2 = Mf // 2
    Nf2 = Nf // 2
    for d in range(D):
        for i in range(M):
            for j in range(N):
                num = 0
                cnt = 0
                for ii in range(Mf):
                    for jj in range(Nf):
                        if 0 <= i-Mf2+ii < M and 0 <= j-Nf2+jj < N:
                            cnt += 1
                            num += (filt[Mf-1-ii, Nf-1-jj] * image[i-Mf2+ii,j-Nf2+jj,d])

                result[i, j, d] = num / cnt



@njit
def filter3d_jit(image, filt):
    result = np.zeros_like(image)
    filter3d_core_jit(image, filt, result)
    return result


# Pure Python
def filter3d_core(image, filt, result):
    M, N, D = image.shape
    Mf, Nf = filt.shape
    Mf2 = Mf // 2
    Nf2 = Nf // 2
    for d in range(D):
        for i in range(M):
            for j in range(N):
                num = 0
                cnt = 0
                for ii in range(Mf):
                    for jj in range(Nf):
                        if 0 <= i-Mf2+ii < M and 0 <= j-Nf2+jj < N:
                            cnt += 1
                            num += (filt[Mf-1-ii, Nf-1-jj] * image[i-Mf2+ii,j-Nf2+jj,d])

                result[i, j, d] = num / cnt

def filter3d(image, filt):
    result = np.zeros_like(image)
    filter3d_core(image, filt, result)
    return result

# Numba GPU

@cuda.jit
def filter3d_core_cuda(image, filt, result):
    M, N, D = image.shape
    Mf, Nf = filt.shape
    Mf2 = Mf // 2
    Nf2 = Nf // 2
    i, j, d = cuda.grid(3)
    num = 0
    cnt = 0
    for ii in range(Mf):
        for jj in range(Nf):
            if 0 <= i-Mf2+ii < M and 0 <= j-Nf2+jj < N:
                cnt += 1
                num += (filt[Mf-1-ii, Nf-1-jj] * image[i-Mf2+ii,j-Nf2+jj,d])

    result[i, j, d] = num / cnt



def filter3d_cuda(image, filt):
    result = np.zeros_like(image)
    threadsperblock = (3, 3, image.shape[2])
    blockspergrid_x = math.ceil(image.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(image.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y, image.shape[2])
    filter3d_core_cuda[blockspergrid, threadsperblock](image, filt, result)
    return result


if __name__ == '__main__': 
    
    path = Path(f'frames/{MOVIE_TITLE}/gg')
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    p = Path(f'frames/{MOVIE_TITLE}')
    frames = [x for x in p.iterdir() if x.is_file()] 


    benchmark = False
    # benchmark = True

    n_frames = len(frames)

    python_t   = 1e64
    jit_t      = 1e18
    parallel_t = 1e18
    cuda_t     = 1e18

    for iteration, frame_path in enumerate(frames):
        image = Image.open(str(frame_path)) # PIL
        # image = image.convert("RGB") # PIL
        image = np.array(image) # numpy

        image = image.astype(np.float32)
        ff = np.ones((15,15), dtype=image.dtype)
       
        # (process)
        if benchmark and iteration == 0:
            ss = timer()
            result = filter3d(image, ff)
            python_t = timer() - ss

        # (numba)
        if benchmark and iteration < 5:
            ss = timer()
            result = filter3d_jit(image, ff)
            jit_t = min(jit_t, timer() - ss)
            

        # (numba_parallel)
        if benchmark and iteration < 5:
            ss = timer()
            result = filter3d_parallel(image, ff)
            parallel_t = min(parallel_t, timer() - ss)
            
        # (numba_cuda)
        ss = timer()
        result = filter3d_cuda(image, ff)
        cuda_t = min(cuda_t, timer() - ss)
        
        print(cuda_t)
        
        result_pil = Image.fromarray(result.astype(np.uint8)) # PIL
        # result_pil = Image.fromarray(result, mode='RGB') # PIL
        result_pil.save(f"{path}/{frame_path.name}")
    
    print("python:",       python_t * n_frames,   's.')
    print("JIT:",          jit_t * n_frames,      's.')
    print("JIT+parallel:", parallel_t * n_frames, 's.')
    print("JIT+cuda:",     cuda_t * n_frames,     's.')
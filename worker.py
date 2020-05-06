from pathlib import Path
from PIL import Image
import numpy as np
from timeit import default_timer as timer

from numba import jit, njit, prange


MOVIE_TITLE = "2sec480p"

@njit(parallel=True)
def filter2d_core_parallel(image, filt, result):
    M, N, D = image.shape
    Mf, Nf = filt.shape
    Mf2 = Mf // 2
    Nf2 = Nf // 2
    for d in prange(D):
        for i in prange(Mf2, M - Mf2):
            for j in prange(Nf2, N - Nf2):
                num = 0
                for ii in prange(Mf):
                    for jj in prange(Nf):
                        num += (filt[Mf-1-ii, Nf-1-jj] * image[i-Mf2+ii,j-Nf2+jj,d])
                result[i, j, d] = num / (Mf * Nf)


@njit
def filter2d_parallel(image, filt):
    result = np.zeros_like(image)
    filter2d_core_parallel(image, filt, result)
    return result

@njit
def filter2d_core_jit(image, filt, result):
    M, N, D = image.shape
    Mf, Nf = filt.shape
    Mf2 = Mf // 2
    Nf2 = Nf // 2
    for d in range(D):
        for i in range(Mf2, M - Mf2):
            for j in range(Nf2, N - Nf2):
                num = 0
                for ii in range(Mf):
                    for jj in range(Nf):
                        num += (filt[Mf-1-ii, Nf-1-jj] * image[i-Mf2+ii,j-Nf2+jj,d])
                result[i, j, d] = num / (Mf * Nf)



@njit
def filter2d_jit(image, filt):
    result = np.zeros_like(image)
    filter2d_core_jit(image, filt, result)
    return result


def filter2d_core(image, filt, result):
    M, N, D = image.shape
    Mf, Nf = filt.shape
    Mf2 = Mf // 2
    Nf2 = Nf // 2
    for d in range(D):
        for i in range(Mf2, M - Mf2):
            for j in range(Nf2, N - Nf2):
                num = 0
                for ii in range(Mf):
                    for jj in range(Nf):
                        num += (filt[Mf-1-ii, Nf-1-jj] * image[i-Mf2+ii,j-Nf2+jj,d])
                result[i, j, d] = num / (Mf * Nf)

def filter2d(image, filt):
    result = np.zeros_like(image)
    filter2d_core(image, filt, result)
    return result



if __name__ == '__main__': 
    
    path = Path(f'frames/{MOVIE_TITLE}/gg')
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    p = Path(f'frames/{MOVIE_TITLE}')
    frames = [x for x in p.iterdir() if x.is_file()] 


    benchmark = False

    for iteration, frame_path in enumerate(frames):
        image = Image.open(str(frame_path)) # PIL
        # image = image.convert("RGB") # PIL
        image = np.array(image) # numpy

       
        # TODO(process)
        if benchmark and iteration == 0:
            ss = timer()
            image = image.astype(np.float64)
            ff = np.ones((7,7), dtype=image.dtype)
            result = filter2d(image, ff)
            print("python:", timer() - ss)

        # TODO(numba)
        if benchmark and iteration < 5:
            ss = timer()
            image = image.astype(np.float64)
            ff = np.ones((15,15), dtype=image.dtype)
            result = filter2d_jit(image, ff)
            print("JIT:", timer() - ss)

        # TODO(numba_parallel)

        ss = timer()
        image = image.astype(np.float64)
        ff = np.ones((15,15), dtype=image.dtype)
        result = filter2d_parallel(image, ff)
        print("JIT+parallel:", timer() - ss)

        # TODO(numba_cuda)


        result_pil = Image.fromarray(result.astype(np.uint8)) # PIL
        # result_pil = Image.fromarray(result, mode='RGB') # PIL
        result_pil.save(f"{path}/{frame_path.name}")
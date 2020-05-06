import cv2
import numpy as np
from pathlib import Path
import time


MOVIE_TITLE = "10sec4k"
fps = 30

def get_frames(path):
    images = [x for x in path.glob('*') if x.is_file()]
    images.sort(key=lambda x: int(x.name.split('.')[0]))
    return [str(x) for x in images]
 
if __name__ == "__main__":

    image_size = None

    path = Path(f"frames/{MOVIE_TITLE}")
    path_list = get_frames(path)

    img_array = []
    for filename in path_list:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        image_size = (width, height)
        img_array.append(img)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out = cv2.VideoWriter(f"{MOVIE_TITLE}-gg-{timestamp}.mp4",cv2.VideoWriter_fourcc(*'XVID'), fps, image_size)

    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()
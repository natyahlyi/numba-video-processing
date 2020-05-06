import cv2 
from pathlib import Path


MOVIE_TITLE = "2sec480p"


# Function to extract frames 
def FrameCapture(path): 
    # Path to video file 
    vidObj = cv2.VideoCapture(f"{MOVIE_TITLE}.mp4") 
  
    # Used as counter variable 
    count = 0
    # checks whether frames were extracted 
    success = 1
  
    while success: 
        # vidObj object calls read 
        # function extract frames 
        success, image = vidObj.read() 

        # Saves the frames with frame-count
        if success:
            cv2.imwrite(f"{path}/%d.jpg" % count, image) 
            count += 1


if __name__ == "__main__":
        
    path = Path(f'frames/{MOVIE_TITLE}')
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    FrameCapture(str(path))
    
import cv2
from visualize_cv2 import model, display_instances, class_names
import sys

args = sys.argv
if (len(args)<2):
    print("Run: python video.py video_path")
    sys.exit(0)
name = args[1]

stream = cv2.VideoCapture(name)
out = cv2.VideoWriter('output.mp4', -1, 20.0, (640,480))


while True:
    ret,frame = stream.read()
    if not ret:
        print("unable to fetch")
        break

    results = model.detect([frame],verbose=1)
    # Visualize results
    r = results[0]
    masked_image = display_instances(frame, r['rois'], r['masks'], r['class_ids'],class_names, r['scores'])
    cv2.imshow("masked_image",masked_image)
    if(cv2.waitKey(1) & 0xFF == ord('q')): break

    out.write(masked_image)

stream.release()
out.release()
cv2.destroyWindow("masked_image")

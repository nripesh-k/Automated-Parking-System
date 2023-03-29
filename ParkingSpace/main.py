import cv2
import matplotlib.pyplot as plt
import numpy as np

#Flask to show current status in a screen
from flask import Flask, render_template, Response
app = Flask(__name__)

from util import get_parking_spots, empty_or_not

# To compare two instances for each crop and look for change
def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))

# saved_mask.png
mask = './mask1.png'

video_path = './carPark1.mp4'

mask = cv2.imread(mask, 0)

cap = cv2.VideoCapture(video_path)

# Getting the boxes with connected components usage
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = get_parking_spots(connected_components)

spots_status = [None for j in spots]
diffs = [None for j in spots]

previous_frame = None # To compare each frames
frame_count = 0
step = 60

def process():
    global frame_count, step, previous_frame, spot_status, diffs
    ret = True
    while ret:
        # Looping video
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        #Read video
        rets, frame = cap.read()

        if frame_count % step == 0 and previous_frame is not None:
            for spot_indx, spot in enumerate(spots):
                x1, y1, w, h = spot

                spot_crop = frame[y1:y1 + h, x1:x1 + w, :]

                diffs[spot_indx] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])

            # print([diffs[j] for j in np.argsort(diffs)][::-1])

        if frame_count % step == 0:
            if previous_frame is None:
                arr_ = range(len(spots))
            else:
                arr_ = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4]
            for spot_indx in arr_:
                spot = spots[spot_indx]
                x1, y1, w, h = spot

                spot_crop = frame[y1:y1 + h, x1:x1 + w, :]

                spot_status = empty_or_not(spot_crop)

                spots_status[spot_indx] = spot_status

        # if frame_count % step == 0:
            previous_frame = frame.copy()

        # Spot red or green
        for spot_indx, spot in enumerate(spots):
            spot_status = spots_status[spot_indx]
            x1, y1, w, h = spots[spot_indx]

            if spot_status:
                frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
            else:
                frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)

        # Available spots text
        cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
        cv2.putText(frame, 'Available spots: {} / {}'.format(str(sum(spots_status)), str(len(spots_status))), (100, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # cv2.namedWindow('frame', cv2.WINDOW_KEEPRATIO)
        cv2.imshow('Current Status', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        frame_count += 1 #increment just by one each frame running

        # Sending every update to flask server
        processed_frame = frame
        # Convert the processed frame to a JPEG image
        ret, jpeg = cv2.imencode('.jpg', processed_frame)
        # Convert the JPEG image to bytes
        data = jpeg.tobytes()
        # Yield the bytes as a multipart response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n')
        

    cap.release()
    cv2.destroyAllWindows()

process()

@app.route('/')
def index():
    return Response(process(),
                     mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)

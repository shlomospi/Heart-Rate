# Heart-rate-measurement-using-camera
[![Alt text](https://github.com/shlomospi/Heart-Rate/blob/master/Example.JPG)]
# Abstract


- Heart Rate (HR) is one of the most important Physiological parameter and a vital indicator of people‘s physiological state
- A non-contact based system to measure Heart Rate: real-time application using camera
- Principal: extract heart rate information from eye color variation caused by blood circulation 
- Code based on the project from https://github.com/habom2310/Heart-rate-measurement-using-camera . Here, A tracker is appled to a user selected area in the video and the results can be stored in a CSV file. some parameters were calibrated for improoved preformance and a parser was added for easier control.

# Methods 
- Manually select area of interest within the eye video.
- Apply Tracker to chosen area and monitor color variation of the selected spot
- Apply band pass filter with fl = 0.8 Hz and fh = 3 Hz, which are 48 and 180 bpm respectively
- Average color value of ROI in each frame is calculate pushed to a data buffer which is 150 in length
- Ignore out liars in the measurments.
- FFT the data buffer. The highest peak is Heart rate
- Amplify color to make the color variation visible 

# Requirements
```
pip install -r requirements.txt
```


# Implementation
```
python GUI.py

```
- --verbose option for printing the tracked bounding box
- --record option for recording the results in a record.csv file
- In case of plotting graphs, run "graph_plot.py" 
- For the Eulerian Video Magnification implementation, run "amplify_color.py"

# Results
- Results Are not tested. Further adaptation is needed for this case.

# Reference
- Reading HR from variation in the color map of an area where studied in the following papers, but not for eye videos. This project is just for practice porpuses and need further adaptation for this use case.
- Real Time Heart Rate Monitoring From Facial RGB Color Video Using Webcam by H. Rahman, M.U. Ahmed, S. Begum, P. Funk
- Remote Monitoring of Heart Rate using Multispectral Imaging in Group 2, 18-551, Spring 2015 by Michael Kellman Carnegie (Mellon University), Sophia Zikanova (Carnegie Mellon University) and Bryan Phipps (Carnegie Mellon University)
- Non-contact, automated cardiac pulse measurements using video imaging and blind source separation by Ming-Zher Poh, Daniel J. McDuff, and Rosalind W. Picard
- Camera-based Heart Rate Monitoring by Janus Nørtoft Jensen and Morten Hannemose
- Graphs plotting is based on https://github.com/thearn/webcam-pulse-detector
- https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/

# Note
- Application can only detect HR for 1 people at a time
- Sudden change can cause incorrect HR calculation and lose of tracked area. In the most case, HR can be correctly detected after 10 seconds being stable infront of the camera without blinking. Please allow for some time after blinking to fix HR reading



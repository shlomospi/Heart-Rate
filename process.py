import cv2
import numpy as np
import time
from face_detection import FaceDetection
from scipy import signal
# from sklearn.decomposition import FastICA
import csv


class Process(object):
    def __init__(self):
        self.frame_in = np.zeros((10, 10, 3), np.uint8)
        self.frame_ROI = np.zeros((10, 10, 3), np.uint8)
        self.frame_out = np.zeros((10, 10, 3), np.uint8)
        self.samples = []
        self.buffer_size = 200
        self.times = [] 
        self.data_buffer = []
        self.fps = 0
        self.fft = []
        self.freqs = []
        self.t0 = time.time()
        self.bpm = 0
        self.fd = FaceDetection()
        self.bpms = []
        self.peaks = []

        self.create_record = False
        self.roiX = 2040
        self.roiY = 1062
        self.roiW = 208
        self.roiH = 164
        self.selectROI = True
        self.first_frame = True
        self.tracker = cv2.TrackerMOSSE_create()
        self.bbox = (self.roiX, self.roiY, self.roiH, self.roiW)
        self.record = []
        self.ok = False
        self.bboxnew = self.bbox
        self.verbose = False
        # self.red = np.zeros((256,256,3),np.uint8)

    def extractColor(self, frame):

        r = np.mean(frame[:, :, 0])
        # g = np.mean(frame[:, :, 1])
        # b = np.mean(frame[:,:,2])
        # return r, g, b
        return r

    def run(self):

        # frame, face_frame, ROI1, ROI2, status, mask = self.fd.face_detect(self.frame_in)
        if self.frame_in is None and self.create_record:
            with open("record.csv", "a") as fp:
                print("Opening CSV")
                wr = csv.writer(fp, dialect='excel')
                wr.writerow(self.record)
        self.frame_out = self.frame_in.copy()
        # self.frame_out = frame
        # self.frame_ROI = face_frame
        
        # g1 = self.extractColor(ROI1)
        # g2 = self.extractColor(ROI2)
        # #g3 = self.extractColor(ROI3)

        L = len(self.data_buffer)
        
        # #calculate average green value of 2 ROIs
        # #r = (r1+r2)/2
        # g = (g1+g2)/2
        # #b = (b1+b2)/2

        if self.first_frame:  # first frame

            resized = cv2.resize(self.frame_out, (self.frame_out.shape[1] // 2, self.frame_out.shape[0] // 2),
                                 interpolation=cv2.INTER_AREA)
            if self.selectROI:
                self.bbox = cv2.selectROI("Tracking", resized, fromCenter=False,
                                          showCrosshair=True)
                cv2.destroyWindow("Tracking")
                self.bbox = (self.bbox[0]*2, self.bbox[1]*2, self.bbox[2]*2, self.bbox[3]*2)
            if self.create_record:
                self.record.append(self.bbox)
            self.ok = self.tracker.init(self.frame_out, self.bbox)

            self.first_frame = False

        else:  # sh not first frame

            self.ok, self.bboxnew = self.tracker.update(self.frame_out)

            if self.ok:
                # Tracking success
                self.bbox = self.bboxnew
                cv2.putText(self.frame_out, "Tracking", (75, 75),
                            cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 4)
            else:
                cv2.putText(self.frame_out, "Not Tracking", (75, 75),
                            cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 4)
            if self.verbose:
                print(self.ok, self.bbox)
        self.frame_ROI = self.frame_out.copy()[int(self.bbox[1]):int(self.bbox[1] + self.bbox[3]),
                                               int(self.bbox[0]):int(self.bbox[0] + self.bbox[2])]
        cv2.rectangle(self.frame_out, (int(self.bbox[0]), int(self.bbox[1])),
                      (int(self.bbox[0] + self.bbox[2]), int(self.bbox[1] + self.bbox[3])), (255, 0, 0), 6, 1)

        g = self.extractColor(self.frame_ROI)

        if abs(g - np.mean(self.data_buffer)) > 10 and L > 99:
            # remove sudden change, if the avg value change is over 10, use the mean of the data_buffer
            g = self.data_buffer[-1]

        self.times.append(time.time() - self.t0)
        self.data_buffer.append(g)

        # only process in a fixed-size buffer
        if L > self.buffer_size:
            self.data_buffer = self.data_buffer[-self.buffer_size:]
            self.times = self.times[-self.buffer_size:]
            self.bpms = self.bpms[-self.buffer_size//2:]
            L = self.buffer_size
            
        processed = np.array(self.data_buffer)

        # start calculating after the first 10 frames
        if L == self.buffer_size:
            
            self.fps = float(L) / (self.times[-1] - self.times[0])
            # calculate HR using a true fps of processor of the computer, not the fps the camera provide
            even_times = np.linspace(self.times[0], self.times[-1], L)
            processed = signal.detrend(processed)  # detrend the signal to avoid interference of light change
            interpolated = np.interp(even_times, self.times, processed)  # interpolation by 1
            interpolated = np.hamming(L) * interpolated
            # make the signal become more periodic (advoid spectral leakage)
            # norm = (interpolated - np.mean(interpolated))/np.std(interpolated)#normalization
            norm = interpolated/np.linalg.norm(interpolated)
            raw = np.fft.rfft(norm*30)  # do real fft with the normalization multiplied by 10
            
            self.freqs = float(self.fps) / L * np.arange(L / 2 + 1)
            freqs = 60. * self.freqs
            
            # idx_remove = np.where((freqs < 50) & (freqs > 180))
            # raw[idx_remove] = 0
            
            self.fft = np.abs(raw)**2  # get amplitude spectrum
        
            idx = np.where((freqs > 50) & (freqs < 180))  # the range of frequency that HR is supposed to be within
            pruned = self.fft[idx]
            pfreq = freqs[idx]
            
            self.freqs = pfreq 
            self.fft = pruned
            
            idx2 = np.argmax(pruned)  # max in the range can be HR
            
            self.bpm = self.freqs[idx2]
            self.bpms.append(self.bpm)
            if self.create_record:
                self.record.append(np.mean(self.bpms))

            processed = self.butter_bandpass_filter(processed, 0.8, 3, self.fps,order = 3)
            # ifft = np.fft.irfft(raw)
        self.samples = processed  # multiply the signal with 5 for easier to see in the plot

    def reset(self):

        self.frame_in = np.zeros((10, 10, 3), np.uint8)
        self.frame_ROI = np.zeros((10, 10, 3), np.uint8)
        self.frame_out = np.zeros((10, 10, 3), np.uint8)
        self.samples = []
        self.times = [] 
        self.data_buffer = []
        self.fps = 0
        self.fft = []
        self.freqs = []
        self.t0 = time.time()
        self.bpm = 0
        self.bpms = []
        
    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = signal.lfilter(b, a, data)
        return y    


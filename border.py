import glob
import json
import time

from centroidtracker import CentroidTracker
from objectTracker import ObjectTracker
import numpy as np
import imutils
import dlib
import cv2
import Constants
import os


class Log:

    def __init__(self):

        self.logTable = dict()
        self.visited_skip_frames = dict()

    def saveToDisk(self, vid_name, total_in, total_out, total_time):
        vid_name = os.path.basename(vid_name).split(".")[0]
        path = "logs/" + str(vid_name)
        os.makedirs("logs", exist_ok=True)
        os.makedirs(path, exist_ok=True)
        files = glob.glob(path + "/*")
        for file in files:
            if "log.json" in file:
                with open(file) as input_file:
                    self.logTable = json.load(input_file)
                    input_file.close()
            else:
                with open(file) as input_file2:
                    self.visited_skip_frames = json.load(input_file2)
                    input_file2.close()

        try:
            self.visited_skip_frames[str(Constants.ABSENCE_BEFORE_REMOVE)]
        except Exception as e:
            self.visited_skip_frames[str(Constants.ABSENCE_BEFORE_REMOVE)] = []

        try:
            self.logTable[str(Constants.ABSENCE_BEFORE_REMOVE)]
        except Exception as e:
            self.logTable[str(Constants.ABSENCE_BEFORE_REMOVE)] = []

        if str(Constants.SKIP_FRAMES) not in self.visited_skip_frames[str(Constants.ABSENCE_BEFORE_REMOVE)]:
            self.logTable[str(Constants.ABSENCE_BEFORE_REMOVE)].append((Constants.SKIP_FRAMES, total_in, total_out, total_time))
            self.visited_skip_frames[str(Constants.ABSENCE_BEFORE_REMOVE)].append(str(Constants.SKIP_FRAMES))

        # Save the current test values.
        with open(path + "/log.json", "w") as output:
            json.dump(self.logTable, output)
            output.close()

        with open(path + "/skip_frames.json", "w") as output2:
            json.dump(self.visited_skip_frames, output2)
            output2.close()


class Border:
    """
    class that represents the pass-way between two areas in the parking lot
    e.g.: from floor -1 to floor -2
    """

    def __init__(self, section_up, section_down, video, net, tid):
        """

        :param section_up: the section that cars that are driving upwards are entering into
        :param section_down: the section that cars that are driving downwards are entering into
        :param video: the video. in the future will be a live string
        :param net: the trained CNN
        :param tid: for debug reasons. we should remember to delete this.
        """
        self.section_up = section_up
        self.section_down = section_down
        self.video = video
        self.tid = tid
        self.net = net
        self.vs = cv2.VideoCapture(self.video)
        self.logger = Log()

    def update_car_down(self):
        """
        update the system that a car was tracked driving "down"
        :return: no return
        """
        self.section_up.update_car_exited()
        self.section_down.update_car_entered()

    def update_car_up(self):
        """
        update the system that a car was tracked driving "up"
        :return: no return
        """
        self.section_up.update_car_entered()
        self.section_down.update_car_exited()


    def add_info_on_screen(self, frame, info, tracked_objects, rects):
        (height, width) = frame.shape[:2]
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, height - ((i * 20) + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        upper_line_height = int(height / Constants.UPPER_LINE_PLACE)
        lower_line_height = int(height / Constants.LOWER_LINE_PLACE)
        cv2.line(frame, (0, upper_line_height), (width, upper_line_height), (0, 255, 255), 2)
        cv2.line(frame, (0, lower_line_height), (width, lower_line_height), (0, 255, 255), 2)
        for (objectID, centroid) in tracked_objects:
            text = str(objectID + 1)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        # add rectangles around the identified objects
        for rect in rects:
            x, y, w, h = rect
            cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), 1)



    def start_counting(self):


            width, height = None, None
            writer = None

            # absence before remove = how many frames can the vehicle not be detected before removing the id
            # max distance = max distance between objects (between frames) to give them the same id
            ct = CentroidTracker(maxDisappeared=Constants.ABSENCE_BEFORE_REMOVE, maxDistance=Constants.MAX_DISTANCE_FROM_CENTROID)

            # for the dlib correlation trackers
            trackers = []

            # map each unique object ID to a Object tracker
            object_trackers = {} # id:tracker

            # initialize the total number of vehicles that have moved either up or down
            totalFrames = 0
            totalDown = 0
            totalUp = 0

            # loop over frames from the video stream
            print("########################## TID: ", self.tid, " #######################")
            vid_name = self.video.split("/")[1]
            vid_name = Constants.OUTPUT_PATH + vid_name.split(".")[0]

            start = time.time()
            while True:
                # time.sleep(0.3)
                # grab the next frame and handle
                frame = self.vs.read()[1]


                # if we did not grab a frame then we have reached the end of the video
                if frame is None:
                    break

                # resize the frame to have a maximum width of 500 pixels (the
                # less data we have, the faster we can process it), then convert
                # the frame from BGR to RGB for dlib tracker
                frame = imutils.resize(frame, width=500)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # if the frame dimensions are empty, it is the first frame so
                # set them and start the writer
                if width is None or height is None:
                    (height, width) = frame.shape[:2]
                    # initialize the writer
                    fourcc = cv2.VideoWriter_fourcc(*"XVID")
                    writer = cv2.VideoWriter(vid_name + ".avi", fourcc, 30, (width, height))  # maybe add true

                # initialize  our list of bounding
                # box rectangles returned by either our object detector or
                # the dlib correlation tracker
                # these rects will be "fed" to the centroid trackers
                rects = []

                # check to see if we should run a more computationally expensive
                # object detection method to aid our tracker
                if totalFrames % Constants.SKIP_FRAMES == 0:
                    # initialize our new set of object trackers
                    trackers = []

                    # convert the frame to a blob and pass the blob through the
                    # network and obtain the detections
                    blob = cv2.dnn.blobFromImage(frame, 0.007843, (width, height), 127.5)
                    self.net.setInput(blob)
                    detections = self.net.forward()

                    # loop over the detections
                    for i in np.arange(0, detections.shape[2]):
                        # extract the confidence (i.e., probability) associated
                        # with the prediction
                        confidence = detections[0, 0, i, 2]

                        # filter out weak detections by requiring a minimum
                        # confidence
                        if confidence > Constants.CONFIDENCE:
                            # extract the index of the class label from the
                            # detections list
                            idx = int(detections[0, 0, i, 1])

                            # filter out irrelevant objects
                            if Constants.CLASSES[idx] not in ["car", "bus", "motorbike"]:
                                continue

                            # compute the (x, y)-coordinates of the bounding box
                            # for the object
                            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                            startX = int(box[0])
                            startY = int(box[1])
                            endX = int(box[2])
                            endY = int(box[3])


                            # construct a dlib rectangle object from the bounding
                            # box coordinates to start the dlib correlation
                            # tracker
                            tracker = dlib.correlation_tracker()
                            rect = dlib.rectangle(startX, startY, endX, endY)
                            tracker.start_track(rgb, rect)

                            # add the tracker to our list of trackers so we can
                            # utilize it during skip frames
                            trackers.append(tracker)

                else:
                    # Track Objects
                    for tracker in trackers:

                        # update the tracker and grab the updated position
                        tracker.update(rgb)
                        pos = tracker.get_position()

                        # unpack the position object
                        startX = int(pos.left())
                        startY = int(pos.top())
                        endX = int(pos.right())
                        endY = int(pos.bottom())

                        # add the bounding box coordinates to the rectangles list
                        rects.append((startX, startY, endX, endY))

                # draw horizontal lines in the frame -- once an
                # object crosses these lines we will determine whether they were
                # moving 'up' or 'down'
                upper_line_height = int(height / Constants.UPPER_LINE_PLACE)
                lower_line_height = int(height / Constants.LOWER_LINE_PLACE)


                # use the centroid tracker to associate the old object
                # centroids with the newly computed object centroids
                objects = ct.update(rects)

                # loop over the tracked objects
                for (objectID, centroid) in objects.items():
                    # check to see if a trackable object exists for the current
                    # object ID
                    t_o = object_trackers.get(objectID, None)

                    # if there is no existing trackable object, create one
                    if t_o is None:
                        t_o = ObjectTracker(objectID, centroid)

                    # otherwise, there is a trackable object so we can use it
                    # to determine direction
                    else:
                        # if curr(y)>prev(y's) then the object is going up.
                        # otherwise it is going down.
                        y = [c[1] for c in t_o.centroids]
                        direction = centroid[1] - np.mean(y)
                        t_o.centroids.append(centroid)

                        # check to see if the object has been counted or not (TO PREVENT DOUBLE COUNTING!!)
                        if not t_o.counted:
                            # if the direction is negative (indicating the object
                            # is moving up) AND the centroid is above the upper
                            # line, count the object (and it wasn't counted of course)
                            if direction < 0 and centroid[1] < upper_line_height:
                                totalUp += 1
                                self.update_car_up()
                                t_o.counted = True

                            # same idea for cars going down
                            elif direction > 0 and centroid[1] > lower_line_height:
                                totalDown += 1
                                self.update_car_down()
                                t_o.counted = True

                    # store the trackable object in our dictionary
                    object_trackers[objectID] = t_o

                # information to display on the frame
                info = [("Up", totalUp),
                        ("Down", totalDown)]
                self.add_info_on_screen(frame, info, objects.items(), rects)

                writer.write(frame)

                # can't imshow with multi-thread !!
                # cv2.imshow("Frame" + str(self.tid), frame)
                key = cv2.waitKey(1) & 0xFF

                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                    break

                totalFrames += 1

            # print information
            length = time.time() - start
            self.logger.saveToDisk(self.video, totalUp, totalDown, length)
            print("DOWN: ", totalDown)
            print("UP: ", totalUp)
            print("########################## END TID: ", self.tid, " #######################")
            writer.release()
            self.vs.release()






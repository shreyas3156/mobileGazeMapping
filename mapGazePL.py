""" Map gaze data from world camera coordinate system to reference image

This script automates mapping of gaze data from a world camera coordinate
system to a fixed reference image. Mobile eye-trackers often record gaze data
with respect to an outward facing world camera approximating the
participant's point-of-view. As a result, the gaze data is expressed in
an egocentric coordinate system which moves along with the participant's head.

Typical eye-tracking research, on the other hand, seeks to analyze gaze
behavior on a particular stimulus over time. In order to use mobile
eye-trackers in this context, one must first map the recorded gaze points from
the world camera coordinate system to the fixed coordinate system of the
target stimulus. This requires 1) identifying the target stimulus in every
frame of the world camera recording, 2) finding a linear transform that will
map between the appearance of the stimulus on the world camera frame and a 2D
reference version of the same stimulus, and 3) using that transform to project
the recorded gaze points to the 2D reference stimulus.

With the help of computers vision tools, this script automates this process
and yeilds output data files that facilitate subsequent analysis, specifically:
    - world_gaze.m4v:           world video w/ gaze points overlaid
    - ref_gaze.m4v:             video of ref image w/ gaze points overlaid
    - ref2world_mapping.m4v     video of reference image projected back into
                                world video
    - gazeData_mapped.tsv:      gazeData mapped to both coordinate systems, the
                                world and reference image

"""

# python 2/3 compatibility
from __future__ import division
from __future__ import print_function

import av
import os
import sys
from os.path import join
import logging
import shutil
import time
import argparse

import numpy as np
import pandas as pd
import cv2
from ultralytics import YOLO

OPENCV3 = (cv2.__version__.split('.')[0] == '3')
print("OPENCV version " + cv2.__version__)


def findMatches(img1_kp, img1_des, img2_kp, img2_des):
    """ Find the matches between the descriptors for two images

    Parameters
    ----------
    img1_kp, img2_kp : list
        list of identified keypoints for each image; returned from
        detectAndCompute method on the cv2 featureDetect class.
    img1_des, img2_des : np.ndarray
        descriptors for each image; returned from detectAndCompute method on
        the cv2 featureDetect class.

    Returns
    -------
    img1_pts, img2_pts : list or None
        list of matched keypoints on each image

    """
    # Match settings
    min_good_matches = 4
    num_matches = 2
    FLANN_INDEX_KDTREE = 0
    distance_ratio = 0.6                # 0-1; lower values more conservative
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=10)        # lower = faster, less accurate
    matcher = cv2.FlannBasedMatcher(index_params, search_params)

    # find all matches
    matches = matcher.knnMatch(img1_des, img2_des, k=num_matches)

    # filter out cases where the 2 matches are too close to each other
    goodMatches = []
    for m, n in matches:
        if m.distance < distance_ratio * n.distance:
            goodMatches.append(m)

    if len(goodMatches) > min_good_matches:
        img1_pts = np.float32([img1_kp[i.queryIdx].pt for i in goodMatches])
        img2_pts = np.float32([img2_kp[i.trainIdx].pt for i in goodMatches])

        return img1_pts, img2_pts

    else:
        return None, None


def mapCoords2D(coords, transform2D):
    """ Map the supplied coords to a new coordinate system using the supplied
    transformation matrix

    Parameters
    ----------
    coords : tuple
        (x,y) coordinates
    transform2D : np.ndarray
        2D transformation matrix; produce by cv2.findHomography

    Returns
    -------
    float, float
        mapped coordinates after applying transform2D

    """

    coords = np.array(coords).reshape(-1, 1, 2)
    mappedCoords = cv2.perspectiveTransform(coords, transform2D)
    mappedCoords = np.round(mappedCoords.ravel())

    return mappedCoords[0], mappedCoords[1]


def projectImage2D(origFrame, transform2D, newImage):
    """ Project newImage into the origFrame

    Warp newImage according to the supplied transformation matrix, then
    project (insert) into the original frame.

    Parameters
    ----------
    origFrame : np.ndarray
        The original image you want to insert the newImage into
    transform2D : np.ndarray
        2D transformation matrix; produce by cv2.findHomography
    newImage : np.ndarray
        The image you would like to warp and project into the origFrame


    Returns
    -------
    newFrame : np.ndarray
        New frame (same dimensions as origFrame) with the warped and projected
        newImage written into it

    """
    # warp the new image to the video frame
    warpedImage = cv2.warpPerspective(newImage,
                                      transform2D,
                                      origFrame.T.shape[1:])

    # mask and subtract new image from video frame
    warpedImage_bw = cv2.cvtColor(warpedImage, cv2.COLOR_BGR2GRAY)
    if warpedImage.shape[2] == 4:
        alpha = warpedImage[:, :, 3]
        alpha[alpha == 255] = 1       # create mask of non-transparent pixels
        warpedImage_bw = cv2.multiply(warpedImage_bw, alpha)

    ret, mask = cv2.threshold(warpedImage_bw, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    origFrame_bg = cv2.bitwise_and(origFrame, origFrame, mask=mask_inv)

    # mask the warped new image, and add to the masked background frame
    warpedImage_fg = cv2.bitwise_and(warpedImage[:, :, :3],
                                     warpedImage[:, :, :3],
                                     mask=mask)
    newFrame = cv2.add(origFrame_bg, warpedImage_fg)

    # return the warped new frame
    return newFrame


def extract_timestamps(video_path):
    """
    Extract timestamps (in seconds) for each frame using PyAV.
    """
    input_container = av.open(video_path)
    input_stream = input_container.streams.video[0]

    frame_rate = input_stream.average_rate  # Get FPS as Fraction (e.g., 30/1)
    frame_duration_ns = int(1 / frame_rate * 1e9)

    timestamps = []  # Store timestamps
    for frame_idx, frame in enumerate(input_container.decode(video=0)):  # Iterate through frames
        current_frame_timestamp = frame_idx * frame_duration_ns
        timestamps.append(current_frame_timestamp)

    timestamps = np.array(timestamps)

    input_container.close()
    return np.array(timestamps), float(frame_rate)


def processRecording(gazeData=None, worldCameraVid=None, screenVid=None, outputDir=None, nFrames=None):
    """ Map the gaze across all frames of mobile eye-tracking session

    This method will iterate over every frame of the supplied video recording.
    On each frame, it will look for the matches with the specified
    screenVid, create a linear transformation matrix, and map the gaze
    data from the world camera coordinate system to the reference image
    coordinate system.

    This parent method will take care of setting up all of the inputs, and at
    the end, writing all of the output files

    Parameters
    ----------
    gazeData : string
        Path to the gazeData file. This file expected to be a .csv/.tsv file
        with columns for:
            timestamp - timestamp (ms) corresponding to each sample
            frame_idx - index (0-based) of the worldCameraVid frame
                        corresponding to each sample
            confidence - confidence of the validity of each sample (0-1)
            norm_pos_x - normalized x position of gaze location (0-1).
                         Normalized with respect to width of worldCameraVid
            norm_pos_y - normalized y position of gaze location (0-1).
                         Normalized with respect to height of worldCameraVid
    worldCameraVid : string
        Path to the video recording from the world camera (.mp4)
    screenVid : string
        Path to the 2D reference image
    outputDir : string
        Path to output directory where data will be saved
    nFrames : int, optional
        If specified, will only process given number of frames (default of
        None means it will process ALL frames in the video). Useful for testing
        on abbreviated number of frames

    Output files
    ------------
    world_gaze.m4v : video
        world video with original gaze points overlaid
    ref_gaze.m4v : video
         ref image with mapped gaze points overlaid
    ref2world_mapping.m4v : video
        world video with reference image projected and inserted into it.
    gazeData_mapped.tsv :  data file
        gazeData represented in both coordinate systems, the world and
        reference image

    """
    # Create output directory
    if not os.path.isdir(outputDir):
        os.makedirs(outputDir)

    # Set up Logging
    fileLogger = logging.FileHandler(join(outputDir, 'mapGazeLog.log'), mode='w')
    fileLogger.setLevel(logging.DEBUG)
    fileLogFormat = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%m-%d %H:%M:%S')
    fileLogger.setFormatter(fileLogFormat)
    consoleLogger = logging.StreamHandler(sys.stdout)
    consoleLogger.setLevel(logging.INFO)
    consoleLogFormat = logging.Formatter('%(message)s')
    consoleLogger.setFormatter(consoleLogFormat)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fileLogger)
    logger.addHandler(consoleLogger)

    # Log Inputs
    logger.info('Gaze Data File: {}'.format(gazeData))
    logger.info('World Camera Video: {}'.format(worldCameraVid))
    logger.info('Screen Video: {}'.format(screenVid))
    logger.info('Output Directory: {}'.format(outputDir))

    # Copy the reference stim into the output dir
    shutil.copy(screenVid, outputDir)

    # Load gaze data
    gazeWorld_df = pd.read_csv(gazeData)

    # Load the reference image
    # refImg = cv2.imread(join(outputDir, screenVid.split('/')[-1]))
    # refImgColor = refImg.copy()      # store a color copy of the image
    # refImg = cv2.cvtColor(refImg, cv2.COLOR_BGR2GRAY)  # convert the orig to bw

    ### Prep the video data #######################################
    # Load the videos, get parameters
    vid1 = cv2.VideoCapture(worldCameraVid)
    vid2 = cv2.VideoCapture(screenVid)
    vid_size1 = (int(vid1.get(cv2.CAP_PROP_FRAME_WIDTH)),
               int(vid1.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    vid_size2 = (int(vid2.get(cv2.CAP_PROP_FRAME_WIDTH)),
               int(vid2.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    vid_codec = cv2.VideoWriter_fourcc(*'mp4v')
    feature_detect = cv2.SIFT_create()

    # don't need to extract timestamps for WorldCameraVid since we already have them in the gaze data.
    _, fps1 = extract_timestamps(worldCameraVid)
    timestamps_screen, fps2 = extract_timestamps(screenVid)
    timestamps_world = gazeWorld_df['timestamp']

    # align the timestamps of vid2 with vid1
    # find the timestamps of frames
    frames_idx_world = gazeWorld_df['frame_idx']
    aligned_frames_idx_screen = [np.argmin(np.abs(timestamps_screen - t)) for t in timestamps_world]
    # aligned_timestamps_screen = [timestamps_screen[np.argmin(np.abs(timestamps_screen - t))] for t in timestamps_world]

    # Screen recording mapping output video
    vidOut_ref_fname = join(outputDir, 'screen_gaze-5.mp4')
    vidOut_ref = cv2.VideoWriter()
    vidOut_ref.open(vidOut_ref_fname,
                    vid_codec,
                    fps2,
                    vid_size2,
                    True)

    ### Find keypoints, descriptors for the reference image
    # refImg_kp, refImg_des = feature_detect.detectAndCompute(refImg, None)
    # logger.info('Reference Image: found {} keypoints'.format(len(refImg_kp)))

    ### Loop over video frames ###############################################
    # if nFrames and nFrames < totalFrames:
    #     framesToUse = np.arange(0, nFrames, 1)
    # else:
    #     framesToUse = np.arange(0, totalFrames, 1)
    frameProcessing_startTime = time.time()
    frameCounter = 0

    for frame_world_idx, frame_screen_idx in zip(frames_idx_world, aligned_frames_idx_screen):
        if not vid1.isOpened() or not vid2.isOpened():
            print("Error: Could not open video.")
            return None

        vid1.set(cv2.CAP_PROP_POS_FRAMES, frame_world_idx)
        vid2.set(cv2.CAP_PROP_POS_FRAMES, frame_screen_idx)

        ret1, frame1 = vid1.read()
        ret2, frame2 = vid2.read()

        #### Sanity check
        # if frame_world_idx < 134:
        #     continue
        # if all([ret1, ret2, frame1 is not None, frame2 is not None]):
        #     cv2.imshow("Extracted Frame 1", frame1)
        #     cv2.imshow("Extracted Frame 2", frame2)
        #     cv2.waitKey(0)  # Wait indefinitely until a key is pressed
        #     cv2.destroyAllWindows()  # Close the window
        # break

        # check if it's a valid frame
        if all([ret1, ret2, frame1 is not None, frame2 is not None]):

            # make copy of the reference image for later use
            screen_frame_copy = frame2.copy()

            # process this frame
            processedFrame = processFrame(frame1,
                                          frame2,
                                          frame_world_idx,
                                          feature_detect)

            # if good match between reference image and this frame
            if processedFrame['foundGoodMatch']:

                # grab the gaze data (world coords) for this frame
                thisFrame_gazeData_world = gazeWorld_df.loc[gazeWorld_df['frame_idx'] == frame_world_idx]

                # project the reference image back into the video as a way to check for good mapping
                 # screen2world_frame = projectImage2D(processedFrame['origFrame1'], processedFrame['origFrame2'], screen_frame_copy)

                # loop over all gaze data for this frame, translate to different coordinate systems
                for i, gazeRow in thisFrame_gazeData_world.iterrows():
                    ts = gazeRow['timestamp']
                    conf = gazeRow['confidence']

                    # translate normalized gaze data to world pixel coords
                    world_gazeX = gazeRow['norm_pos_x'] * processedFrame['frame_gray1'].shape[1]
                    world_gazeY = gazeRow['norm_pos_y'] * processedFrame['frame_gray1'].shape[0]

                    # convert from world to screen pixel coordinates
                    screen_gazeX, screen_gazeY = mapCoords2D((world_gazeX, world_gazeY), processedFrame['world2screen'])

                    # create dict for this row
                    thisRow_df = pd.DataFrame({'gaze_ts': ts,
                                               'worldFrame': frameCounter,
                                               'confidence': conf,
                                               'world_gazeX': world_gazeX,
                                               'world_gazeY': world_gazeY,
                                               'screen_gazeX': screen_gazeX,
                                               'screen_gazeY': screen_gazeY},
                                               index=[i])

                    # append row to gazeMapped_df output
                    if 'gazeMapped_df' in locals():
                        gazeMapped_df = pd.concat([gazeMapped_df, thisRow_df])
                    else:
                        gazeMapped_df = thisRow_df

                    ### Draw gaze circles on frames
                    if i == thisFrame_gazeData_world.index.max():
                        dotColor = [96, 52, 234]            # pinkish/red
                        dotSize = 12
                    else:
                        dotColor = [168, 231, 86]            # minty green
                        dotSize = 8

                    # screen frame
                    cv2.circle(frame2,
                               (int(screen_gazeX), int(screen_gazeY)),
                               dotSize,
                               dotColor,
                               -1)
            else:
                # if not a good match, use the original frame for the screen2world
                screen2world_frame = processedFrame['origFrame1']

            mps_device = "mps"  # Apple ARM64
            # Load pre-trained YOLOv8 model (default COCO model)
            model = YOLO("best-8m-15-10.pt")  # 'n' = nano model, can also use yolov8s.pt, yolov8m.pt, etc.
            model.to(mps_device)

            # Run object detection
            results = model.track(frame2, device=mps_device)

            for r in results:
                class_names = r.names
                # iterate over each box
                for box in r.boxes:
                    # check if confidence is greater than 40 percent
                    if box.conf[0] > 0.5:
                        # get coordinates
                        [x1, y1, x2, y2] = box.xyxy[0]
                        # convert to int
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        # get the class
                        cls = int(box.cls[0])

                        # get the class name
                        class_name = class_names[cls]

                        color = (0, 255, 0) if class_name == 'video' else (255, 0, 0)

                        # draw the rectangle
                        cv2.rectangle(frame2, (x1, y1), (x2, y2), color, 2)

                        # # put the class name and confidence on the image
                        # cv2.putText(frame2, f'{class_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # write outputs to video
            # vidOut_world.write(frame1)
            vidOut_ref.write(frame2)
            # vidOut_ref2world.write(screen2world_frame)

        # write out gaze data
        try:
            colOrder = ['worldFrame', 'gaze_ts', 'confidence',
                        'world_gazeX', 'world_gazeY',
                        'screen_gazeX', 'screen_gazeY']
            gazeMapped_df[colOrder].to_csv(join(outputDir, 'gazeData_mapped.tsv'),
                                           sep='\t',
                                           index=False,
                                           float_format='%.3f')
        except Exception as e:
            logger.info(e)
            logger.info('could not write gazeData_mapped to csv')
            pass
    vidOut_ref.release()
    endTime = time.time()
    frameProcessing_time = endTime - frameProcessing_startTime
    logger.info('Total time: %s seconds' % frameProcessing_time)
    # logger.info('Avg time/frame: %s seconds' % (frameProcessing_time / framesToUse.shape[0]))


def processFrame(frame1, frame2, frameIdx, feature_detect):
    """ Process single frame from the world camera to determine mapping to
    ref image

    Parameters
    ---------
    frame1 : np.ndarray
        frame from world camera video
    frame2 : np.ndarray
        frame from screen recording video
    frameIdx : int
        frame index (0-based)

    featureDetect : object
        instance of cv2 SIFT class

    Returns
    -------
    fr : dict
        dictionary with entries storing all of the relevant output for this
        particular frame

    """
    logger = logging.getLogger()

    fr = {}  # create dict to store info for this frame pair

    # create copy of original frame
    origFrame1 = frame1.copy()
    origFrame2 = frame2.copy()
    fr['origFrame1'] = origFrame1  # store
    fr['origFrame2'] = origFrame2

    frame_gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame_gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # convert to grayscale
    fr['frame_gray1'] = frame_gray1
    fr['frame_gray2'] = frame_gray2

    # try to match the frame and the reference image
    try:
        # Detect keypoints and descriptors
        kp1, des1 = feature_detect.detectAndCompute(frame_gray1, None)
        kp2, des2 = feature_detect.detectAndCompute(frame_gray2, None)

        logger.info('found {} features on screen frame {}'.format(len(kp2), frameIdx))

        if len(kp1) < 2 and len(kp2) < 2:
            screen_matchPts = None
        else:
            screen_matchPts, world_matchPts = findMatches(kp2, des2, kp1, des1)

        # check if matches were found
        try:
            numMatches = screen_matchPts.shape[0]

            # if sufficient number of matches....
            if numMatches > 10:
                logger.info('found {} matches on screen frame {}'.format(numMatches, frameIdx))
                sufficientMatches = True
            else:
                logger.info('Insufficient matches ({}} matches) on frame {}'.format(numMatches, frameIdx))
                sufficientMatches = False

        except:
            print('no matches found on frame {}'.format(frameIdx))
            sufficientMatches = False
            pass

        fr['foundGoodMatch'] = sufficientMatches

        # figure out homographies between coordinate systems
        if sufficientMatches:
            world2screen_transform, mask = cv2.findHomography(world_matchPts.reshape(-1, 1, 2),
                                                           screen_matchPts.reshape(-1, 1, 2),
                                                           cv2.RANSAC,
                                                           5.0)
            # world2ref_transform = cv2.invert(ref2world_transform)

            fr['world2screen'] = world2screen_transform
            # fr['world2ref'] = world2ref_transform[1]

    except:
        fr['foundGoodMatch'] = False

    # return the processed frame
    return fr


if __name__ == '__main__':

    # Parse arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument('gazeData',
    #                     help='path to gaze data file')
    # parser.add_argument('worldCameraVid',
    #                     help='path to world camera video file')
    # parser.add_argument('screenVid',
    #                     help='path to reference image file')
    # parser.add_argument('-o', '--outputDir',
    #                     help='output directory [default: create "mappedGazeOutput" dir in same directory as gazeData file]')
    # args = parser.parse_args()

    # Input error checking
    # badInputs = []
    # for arg in [args.gazeData, args.worldCameraVid, args.screenVid]:
    #     if not os.path.exists(arg):
    #         badInputs.append(arg)
    # if len(badInputs) > 0:
    #     [print('{} does not exist! Check your input file path'.format(x)) for x in badInputs]
    #     sys.exit()

    # Set output directory
    # if args.outputDir is None:
    #     inputDir, tmp = os.path.split(args.gazeData)
    #     outputDir = join(inputDir, 'mappedGazeOuput')
    # else:
    #     outputDir = args.outputDir

    preprocessDir = 'preprocessing'
    inputDir = 'pl_input'
    gazeData = join(preprocessDir, 'pl_preprocessed_out', 'gaze_preprocessed_yt.csv')
    worldCameraVid = join(preprocessDir, inputDir, '5.mp4')
    screenVid = join(preprocessDir, inputDir, 'ad-5-screenrec-yt.mp4')
    outputDir = join(preprocessDir, inputDir, 'test_output')

    ## process the recording
    print('processing the recording...')
    print('Output saved in: {}'.format(outputDir))
    processRecording(gazeData=gazeData,
                     worldCameraVid=worldCameraVid,
                     screenVid=screenVid,
                     outputDir=outputDir)

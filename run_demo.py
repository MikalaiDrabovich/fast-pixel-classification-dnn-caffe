# coding=utf-8
"""
Fast object recognition with a pretrained deep neural net of road objects.
Recognized objects will have the same colors as in CityScapes dataset. 
See __main__ for the settings.

Mikalai Drabovich (nick.drabovich@amd.com)
"""
from __future__ import print_function
import os
import sys
import time

import cv2
import numpy as np
import cProfile
import weave

#if necessary, use custom caffe build
#sys.path.insert(0, 'build/install/python')
import caffe


def id2bgr(im):
    """
    A fast conversion from object id to color.
    :param im: 2d array with shape (w,h) with recognized object IDs as pixel values
    :return: color_image: BGR image with colors corresponding to detected object.
    The BGR values are compatible with CityScapes dataset:
    github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
    """
    w, h = im.shape
    color_image = np.empty((w, h, 3), dtype=np.uint8)
    code = """
    unsigned char cityscape_object_colors[19][3] = {
    {128, 64,128}, // road
    {232, 244, 35}, // sidewalk
    { 70, 70, 70}, // building
    {156, 102,102}, // wall
    {190,153,153}, // fence
    {153,153,153}, // pole
    {30,170, 250}, // traffic light
    {0, 220,  220}, // traffic sign
    {35,142, 107}, // vegetation
    {152,251,152}, // terrain
    { 180, 130,70}, // sky
    {60, 20, 220}, // person
    {0,  0,  255}, // rider
    { 142,  0,0}, // car
    { 70,  0, 0}, // truck
    {  100, 60,0}, // bus
    {  100, 80,0}, // train
    {  230,  0,0}, // motorcycle
    {32, 11, 119}  // bicycle
    };

    int impos=0;
    int retpos=0;
    for(int j=0; j<Nim[0]; j++) {
        for (int i=0; i<Nim[1]; i++) {
            unsigned char d=im[impos++];
            color_image[retpos++] = cityscape_object_colors[d][0];
            color_image[retpos++] = cityscape_object_colors[d][1];
            color_image[retpos++] = cityscape_object_colors[d][2];
        }
    }
    """
    weave.inline(code, ["im", "color_image"])
    return color_image


def fast_mean_subtraction_bgr(im):
    """
    Fast mean subtraction
    :param im: input image
    :return: image with subtracted mean values of ImageNet dataset
    """
    code = """
    float mean_r = 123;
    float mean_g = 117;
    float mean_b = 104;
    int retpos=0;
    for(int j=0; j<Nim[0]; j++) {
        for (int i=0; i<Nim[1]; i++) {
            im[retpos++] -=  mean_b;
            im[retpos++] -=  mean_g;
            im[retpos++] -=  mean_r;
        }
    }
    """
    weave.inline(code, ["im"])
    return im


def feed_and_run(input_frame):
    """
    Format input data and run object recognition 
    :param input_frame: image data from file
    :return: forward_time, segmentation_result
    """
    start = time.time()
    input_frame = np.array(input_frame, dtype=np.float32)
    input_frame = fast_mean_subtraction_bgr(input_frame)
    input_frame = input_frame.transpose((2, 0, 1))
    net.blobs['data'].data[...] = input_frame
    print("Data input took {} ms.".format(round((time.time() - start) * 1000)))

    start = time.time()
    net.forward()
    forward_time = round((time.time() - start) * 1000)
    print("Net.forward() took {} ms.".format(forward_time))

    start = time.time()

    result_with_train_ids = net.blobs['score'].data[0].argmax(axis=0).astype(np.uint8)

    print("ArgMax took {} ms.".format(round((time.time() - start) * 1000)))

    start = time.time()
    segmentation_result = id2bgr(result_with_train_ids)
    print("Conversion from object class ID to color took {} ms.".format(round((time.time() - start) * 1000)))

    return forward_time, segmentation_result


if __name__ == "__main__":

    #------------------------ Change main parameters here ---------------

    model_weights = './models/this_needs_to_be_provided_by_you.caffemodel'
    model_description = './models/this_needs_to_be_provided_by_you.prototxt'

    createVideoFromResults = False
    show_gui = False
    save_results = False
    results_folder = './results/'

    input_w = 2048
    input_h = 1024

    #--------------------------------------------------------------------

    profiler = cProfile.Profile()
    profiler.enable()

    os.system("./generate_image_list_for_demo.sh")
    image_list_file = open('./image_list_video.txt')

    if not os.path.exists(image_results_folder):
        os.makedirs(image_results_folder)

    input_images_for_demo = image_list_file.read().splitlines()
    image_list_file.close()

    writer = None
    if createVideoFromResults:
        fps = 30
        codec = 'mp4v'
        videoFileName = 'result.mkv'
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(videoFileName, fourcc, fps, (input_w, input_h))

    # Cache first 100 images for fast access
    prefetchNumFiles = 100
    input_data = []
    if prefetchNumFiles > 0:
        print("Prefetching first %d images" % prefetchNumFiles)
        start = time.time()
        num_prefetched = 0

        for file_path in input_images_for_demo:
            if num_prefetched > prefetchNumFiles:
                break
            frame = cv2.imread(file_path)
            input_data.append(frame)
            print('\r' + "Prefetching files: %d%% " % (100 * num_prefetched / float(prefetchNumFiles)))
            sys.stdout.flush()
            num_prefetched += 1

        print("")
        print("Prefetch completed in {} seconds.".format(round((time.time() - start))))

    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Net(model_description, 1, weights=model_weights)

    result_out_upscaled = np.empty((input_h, input_w, 3), dtype=np.uint8)
    # transparency of the overlaid object segments
    alpha = 0.7
    blended_result = np.empty((input_h, input_w, 3), dtype=np.uint8)

    if show_gui:
        cv2.namedWindow("Demo")

    num_images_processed = 0
    for image in input_images_for_demo:     # main loop

        initial_time = time.time()

        start = time.time()
        if num_images_processed < prefetchNumFiles:
            frame = input_data[num_images_processed]
        else:
            frame = cv2.imread(image)

        print("File read time: {} ms.".format(round((time.time() - start) * 1000)))

        core_forward_time, recognition_result = feed_and_run(frame)

        start = time.time()
        result_out_upscaled = cv2.resize(recognition_result, (input_w, input_h), interpolation=cv2.INTER_NEAREST)
        print("Resize time: {} ms.".format(round((time.time() - start) * 1000)))

        start = time.time()
        cv2.addWeighted(result_out_upscaled, alpha, frame, 1.0 - alpha, 0.0, blended_result)
        print("Overlay detection results: {} ms.".format(round((time.time() - start) * 1000)))

        start = time.time()

	if show_gui:
            cv2.imshow("Demo", blended_result)

	if save_results:
	    cv2.imwrite(results_folder + os.path.basename(image))

        print("cv2 output time: {} ms.".format(round((time.time() - start) * 1000)))

        if createVideoFromResults:
            start = time.time()
            writer.write(blended_result)
            print("Add frame to video file: {} ms.".format(round((time.time() - start) * 1000)))

        key = cv2.waitKey(1)
        if key == 27:  # exit on ESC
            break

        num_images_processed += 1

        print("Total time with data i/o and image pre/post postprocessing - {} ms.".format(
            round((time.time() - initial_time) * 1000)))
        print("---------> Finished processing image #{}, {}, net.forward() time: {} ms.".format(num_images_processed,
                                                                                         os.path.basename(image),
                                                                                         core_forward_time))

    if createVideoFromResults:
        writer.release()

    if show_gui:
        cv2.destroyWindow("Demo")


    profiler.disable()
    print('\n\n\nProfiling results:')
    profiler.print_stats(sort='time')


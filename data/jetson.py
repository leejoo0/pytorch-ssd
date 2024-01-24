#!/usr/bin/env python3
#


import sys
import argparse
import os

from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput, Log

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.",
                                 formatter_class=argparse.RawTextHelpFormatter,
                                 epilog=detectNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("inputDir", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("outputDir", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use")



try:
    args = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)

# create video sources and outputs
# print(f"@@@@@@@@@@@@@@@@{args}")

inputDir = args.inputDir
outputDir = args.outputDir

fileList = os.listdir(inputDir)
print(f"@@@@@@@@@@@@@@@@@@@@{fileList}")


for filename in fileList:
    input = videoSource(os.path.join(inputDir, "*"), argv=sys.argv)
    output = videoOutput(os.path.join(outputDir, filename), argv=sys.argv)
   
# input = videoSource(args.input, argv=sys.argv)
# output = videoOutput(args.output, argv=sys.argv)

# load the object detection network
    net = detectNet(args.network, sys.argv, args.threshold)

   

    # process frames until EOS or the user exits
    while True:
        # capture the next image
        img = input.Capture()

        if img is None:  # timeout
            continue

        # detect objects in the image (with overlay)
        detections = net.Detect(img, overlay=args.overlay)

        # extract file name without extension
        base_filename = os.path.splitext(filename)[0]

        # get the parent folder name from the input path
        input_folder = os.path.basename(inputDir)

        # create an XML file for storing detection results
        xml_filename = f"{base_filename}.xml"
        with open(os.path.join(outputDir, xml_filename), "w") as xml_file:
            xml_file.write("<annotation>\n")
            xml_file.write(f"    <filename>{filename}</filename>\n")
            xml_file.write(f"    <folder>{input_folder}</folder>\n")
            xml_file.write(f"    <source>\n")
            xml_file.write(f"        <database>{input_folder}</database>\n")
            xml_file.write("        <annotation>custom</annotation>\n")
            xml_file.write("        <image>custom</image>\n")
            xml_file.write("    </source>\n")
            xml_file.write("<size>\n")
            xml_file.write(f"    <width>{img.width}</width>\n")
            xml_file.write(f"    <height>{img.height}</height>\n")
            xml_file.write(f"    <depth>{img.channels}</depth>\n")
            xml_file.write("</size>\n")
            xml_file.write("    <segmented>0</segmented>\n")
            for detection in detections:
                xml_file.write("<object>\n")
                xml_file.write(f"    <name>{detection.ClassID}</name>\n")  # assuming ClassID is the object name
                xml_file.write("    <pose>unspecified</pose>\n")
                xml_file.write("    <truncated>0</truncated>\n")
                xml_file.write("    <difficult>0</difficult>\n")
                xml_file.write("<bndbox>\n")
                xml_file.write(f"        <xmin>{detection.Left}</xmin>\n")
                xml_file.write(f"        <ymin>{detection.Top}</ymin>\n")
                xml_file.write(f"        <xmax>{detection.Right}</xmax>\n")
                xml_file.write(f"        <ymax>{detection.Bottom}</ymax>\n")
                xml_file.write("</bndbox>\n")
                xml_file.write("</object>\n")

        # print the detections
        # print(f"Detections saved in {xml_filename}")

        # set file permissions
        os.chmod(os.path.join(outputDir, xml_filename), 0o777)

        # render the image
        output.Render(img)
    

        print(f"!!!!!!!!!!!!!!!!!!!!{detection}")
        # print(f"!!!!!!!!!!!!!!!!!!!!{output}")

        # update the title bar
        # output.SetStatus("{:s} | Network {:.0f} FPS".format(args.network, net.GetNetworkFPS()))

        # print out performance info
        # net.PrintProfilerTimes()

        

        # exit on input/output EOS
        if not input.IsStreaming() or not output.IsStreaming():
            break
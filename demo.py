import argparse
from collections import Counter
import os
import pickle as pkl

import cv2
import torch
from util import load_classes, write_results
from darknet import Darknet
from preprocess import prep_image


ROOT_DIR = os.getcwd()


def draw_object_labels(output_tensor, img, classes):
    """
    Draw bounding box w/ class label for each detected object
    """
    bb_coordinates1 = tuple(output_tensor[1:3].int())
    bb_coordinates2 = tuple(output_tensor[3:5].int())
    class_label = int(output_tensor[-1])

    label = "{0}".format(classes[class_label])
    color = (0, 0, 255)
    cv2.rectangle(img, bb_coordinates1, bb_coordinates2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    bb_coordinates2 = bb_coordinates1[0] + t_size[0] + 3, bb_coordinates1[1] + t_size[1] + 4

    cv2.rectangle(img, bb_coordinates1, bb_coordinates2, color, -1)
    cv2.putText(
        img, label, (bb_coordinates1[0], bb_coordinates1[1] + t_size[1] + 4),
        cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1
    )
    return img


def arg_parse():
    parser = argparse.ArgumentParser(description='YOLO v3 Image Detection')

    parser.add_argument("--image", dest='image', help="Image to run detection upon",
                        default="test_img.jpg", type=str)
    parser.add_argument("--cfg", dest='cfgfile', help="Config file",
                        default="./cfg/yolov3.cfg", type=str)
    parser.add_argument("--reso", dest='reso', help =
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="704", type=str)
    parser.add_argument("--classes", dest="classes", help="List of class names (from data/ folder",
        default="./data/coco-car.names", type=str
    )
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = arg_parse()

    print("Loading network.....")
    model = Darknet(args.cfgfile)
    model.load_weights("yolov3.weights")
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])

    model.eval()

    img = cv2.imread(args.image)
    img, orig_im, dim = prep_image(args.image, inp_dim)
    im_dim = torch.FloatTensor(dim).repeat(1,2)

    with torch.no_grad():
        output = model(torch.autograd.Variable(img), False)
    classes = load_classes(args.classes)
    output = write_results(
        output, confidence=0.5, num_classes=len(classes), nms=True, nms_conf=0.4
    )

    class_counter = Counter([classes[int(obj[-1])] for obj in output])
    print("Class counts: " + str(class_counter))

    tot_objects = output.size(0)
    tot_objs_str = f"Total objects detected: {tot_objects}"
    print(tot_objs_str)

    colors = pkl.load(open("pallete", "rb"))

    im_dim = im_dim.repeat(tot_objects, 1)
    scaling_factor = torch.min(inp_dim/im_dim, 1)[0].view(-1, 1)

    output[:, [1, 3]] -= (inp_dim - scaling_factor*im_dim[:, 0].view(-1, 1))/2
    output[:, [2, 4]] -= (inp_dim - scaling_factor*im_dim[:, 1].view(-1, 1))/2

    output[:, 1:5] /= scaling_factor

    for i in range(output.shape[0]):
        output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
        output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

    list(map(lambda x: draw_object_labels(x, orig_im, classes), output))
    cv2.putText(orig_im, tot_objs_str, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 255, 255], 1,
        cv2.LINE_AA
    )

    name = os.path.join('./output', 'test_output.jpg')
    cv2.imwrite(name, orig_im)

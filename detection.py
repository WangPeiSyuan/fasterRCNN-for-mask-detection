import torch 
import torchvision 
import os
import cv2
import argparse
import numpy as np
import imutils
from datetime import datetime
from PIL import Image
from torchvision import transforms
# from utils import single_img_predict, draw_boxes

def parse_args():
    parser = argparse.ArgumentParser(description='faster RCNN object detection')
    parser.add_argument('-i', '--input', required=True, help='input image or directory or video; webcam input 0')
    parser.add_argument('-t', '--score_thrs', type=float, default=0.8, help='objectness threshold, DEFAULT: 0.8')
    parser.add_argument('-n', '--num_thrs', type=float, default=0.3, help='non max suppression threshold, DEFAULT: 0.3')
    parser.add_argument('-o', '--outdir', default='detection', help='output directory, DEFAULT: detection/')
    parser.add_argument('-v', '--video', action='store_true', default=False, help='flag for detecting a video input')
    parser.add_argument('-w', '--webcam', action='store_true',  default=False, help='flag for detecting from webcam. Specify webcam ID in the input. usually 0 for a single webcam connected')

    args = parser.parse_args()

    return args

def single_img_predict(img, model, nm_thrs, score_thrs):
    test_img = transforms.ToTensor()(img)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        predictions = model(test_img.unsqueeze(0).to(device))
#     print(predictions[0])
    test_img = test_img.permute(1,2,0).numpy()
    
    # non-max supression
    keep_boxes = torchvision.ops.nms(predictions[0]['boxes'].cpu(),predictions[0]['scores'].cpu(),nm_thrs)
    
    # Only display the bounding boxes which higher than the threshold
    score_filter = predictions[0]['scores'].cpu().numpy()[keep_boxes] > score_thrs
    
    # get the filtered result
    test_boxes = predictions[0]['boxes'].cpu().numpy()[keep_boxes][score_filter]
    test_labels = predictions[0]['labels'].cpu().numpy()[keep_boxes][score_filter]
    
    return test_img, test_boxes, test_labels

def draw_boxes(img, boxes,labels, thickness=1):
    """
    Function to draw bounding boxes
    Input:
        img: array of img (h, w ,c)
        boxes: list of boxes (int)
        labels: list of labels (int)
    
    """
    
    for box,label in zip(boxes,labels):
        box = [int(x) for x in box]
        if label == 2:
            color = (0,225,0) # green
        elif label == 1:
            color = (0,0,225) # red
        cv2.rectangle(img, (box[0],box[1]),(box[2],box[3]),color,thickness)
    return img   


def detect_video(model, args):
    if args.webcam:
        cap = cv2.VideoCapture(int(args.input))
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
        output_path = os.path.join(args.outdir, 'det_webcam.avi')
    else:
        cap = cv2.VideoCapture(args.input)
        output_path = os.path.join(args.outdir, 'det_' + os.path.basename(args.input).rsplit('.')[0] + '.avi')
    # width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # fps = cap.get(cv2.CAP_PROP_FPS)
    # print(fps)
    # print(width, "*", height)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20, (240, 180))
    start_time = datetime.now()
    read_frame=0
    while(True):
    # 從攝影機擷取一張影像
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=240)
        read_frame +=1
        if ret:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img, boxes, labels = single_img_predict(img, model, args.num_thrs, args.score_thrs)
            # img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            img = draw_boxes(frame, boxes, labels, thickness=1)
            # 顯示圖片
            cv2.imshow('frame', img)
            if(read_frame%30==0):
                print('Number of frames processed:', read_frame)
            out.write(frame)
        # 若按下 q 鍵則離開迴圈
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    
    # 釋放攝影機
    end_time = datetime.now()
    print('Detection finished in %s secs' % (end_time - start_time).seconds)
    print('Total frames:', read_frame)
    cap.release()
    out.release()
    # 關閉所有 OpenCV 視窗
    cv2.destroyAllWindows()
    print('Detected video saved to ' + output_path)

def detect_image(model, args):

    print('Loading input image(s)...')
#     print(args.input)
#     print(os.path.isdir(args.input))
    if(os.path.isdir(args.input)): 
        print('processing...')
        start_time = datetime.now()
        for file in os.listdir(args.input):
            img = Image.open(os.path.join(args.input,file)).convert('RGB')
            test_img, boxes, labels = single_img_predict(img, model, args.num_thrs, args.score_thrs)
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            result = draw_boxes(img, boxes, labels, thickness=3)
            output_path = os.path.join(args.outdir, 'det_' + file)
            cv2.imwrite(output_path, result)
        end_time = datetime.now()
    else:
        print('processing...')
        start_time = datetime.now()
        img = Image.open(args.input).convert('RGB')
        test_img, boxes, labels = single_img_predict(img, model, args.num_thrs, args.score_thrs)
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        result = draw_boxes(img, boxes, labels, thickness=2)
        output_path = os.path.join(args.outdir, 'det_' + os.path.basename(args.input))
        print(output_path)
        cv2.imwrite(output_path, result)
        end_time = datetime.now()
        
    print('Detection finished in %s secs' % (end_time - start_time).seconds)

    return
   


def main():

    args = parse_args()
    if not os.path.exists(args.outdir): #create output directory
        os.makedirs(args.outdir)
    print('Loading network...')
    if torch.cuda.is_available():
        model = torch.load('model/20210607fasterRCNN.pth')
    else:
        model = torch.load('model/20210607fasterRCNN.pth', map_location='cpu')
    model.eval()
    print('Network loaded')
    if args.video or args.webcam:
        detect_video(model, args)
    else:
        detect_image(model, args)

if __name__ == '__main__':
    main()


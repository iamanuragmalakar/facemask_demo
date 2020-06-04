import cv2
import numpy as np

# def load_yolo():
# 	net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
# 	classes = []
# 	with open("coco.names", "r") as f:
# 		classes = [line.strip() for line in f.readlines()]
# 	layers_names = net.getLayerNames()
# 	output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
# 	colors = np.random.uniform(0, 255, size=(len(classes), 3))
# 	return net, classes, colors, output_layers

# def load_image(img_path):
# 	# image loading
# 	img = cv2.imread(img_path)
# 	img = cv2.resize(img, None, fx=0.4, fy=0.4)
# 	height, width, channels = img.shape
# 	return img, height, width, channels

def detect_objects(img, net, outputLayers):			
	blob = cv2.dnn.blobFromImage(img, scalefactor=1/255.0, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
	net.setInput(blob)
	outputs = net.forward(outputLayers)
	return blob, outputs

# def get_box_dimensions(img, outputs, confidence_threshold):
# 	boxes = []
# 	confs = []
# 	class_ids = []
#     H, W = img.shape[:2]
#     for output in outputs:
# 		for detect in output:
# 			scores = detect[5:]
# 			print(scores)
# 			class_id = np.argmax(scores)
# 			conf = scores[class_id]
# 			if conf > confidence_threshold:
#                 box = detect[0:4] * np.array([W, H, W, H])
#                 centerX, centerY, width, height = box.astype("int")
# 				x, y = int(centerX - (width / 2)), int(centerY - (height / 2))
#                 boxes.append([x, y, int(width), int(height)])
#                 confs.append(float(conf))
#                 class_ids.append(class_id)
#                 #center_x = int(detect[0] * width)
# 				#center_y = int(detect[1] * height)
# 				# w = int(detect[2] * width)
# 				# h = int(detect[3] * height)
# 				# x = int(center_x - w/2)
# 				# y = int(center_y - h / 2)
# 				# boxes.append([x, y, w, h])
# 				# confs.append(float(conf))
# 				# class_ids.append(class_id)
# 	return boxes, confs, class_ids


# def draw_labels(boxes, confs, colors, class_ids, classes, img, confidence_threshold, overlap_thershold): 
# 	indexes = cv2.dnn.NMSBoxes(boxes, confs, confidence_threshold, overlap_thershold)
#     #xmin, xmax, ymin, ymax, labels = [], [], [], [], []
# 	font = cv2.FONT_HERSHEY_PLAIN
# 	for i in range(len(boxes)):
# 		if i in indexes:
# 			x, y, w, h = boxes[i]
# 			label = str(classes[class_ids[i]])
# 			color = colors[i]
# 			cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
# 			cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
# 			#cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
# 			#cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
# 	#cv2.imshow("Image", img)
#     #boxes = pd.DataFrame({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})
#     #image_with_boxes = image.astype(np.float64)
#     #for _, (xmin, ymin, xmax, ymax) in boxes.iterrows():
# 	#	image_with_boxes[int(ymin):int(ymax),int(xmin):int(xmax),:] += LABEL_COLORS
# 	#	image_with_boxes[int(ymin):int(ymax),int(xmin):int(xmax),:] /= 2
#     st.image(cv2.imshow("Image", img), use_column_width=True)

def get_box_dimensions(img, outputs, height, width, confidence_threshold):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            print(scores)
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > confidence_threshold:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids


def draw_labels(boxes, confs, colors, class_ids, classes, img, confidence_threshold, overlap_thershold): 
	indexes = cv2.dnn.NMSBoxes(boxes, confs, confidence_threshold, overlap_thershold)
    #xmin, xmax, ymin, ymax, labels = [], [], [], [], []
	font = cv2.FONT_HERSHEY_PLAIN
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			label = str(classes[class_ids[i]])
			color = colors[i]
			cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
			cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
	st.image( img, use_column_width=True)
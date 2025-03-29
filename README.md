# Object Detection using MobileNet SSD

This project implements real-time object detection using the MobileNet SSD model with OpenCV. It uses a pre-trained model to detect objects from a live webcam feed.

## Features
- Real-time object detection using webcam
- Detects multiple objects simultaneously
- Displays bounding boxes with labels and confidence scores
- Simple and interactive interface using OpenCV
- Supports a wide range of object classes

## Technologies Used
- Python
- OpenCV
- MobileNet SSD (Single Shot Multibox Detector)
- Caffe Framework

## Installation
1. Clone the repository:
```
git clone https://github.com/username/object-detection-mobilenet.git
```

2. Install required libraries:
```
pip install opencv-python numpy
```

3. Download the pre-trained model files:
- [deploy.prototxt](https://github.com/chuanqi305/MobileNet-SSD/blob/master/deploy.prototxt)
- [mobilenet_iter_73000.caffemodel](https://github.com/chuanqi305/MobileNet-SSD/blob/master/mobilenet_iter_73000.caffemodel)

4. Place the model files in the project directory.

## Running the Project
Run the following command to start object detection:
```
python object_detection.py
```

Press `q` to exit the application.

## How It Works
1. Loads the MobileNet SSD model using Caffe.
2. Captures real-time video from the webcam.
3. Preprocesses each frame to create a blob.
4. Passes the blob through the network for object detection.
5. Displays bounding boxes and class labels on detected objects.

## Customization
- Modify the confidence threshold in the code to adjust detection sensitivity.
- Update the class list to include additional objects if needed.

## Troubleshooting
- Ensure that your webcam is properly connected.
- Verify the paths to the prototxt and caffemodel files.
- Update OpenCV to the latest version if you encounter compatibility issues.

## License
This project is licensed under the MIT License.

## Contributing
Feel free to open issues or submit pull requests to enhance the project!


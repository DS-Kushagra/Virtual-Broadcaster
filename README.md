# Virtual Broadcaster

## Overview
Virtual Broadcaster is a real-time video streaming application that enables users to apply background effects, such as blurring and image replacement, during live video streaming. The project is built using FastAPI for backend services and OpenCV along with YOLOv8 for background segmentation.

## Features
- Real-time video streaming with background manipulation.
- Supports blurring, black background, and custom background images.
- Utilizes YOLOv8 for efficient segmentation.
- Works with virtual cameras for seamless integration with other applications.
- FastAPI-based backend for handling video stream configuration.

## Technologies Used
- **Python** (FastAPI, OpenCV, NumPy, PyTorch)
- **Computer Vision** (YOLOv8 for segmentation)
- **Web Technologies** (HTML, JavaScript for UI)
- **Virtual Camera Integration** (pyvirtualcam)

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- pip (Python package manager)
- A working webcam

### Steps to Install
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/virtual-broadcaster.git
   cd virtual-broadcaster
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the YOLOv8 segmentation model and place it in the project directory.
4. Run the server:
   ```bash
   python main.py
   ```
5. Access the UI at `http://localhost:8000`.

## Usage
1. Open the web UI to configure streaming settings.
2. Choose the desired background effect (blur, none, default).
3. Start the stream, and it will be processed and displayed through a virtual camera.

## API Endpoints
| Endpoint  | Method | Description |
|-----------|--------|-------------|
| `/` | GET | Serves the UI |
| `/start` | GET | Starts the video stream with specified parameters |
| `/stop` | GET | Stops the ongoing video stream |
| `/devices` | GET | Lists available camera devices |

## Project Structure
```
virtual-broadcaster/
│── static/                # Static files (HTML, images)
│── main.py                # FastAPI application
│── stream_utils.py        # Streaming utilities and background processing
│── utils.py               # Helper functions
│── engine.py              # YOLO segmentation logic
│── requirements.txt       # Dependencies
│── README.md              # Project documentation
```

## Future Enhancements
- Support for additional background effects.
- Enhanced segmentation accuracy.
- WebRTC integration for direct browser-based streaming.

## Contributors
- **Kushagra** - Developer & Maintainer

## License
This project is licensed under the MIT License - see the LICENSE file for details.


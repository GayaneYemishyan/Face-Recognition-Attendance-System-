# Face Recognition Attendance System

A Face Recognition Attendance System using computer vision and machine learning to automate and streamline the process of taking attendance. This project leverages facial recognition algorithms to identify individuals in real-time and record their attendance accurately.

## Features

- Real-time face detection and recognition using a webcam or image files
- Face outlining on detected faces using NumPy and OpenCV
- Attendance logging with timestamps (CSV export)
- Register new users by capturing their face images

## Technologies Used

- Python
- OpenCV
- face_recognition (dlib)
- numpy
- csv (Python standard library)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/GayaneYemishyan/Face-Recognition-Attendance-System-.git
   cd Face-Recognition-Attendance-System-
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python main.py
   ```

## Usage

- Register new users by capturing their face images.
- Start the attendance process; the system will recognize faces and automatically mark attendance.
- Attendance records are saved in CSV format for further processing.

## Repository Structure

```
├── main.py         # Main application file
├── README.md       # Project description and instructions
```

## License

This project is licensed under the MIT License.

---

*For any questions or suggestions, please open an issue or contact the repository owner.*

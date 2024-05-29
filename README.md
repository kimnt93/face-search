# Face Similarity Search

This application helps charity groups identify individuals who have received aid two or more times by performing face similarity searches. The app is designed to be user-friendly and efficient, with features that make it suitable for both standalone use and collaborative environments.

## Features

- **Standalone Operation**: The application can run independently on a single machine.
- **Network Collaboration**: Multiple instances of the application can join together within the same network to share workloads.
- **Milvus Server Integration**: The application can ping and interact with a Milvus server for managing and querying face embeddings.

## Technology Stack

- **Milvus**: A high-performance vector database for similarity search.
- **Redis**: An in-memory data structure store for caching and message brokering.
- **OpenCV**: A library for computer vision tasks, used for face detection and preprocessing.
- **Flask**: A lightweight web application framework for building the app's RESTful API.
- **face_recognition**: A library for face detection and recognition.
- **InsightFace**: An alternative library for advanced face detection and recognition.

## Installation

1. **Install Python Dependencies**:

   Run the following command to install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Database Server**:

   Use Docker Compose to start the Milvus and Redis database servers:
   ```bash
   docker-compose up
   ```

## Running the Application

After installing the dependencies and starting the database server, run the application with:
```bash
python app.py
```

This will start the Flask server, allowing you to interact with the face similarity search functionality through the provided API endpoints.

## Usage

1. **Face Enrollment**:
   - Use the API to enroll faces into the database by sending images. The application will process these images using face_recognition or InsightFace to detect faces and store the resulting embeddings in Milvus.

2. **Face Detection and Embedding**:
   - The application automatically detects faces in the provided images using [face_recognition](https://github.com/ageitgey/face_recognition) or [InsightFace](https://github.com/deepinsight/insightface). This feature ensures that only the face region is used for embedding and similarity search, improving accuracy.

3. **Face Search**:
   - Query the API to search for similar faces in the Milvus database. The application will compare the detected face against stored embeddings and return similarity matches that indicate individuals who have received aid more than once.

4. **Network Collaboration**:
   - Configure multiple instances to join the same network. This enables the application to distribute the workload and share data between different nodes, enhancing scalability and reliability.

These tutorials will guide you through integrating and utilizing the Groq API effectively for various applications.

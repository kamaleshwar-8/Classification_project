# Indian Cricket Players Image Classifier

This project is an image classification web application that can identify Indian cricket players. It's part of my learning journey in machine learning and computer vision, inspired by [codebasics](https://www.youtube.com/@codebasics).

## Players Included
- MS Dhoni
- Virat Kohli
- R Ashwin
- Suresh Raina
- Rohit Sharma

## Technologies Used
- Python (Flask backend)
- OpenCV for face and eye detection
- Wavelet transforms for feature extraction
- Machine Learning models (SVM, Random Forest, Logistic Regression)
- HTML/CSS for frontend
- JavaScript for client-side interactions

## Setup and Running

1. Make sure you have Python installed and the following dependencies:
   ```bash
   pip install flask opencv-python numpy pandas pywt scikit-learn joblib
   ```

2. Start the Flask server:
   ```bash
   python server.py
   ```

3. Open `index.html` in a web browser

4. Upload an image of any of the above cricket players to classify them

## How it Works
1. The application uses Haar Cascade classifiers to detect faces and eyes
2. Images are preprocessed using wavelet transforms
3. The preprocessed images are fed into a trained machine learning model
4. The model returns probability scores for each player
5. Results are displayed on the web interface

## Note
- The classifier works best with clear, front-facing images
- The model requires both face and eyes to be visible for accurate classification
- If the face or eyes are not detected properly, an error message will be displayed

## Output
![one - Copy](https://github.com/user-attachments/assets/ec369c97-14bc-4194-bfab-1d7aaf177e67)

![three - Copy](https://github.com/user-attachments/assets/5dcac70d-e564-4528-a4a3-b5a7e202e064)

![four - Copy](https://github.com/user-attachments/assets/b87cdb91-4e96-46e8-a37b-599f0fbf4cd0)


## Credits
This project was created as part of my learning process, following tutorials from [codebasics](https://www.youtube.com/@codebasics)

# 🎯 L2CS-Net Gaze Tracking - Inference

 L2CS-Net (Log Softmax Classification and Soft Regression Network) is a CNN-based deep learning model designed for real-time gaze estimation in the wild. 

 It improves gaze direction prediction by separating pitch and yaw angles and using a combination of classification and regression techniques.

---

## For more info visit the [Official Repository](https://github.com/Ahmednull/L2CS-Net)


## 📥 Installation

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/your-username/GazeTrackerUsingL2CS-Net.git
cd GazeTrackerUsingL2CS-Net/Inference
```


### **2️⃣ Create a Virtual Environment (Optional)**
```bash
python3 -m venv .env
source .env/bin/activate    # For Linux
.env\Scripts\activate.bat   # For Windows
```

### **3️⃣ Install the Required Libraries**
```bash
pip install -r requirements.txt
```

## 🎯 Usage

Run the Inference Script
```bash
python main.py --video_path <path-to-video> --model_path <path-to-model> 
```

## 🙌 Acknowledgements

- **Authors**:
  - [Ahmednull](https://github.com/Ahmednull)
  - [Paper Link](https://arxiv.org/pdf/2203.03339)

- **Dataset**: [MPIIGaze](https://paperswithcode.com/dataset/mpiigaze) , [Gaze360](https://gaze360.csail.mit.edu/)

- **face-detection**: [face-detection](https://github.com/elliottzheng/face-detection)

- **PyTorch**: [PyTorch](https://pytorch.org/)

- **OpenCV**: [OpenCV](https://opencv.org/)

- **Official Repository**: [GazeTrackerUsingL2CS-Net](https://github.com/Ahmednull/L2CS-Net/tree/main)


## ⭐ Contribute

- If you find this useful, please ⭐ star this repo and feel free to open issues or PRs!

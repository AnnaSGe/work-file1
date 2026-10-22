
# ============================================================
# BCSE425P ROBOTIC PERCEPTION LAB
# EXPERIMENTS 6 TO 10
# ============================================================


# ============================================================
# EXPERIMENT 6: HOUGH TRANSFORM LINE DETECTION
# ============================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("img.jpg")
if img is None:
    raise FileNotFoundError("Check img.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 100, 180)

lines = cv2.HoughLinesP(
    edges, 1, np.pi / 180,
    threshold=20,
    minLineLength=25,
    maxLineGap=10
)

result = img.copy()

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(edges, cmap="gray")
plt.title("Canny Edges")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("Hough Lines")
plt.axis("off")
plt.show()


# ============================================================
# EXPERIMENT 6: HARRIS CORNER DETECTION
# ============================================================

img = cv2.imread("img.jpg")
if img is None:
    raise FileNotFoundError("Check img.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray32 = np.float32(gray)

dst = cv2.cornerHarris(gray32, 2, 3, 0.04)
dst = cv2.dilate(dst, None)

result = img.copy()
result[dst > 0.05 * dst.max()] = [0, 0, 255]

plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("Harris Corners")
plt.axis("off")
plt.show()


# ============================================================
# EXPERIMENT 7: SIFT FEATURE DETECTION
# ============================================================

img = cv2.imread("img.jpg")
if img is None:
    raise FileNotFoundError("Check img.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)

print("Number of keypoints:", len(keypoints))

result = cv2.drawKeypoints(
    img, keypoints, None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("SIFT Features")
plt.axis("off")
plt.show()


# ============================================================
# EXPERIMENT 7: SIFT OBJECT MATCHING
# ============================================================

img1 = cv2.imread("img1.jpg", 0)
img2 = cv2.imread("img2.jpg", 0)

if img1 is None or img2 is None:
    raise FileNotFoundError("Check img1.jpg and img2.jpg")

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

if des1 is None or des2 is None:
    print("Not enough features to match.")
else:
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

    print("Good matches:", len(good))

    result = cv2.drawMatches(
        img1, kp1, img2, kp2, good, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    plt.figure(figsize=(12, 5))
    plt.imshow(result, cmap="gray")
    plt.title("SIFT Feature Matching")
    plt.axis("off")
    plt.show()


# ============================================================
# EXPERIMENT 7: FASTER R-CNN USING OPENCV DNN
# Requires compatible TensorFlow model files and coco.names
# ============================================================

img = cv2.imread("img.jpg")
if img is None:
    raise FileNotFoundError("Check img.jpg")

h, w = img.shape[:2]

net = cv2.dnn.readNetFromTensorflow(
    "frozen_inference_graph.pb",
    "config.pbtxt"
)

blob = cv2.dnn.blobFromImage(
    img, size=(300, 300), swapRB=True, crop=False
)

net.setInput(blob)
detections = net.forward()

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f]

result = img.copy()
threshold = 0.5

for i in range(detections.shape[2]):
    confidence = float(detections[0, 0, i, 2])

    if confidence > threshold:
        class_id = int(detections[0, 0, i, 1])
        box = detections[0, 0, i, 3:7] * [w, h, w, h]
        x1, y1, x2, y2 = box.astype(int)

        label = (
            classes[class_id]
            if 0 <= class_id < len(classes)
            else str(class_id)
        )

        cv2.rectangle(
            result, (x1, y1), (x2, y2), (0, 255, 0), 2
        )

        cv2.putText(
            result, f"{label} {confidence:.2f}",
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("Faster R-CNN Detection")
plt.axis("off")
plt.show()


# ============================================================
# EXPERIMENT 8: YOLO OBJECT DETECTION
# ============================================================

from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model("img.jpg", conf=0.5)

result = results[0].plot()

plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("YOLO Detection")
plt.axis("off")
plt.show()

for box in results[0].boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    print(model.names[class_id], round(confidence, 3))


# ============================================================
# EXPERIMENT 8: COMPARE YOLO AND FASTER R-CNN
# Requires net from the Faster R-CNN section
# ============================================================

import time

img = cv2.imread("img.jpg")
if img is None:
    raise FileNotFoundError("Check img.jpg")

yolo = YOLO("yolov8n.pt")

start = time.perf_counter()
yolo_results = yolo(img, verbose=False, conf=0.5)
yolo_time = time.perf_counter() - start

blob = cv2.dnn.blobFromImage(
    img, size=(300, 300), swapRB=True, crop=False
)

net.setInput(blob)

start = time.perf_counter()
faster_results = net.forward()
faster_time = time.perf_counter() - start

print("YOLO time:", round(yolo_time, 4), "seconds")
print("Faster R-CNN time:", round(faster_time, 4), "seconds")

plt.bar(
    ["YOLO", "Faster R-CNN"],
    [yolo_time, faster_time]
)
plt.ylabel("Time (seconds)")
plt.title("Inference Time Comparison")
plt.show()


# ============================================================
# EXPERIMENT 8: REAL-TIME YOLO WEBCAM
# ============================================================

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

while True:
    start = time.perf_counter()
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, verbose=False, conf=0.5)
    output = results[0].plot()

    fps = 1 / max(time.perf_counter() - start, 1e-9)

    cv2.putText(
        output, f"FPS: {fps:.1f}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )

    cv2.imshow("YOLO Webcam", output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


# ============================================================
# EXPERIMENT 9: BUNDLE ADJUSTMENT AND 3D RECONSTRUCTION
# ============================================================

from scipy.optimize import least_squares

img1 = cv2.imread("img1.jpg")
img2 = cv2.imread("img2.jpg")

if img1 is None or img2 is None:
    raise FileNotFoundError("Check img1.jpg and img2.jpg")

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

if des1 is None or des2 is None:
    raise ValueError("Not enough SIFT features")

pairs = cv2.BFMatcher().knnMatch(des1, des2, k=2)

good = []
for pair in pairs:
    if len(pair) == 2:
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)

if len(good) < 8:
    raise ValueError("Need at least 8 good matches")

p1 = np.float32([kp1[m.queryIdx].pt for m in good])
p2 = np.float32([kp2[m.trainIdx].pt for m in good])

h, w = gray1.shape
f = float(max(h, w))

K = np.array([
    [f, 0, w / 2],
    [0, f, h / 2],
    [0, 0, 1]
], dtype=float)

E, mask = cv2.findEssentialMat(
    p1, p2, K,
    method=cv2.RANSAC,
    prob=0.999,
    threshold=1.0
)

if E is None or mask is None:
    raise ValueError("Essential matrix estimation failed")

inliers = mask.ravel().astype(bool)
p1, p2 = p1[inliers], p2[inliers]

_, R, t, pose_mask = cv2.recoverPose(E, p1, p2, K)

valid = pose_mask.ravel() > 0
p1, p2 = p1[valid], p2[valid]

P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
P2 = K @ np.hstack((R, t))

X4 = cv2.triangulatePoints(P1, P2, p1.T, p2.T)
X = (X4[:3] / X4[3]).T

rvec2, _ = cv2.Rodrigues(R)
tvec2 = t.reshape(3, 1)

def project_points(points, rvec, tvec):
    projected, _ = cv2.projectPoints(
        points, rvec, tvec, K, None
    )
    return projected.reshape(-1, 2)

def residual(params):
    rvec = params[:3].reshape(3, 1)
    tvec = params[3:6].reshape(3, 1)
    points = params[6:].reshape(-1, 3)

    pred1 = project_points(
        points, np.zeros((3, 1)), np.zeros((3, 1))
    )
    pred2 = project_points(points, rvec, tvec)

    return np.concatenate([
        (pred1 - p1).ravel(),
        (pred2 - p2).ravel()
    ])

initial = np.concatenate([
    rvec2.ravel(), tvec2.ravel(), X.ravel()
])

error_before = np.sqrt(np.mean(residual(initial) ** 2))

opt = least_squares(residual, initial, max_nfev=50)
refined = opt.x[6:].reshape(-1, 3)

error_after = np.sqrt(np.mean(opt.fun ** 2))

print("Error before:", error_before)
print("Error after:", error_after)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(refined[:, 0], refined[:, 1], refined[:, 2], s=5)
ax.set_title("3D Reconstruction")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()


# ============================================================
# EXPERIMENT 10: KALMAN FILTER
# ============================================================

np.random.seed(42)

dt = 1.0
N = 50

true_velocity = 2.0
true_pos = np.arange(N) * true_velocity
measurements = true_pos + np.random.normal(0, 5, N)

# State = position and velocity
A = np.array([
    [1, dt],
    [0, 1]
])

H = np.array([[1, 0]])

Q = np.array([
    [0.01, 0],
    [0, 0.01]
])

R = np.array([[25.0]])

x = np.array([[0.0], [0.0]])
P = np.eye(2) * 100

estimates = []
velocity_estimates = []

for z in measurements:

    # Prediction
    x = A @ x
    P = A @ P @ A.T + Q

    # Measurement update
    z = np.array([[z]])

    y = z - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)

    x = x + K @ y
    P = (np.eye(2) - K @ H) @ P

    estimates.append(x[0, 0])
    velocity_estimates.append(x[1, 0])

estimates = np.array(estimates)
velocity_estimates = np.array(velocity_estimates)

rmse_measurements = np.sqrt(
    np.mean((measurements - true_pos) ** 2)
)
rmse_filtered = np.sqrt(
    np.mean((estimates - true_pos) ** 2)
)

print("Measurement RMSE:", round(rmse_measurements, 3))
print("Filtered RMSE:", round(rmse_filtered, 3))

plt.plot(true_pos, label="True Position")
plt.plot(measurements, ".", label="Noisy Measurements")
plt.plot(estimates, label="Kalman Estimate")
plt.xlabel("Time Step")
plt.ylabel("Position")
plt.legend()
plt.show()

plt.plot(velocity_estimates, label="Estimated Velocity")
plt.axhline(true_velocity, linestyle="--", label="True Velocity")
plt.xlabel("Time Step")
plt.ylabel("Velocity")
plt.legend()
plt.show()


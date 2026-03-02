# =====================================================
# Coconut Quality Detection - Final Working Version
# =====================================================

from flask import Flask, request, render_template_string
from ultralytics import YOLO
import os
import cv2

app = Flask(__name__)

model = YOLO("best.pt")

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Coconut Quality Detection</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body style="background-color:#f4f6f9;">

<div class="container mt-5">
    <div class="card shadow-lg p-4">
        <h2 class="text-center mb-4">🥥 Coconut Quality Detection (YOLOv8)</h2>

        <form method="POST" enctype="multipart/form-data" class="text-center">
            <input class="form-control mb-3" type="file" name="file" required>
            <button class="btn btn-primary" type="submit">Upload & Detect</button>
        </form>

        {% if image_path %}
        <hr>
        <div class="text-center">
            <h4>Prediction Result</h4>
            <img src="{{ image_path }}" class="img-fluid rounded shadow" width="500"><br><br>

            <h3>
                Quality:
                <span class="badge bg-{{ color }}">
                    {{ quality }}
                </span>
            </h3>

            <p>Confidence: {{ confidence }}%</p>
        </div>
        {% endif %}
    </div>
</div>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    image_path = None
    quality = None
    confidence = None
    color = "secondary"

    if request.method == "POST":
        file = request.files["file"]

        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Run prediction
            results = model(filepath)
            result = results[0]

            if len(result.boxes) > 0:
                box = result.boxes[0]
                cls_id = int(box.cls[0])
                quality = model.names[cls_id]
                confidence = round(float(box.conf[0]) * 100, 2)

                if quality == "green":
                    color = "success"
                elif quality == "dry":
                    color = "warning"
                elif quality == "tender":
                    color = "info"
                else:
                    color = "secondary"
            else:
                quality = "No coconut detected"
                confidence = 0
                color = "danger"

            # Draw prediction image
            plotted_img = result.plot()
            output_path = os.path.join(OUTPUT_FOLDER, file.filename)
            cv2.imwrite(output_path, plotted_img)

            image_path = f"/static/{file.filename}"

    return render_template_string(
        HTML_PAGE,
        image_path=image_path,
        quality=quality,
        confidence=confidence,
        color=color
    )

if __name__ == "__main__":

    app.run(debug=True)

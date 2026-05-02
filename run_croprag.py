import os
import cv2
from PIL import Image

from rag.pipeline import RAGPipeline
from rag.image_predictor import ImagePredictor
from llm.mistral_client import MistralClientWrapper


# ---------------- CAMERA ----------------
def capture_image():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Camera not accessible")
        return None

    print("📸 Press SPACE to capture | ESC to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("CropRAG Camera", frame)
        key = cv2.waitKey(1)

        if key == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            return None

        if key == 32:  # SPACE
            cap.release()
            cv2.destroyAllWindows()
            return Image.fromarray(frame)


# ---------------- HELPERS ----------------
def extract_crop_disease_from_path(path):
    """
    Extract crop & disease from folder name like:
    Tomato___Leaf_Mold  ->  tomato, leaf_mold
    """
    folder = os.path.basename(os.path.normpath(path))

    if "___" in folder:
        crop, disease = folder.split("___", 1)
        crop = crop.replace(",", "").lower()
        disease = disease.lower().replace(" ", "_")
        return crop, disease

    return None, None


def severity_from_confidence(conf):
    if conf >= 75:
        return "HIGH"
    elif conf >= 40:
        return "MEDIUM"
    else:
        return "LOW"


# ---------------- MAIN ----------------
def main():
    rag = RAGPipeline()
    predictor = ImagePredictor()
    llm = MistralClientWrapper()

    print("\n🌿 CropRAG – CLI Mode")
    print("=" * 50)
    print("1. Text query")
    print("2. Image query (Camera)")
    print("3. Image query (Image file / Folder)")

    choice = input("Choose option (1/2/3): ").strip()

    confidence = None

    # -------- TEXT QUERY --------
    if choice == "1":
        crop = input("Enter crop name: ").strip().lower()
        disease = input("Enter disease name: ").strip().lower()

    # -------- CAMERA IMAGE --------
    elif choice == "2":
        image = capture_image()
        if image is None:
            return

        disease, confidence = predictor.predict(image)
        crop = input("Detected disease. Enter crop name: ").strip().lower()

    # -------- IMAGE FILE / FOLDER --------
    elif choice == "3":
        image_path = input("Enter image file path: ").strip().strip('"')

        if not os.path.exists(image_path):
            print("❌ Image file or folder not found")
            return

        # ✅ YOUR REQUESTED CODE (ADDED)
        if os.path.isdir(image_path):
            files = [
                f for f in os.listdir(image_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

            if not files:
                print("❌ No image files found in folder")
                return

            image_path = os.path.join(image_path, files[0])
            print(f"📂 Using image: {files[0]}")

        image = Image.open(image_path).convert("RGB")

        disease, confidence = predictor.predict(image)

        crop, inferred_disease = extract_crop_disease_from_path(os.path.dirname(image_path))
        if inferred_disease:
            disease = inferred_disease

        if not crop:
            crop = input("Enter crop name: ").strip().lower()

    else:
        print("❌ Invalid choice")
        return

    # -------- RAG QUERY --------
    docs = rag.query_text(crop, disease)

    answer = llm.generate(crop, disease, docs)

    severity = severity_from_confidence(confidence) if confidence is not None else None

    # -------- OUTPUT --------
    print("\n📄 DISEASE ANALYSIS REPORT")
    print("=" * 60)
    print(f"🌱 Crop              : {crop.capitalize()}")
    print(f"🦠 Disease           : {disease.replace('_', ' ').title()}")

    if confidence is not None:
        print(f"📊 Prediction Confidence : {confidence:.2f}%")
        print(f"⚠️ Severity Level        : {severity}")

    print("-" * 60)
    print(answer)
    print("=" * 60)


if __name__ == "__main__":
    main()

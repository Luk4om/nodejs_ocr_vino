# OCR Text Detection with OpenVINO (Node.js)

โปรเจกต์นี้เป็นตัวอย่างการใช้งาน **OpenVINO** ร่วมกับ **Node.js** เพื่อตรวจจับข้อความ (Text Detection) บนรูปภาพ โดยใช้โมเดล **PP-OCRv3** ที่ทำการแปลงเป็นฟอร์แมต OpenVINO IR แล้ว

## ฟีเจอร์หลัก
- **Text Detection:** ตรวจจับตำแหน่งของข้อความในภาพ
- **OpenVINO Inference:** ใช้ `openvino-node` ในการประมวลผลโมเดล ซึ่งให้ประสิทธิภาพรวดเร็วบน CPU
- **Image Processing:** ใช้ `sharp` ในการจัดการรูปภาพ (Resize, Pre-processing, Overlay)
- **Visualization:** สามารถสร้างภาพ Heatmap แสดงตำแหน่งที่ตรวจพบข้อความได้

---

## โครงสร้างโปรเจกต์ (Project Structure)

- `ocr_det.js`: สคริปต์หลักสำหรับรัน Detection และแสดงผลลัพธ์เป็นตัวเลข (Log output)
- `plot_result.js`: สคริปต์สำหรับรัน Detection และวาด Heatmap สีแดงทับลงบนภาพต้นฉบับ เพื่อแสดงตำแหน่งข้อความ บันทึกเป็นไฟล์ `result_overlay.png`
- `modelvino/`: โฟลเดอร์เก็บไฟล์โมเดล OpenVINO (`model.xml`, `model.bin`)
- `basemodel/`: โฟลเดอร์เก็บโมเดลต้นฉบับ (PaddlePaddle) และค่าพารามิเตอร์ต่างๆ
- `image/`: โฟลเดอร์เก็บรูปภาพสำหรับทดสอบ (เช่น `image_test.png`)

---

## การติดตั้ง (Installation)

1. **Clone Repository**
   ```bash
   git clone <repository-url>
   cd ocr_vino
   ```

2. **ติดตั้ง Git LFS (Large File Storage)**
   โปรเจกต์นี้เก็บไฟล์โมเดลขนาดใหญ่ด้วย Git LFS จำเป็นต้องดึงไฟล์จริงลงมาก่อน
   ```bash
   git lfs install
   git lfs pull
   ```

3. **ติดตั้ง Dependencies**
   จำเป็นต้องมี [Node.js](https://nodejs.org/) ติดตั้งอยู่ในเครื่องก่อน
   ```bash
   npm install
   ```
   คำสั่งนี้จะติดตั้งไลบรารีที่จำเป็นคือ `openvino-node` และ `sharp`

---

## วิธีการใช้งาน (Usage)

### 1. รันการตรวจจับเบื้องต้น (Basic Detection)
ใช้สำหรับตรวจสอบว่าโมเดลทำงานได้หรือไม่ และดูค่าความมั่นใจ (Confidence) ผ่าน Terminal
```bash
node ocr_det.js
```
**ผลลัพธ์:** จะแสดง Log การทำงาน และจำนวนพิกเซลที่คาดว่าเป็นตัวอักษร

### 2. รันและสร้างภาพผลลัพธ์ (Visual Result)
ใช้สำหรับดูว่าโมเดลจับข้อความตรงไหนได้บ้าง โดยจะวางเลเยอร์สีแดงทับลงบนภาพ
```bash
node plot_result.js
```
**ผลลัพธ์:** สร้างไฟล์ใหม่ชื่อ `result_overlay.png` ในโฟลเดอร์เดียวกัน ซึ่งจะแสดงพื้นที่สีแดงตรงจุดที่เป็นข้อความ

---

## การปรับแต่ง (Configuration)

คุณสามารถแก้ไขตัวแปรที่ส่วนหัวของไฟล์ `.js` เพื่อปรับเปลี่ยนการทำงานได้:

- **เปลี่ยนรูปภาพ:** แก้ไขตัวแปร `imagePath` หรือ `IMAGE_PATH` ให้ชี้ไปที่รูปของคุณ
  ```javascript
  const imagePath = 'image/your_image.jpg';
  ```
- **ปรับ Threshold:** แก้ไขตัวแปร `threshold` หรือ `THRESHOLD` (ค่าระหว่าง 0.0 - 1.0)
  - ค่าสูง: ตรวจจับเฉพาะจุดที่มั่นใจมากๆ (Noise น้อย แต่อาจตกหล่น)
  - ค่าต่ำ: ตรวจจับง่ายขึ้น (แต่อาจมี Noise หรือจุดผิดพลาดปนมา)

## หมายเหตุ
- โมเดลนี้รันบน **CPU** เป็นค่าเริ่มต้น หากต้องการใช้ GPU (Intel) สามารถแก้ไขโค้ดตรง `core.compileModel(model, 'GPU')`
- ไฟล์โมเดลควรเป็นเวอร์ชันที่เข้ากันได้กับ `openvino-node`

---

## การแปลงโมเดล (Model Conversion Guide)

โปรเจกต์นี้รองรับโมเดลจากหลายแหล่ง โดยมีวิธีการแปลงเป็น OpenVINO ดังนี้:

### 1. PaddleOCR Detection (e.g., `en_PP-OCRv3_det_infer`)
สำหรับการแปลงโมเดลจาก PaddlePaddle ให้ใช้คำสั่ง `paddle2onnx` หรือ `ovc` (OpenVINO Model Converter) โดยตรงถ้าไฟล์เป็น `.pdmodel`

```bash
uv run ovc basemodel/en_PP-OCRv3_det_infer/inference.pdmodel --output_model modelvino/model.xml
```

### 2. YOLO Object Detection (e.g., `yolo11n.pt`)
สำหรับโมเดล YOLO (v8, v11) ที่เป็น `.pt` สามารถใช้สคริปต์ `convert.py` ที่เตรียมไว้ให้ ซึ่งจะเรียกใช้ `ultralytics` ในการแปลง:

```bash
# 1. ติดตั้ง Dependencies (ทำอัตโนมัติด้วย uv)
# 2. แปลงไฟล์และย้ายไปที่ modelvino/
uv run convert.py
```
> สคริปต์นี้จะสร้าง `model_yolo.xml` และ `model_yolo.bin`

### 3. UniFormer-XXS Semantic Segmentation (e.g., `fpn_xxs_uniformer.pth`)
เนื่องจากไฟล์ `.pth` ของ PyTorch มักเก็บเฉพาะค่า Weights (State Dict) การแปลงจำเป็นต้องมี **Source Code ต้นฉบับ** ของโมเดลนั้นๆ เพื่อโหลดโครงสร้างก่อน

1. แก้ไขไฟล์ `convert_uniformer.py`
2. นำเข้า Class โมเดล (เช่น `from uniformer import UniFormer`) และสร้างอินสแตนซ์
3. รันสคริปต์เพื่อแปลงเป็น OpenVINO

```bash
uv run convert_uniformer.py
```

const { addon: ov } = require('openvino-node');
const sharp = require('sharp');
const fs = require('fs');

async function main() {
    // --- 1. การตั้งค่า ---
    const modelPath = 'modelvino/model.xml';
    const imagePath = 'image/image_test.png';
    const inputSize = 640;

    if (!fs.existsSync(modelPath)) {
        console.error(`ไม่พบไฟล์โมเดลที่: ${modelPath}`);
        return;
    }

    // --- 2. โหลดโมเดล OpenVINO ---
    const core = new ov.Core();
    const model = await core.readModel(modelPath);
    const compiledModel = await core.compileModel(model, 'CPU');
    const inferRequest = compiledModel.createInferRequest();

    // --- 3. เตรียมรูปภาพ (Pre-processing) ---
    // โหลดภาพ -> Resize -> แปลงเป็น Buffer
    const imageBuffer = await sharp(imagePath)
        .resize(inputSize, inputSize, { fit: 'fill' }) // Resize บังคับขนาด (เพื่อความง่าย)
        .removeAlpha() // ตัด channel alpha (transparency) ออก
        .raw()
        .toBuffer();

    // สร้าง Float32Array เพื่อเก็บข้อมูลภาพ
    // ขนาด array = 1 batch * 3 channels * height * width
    const inputData = new Float32Array(3 * inputSize * inputSize);

    // ค่า Mean และ Std สำหรับ Normalize (มาตรฐานของ PaddleOCR/ImageNet)
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    // วนลูปเพื่อย้ายข้อมูลจาก Buffer (HWC) ไปยัง inputData (NCHW) และทำการ Normalize
    // HWC = [Red, Green, Blue, Red, Green, Blue, ...]
    // NCHW = [Red, Red..., Green, Green..., Blue, Blue...]
    for (let i = 0; i < inputSize * inputSize; i++) {
        const r = imageBuffer[i * 3];
        const g = imageBuffer[i * 3 + 1];
        const b = imageBuffer[i * 3 + 2];

        // สูตร: (ค่าสี/255 - mean) / std
        inputData[i] = ((r / 255.0) - mean[0]) / std[0];                             // Channel Red
        inputData[i + (inputSize * inputSize)] = ((g / 255.0) - mean[1]) / std[1];     // Channel Green
        inputData[i + (inputSize * inputSize * 2)] = ((b / 255.0) - mean[2]) / std[2]; // Channel Blue
    }

    // --- 4. สร้าง Tensor และสั่ง Run (Inference) ---
    const inputTensor = new ov.Tensor(
        'f32',                              // precision
        [1, 3, inputSize, inputSize],       // shape [Batch, Channels, Height, Width]
        inputData                           // data
    );

    // ส่ง Tensor เข้าโมเดล (ต้องเช็คชื่อ input ของโมเดล ปกติจะเป็น x)
    // การใช้ setInputTensor(0, ...) คือใส่ช่องแรกสุด
    inferRequest.setInputTensor(0, inputTensor);

    console.log("กำลังประมวลผล...");
    inferRequest.infer();

    // --- 5. รับผลลัพธ์ (Post-processing) ---
    const outputTensor = inferRequest.getOutputTensor(0);
    const resultData = outputTensor.data; // ผลลัพธ์เป็น Float32Array (Heatmap)
    const outputShape = outputTensor.getShape(); // ปกติจะเป็น [1, 1, 640, 640]

    console.log(`\nInference สำเร็จ! Output Shape: ${outputShape}`);

    // *อธิบาย:* ผลลัพธ์ที่ได้คือ Heatmap (ความน่าจะเป็น 0.0 - 1.0) ในแต่ละพิกเซล
    // ว่าตรงนั้นเป็นตัวอักษรหรือไม่

    let detectedPixels = 0;
    const threshold = 0.3; // ค่าความมั่นใจ (ปกติใช้ 0.3)

    // ลองนับจำนวนจุดที่โมเดลมั่นใจว่าเป็นตัวอักษร
    for (let i = 0; i < resultData.length; i++) {
        if (resultData[i] > threshold) {
            detectedPixels++;
        }
    }

    console.log(`จำนวนพิกเซลที่คาดว่าเป็นตัวอักษร: ${detectedPixels}`);

    if (detectedPixels > 0) {
        console.log(">> ตรวจพบข้อความในรูปภาพ! (Detection Found)");
    } else {
        console.log(">> ไม่พบข้อความในรูปภาพ");
    }
}

main().catch(error => {
    console.error("เกิดข้อผิดพลาด:", error);
});
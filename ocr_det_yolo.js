const { addon: ov } = require('openvino-node');
const sharp = require('sharp');
const fs = require('fs');
const path = require('path');

// ตั้งค่า Path และ Parameter
const MODEL_PATH = 'modelvino/model.xml';
const IMAGE_PATH = 'image/image_test.png'; // เปลี่ยนเป็นไฟล์รูปของคุณ
const OUTPUT_PATH = 'result_overlay.png'; // ชื่อไฟล์ผลลัพธ์
const INPUT_SIZE = 640; // ขนาดที่ใช้ Inference
const THRESHOLD = 0.5;  // ค่าความมั่นใจที่จะแสดงผล (0.0 - 1.0)

async function main() {
    console.log("--- เริ่มต้นกระบวนการ ---");
    if (!fs.existsSync(MODEL_PATH) || !fs.existsSync(IMAGE_PATH)) {
        console.error("Error: ไม่พบไฟล์โมเดลหรือไฟล์รูปภาพ");
        return;
    }

    // =========================================
    // ส่วนที่ 1: Inference (เหมือนโค้ดเดิม)
    // =========================================
    console.log("1. Loading Model...");
    const core = new ov.Core();
    const model = await core.readModel(MODEL_PATH);
    // **สำคัญ** ต้องใส่ 'CPU' หรือ device ที่ต้องการ
    const compiledModel = await core.compileModel(model, 'CPU');
    const inferRequest = compiledModel.createInferRequest();

    console.log("2. Pre-processing Image...");
    // เก็บ Metadata รูปต้นฉบับไว้ เพื่อใช้ resize กลับตอนจบ
    const originalMetadata = await sharp(IMAGE_PATH).metadata();

    const imageBuffer = await sharp(IMAGE_PATH)
        .resize(INPUT_SIZE, INPUT_SIZE, { fit: 'fill' })
        .removeAlpha()
        .raw()
        .toBuffer();

    const inputData = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    for (let i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
        const r = imageBuffer[i * 3];
        const g = imageBuffer[i * 3 + 1];
        const b = imageBuffer[i * 3 + 2];
        inputData[i] = ((r / 255.0) - mean[0]) / std[0];
        inputData[i + (INPUT_SIZE * INPUT_SIZE)] = ((g / 255.0) - mean[1]) / std[1];
        inputData[i + (INPUT_SIZE * INPUT_SIZE * 2)] = ((b / 255.0) - mean[2]) / std[2];
    }

    const inputTensor = new ov.Tensor('f32', [1, 3, INPUT_SIZE, INPUT_SIZE], inputData);
    inferRequest.setInputTensor(0, inputTensor);

    console.log("3. Running Inference...");
    inferRequest.infer();

    const outputTensor = inferRequest.getOutputTensor(0);
    const resultData = outputTensor.data; // Float32Array (Heatmap 640x640)
    console.log("Inference Finished. Output shape:", outputTensor.getShape());


    // =========================================
    // ส่วนที่ 2: Plot Heatmap ลงบนภาพ (Visualization)
    // =========================================
    console.log("\n--- เริ่มการสร้างภาพผลลัพธ์ (Visualization) ---");
    console.log("4. Creating Heatmap Overlay...");

    // เราจะสร้าง Raw Buffer สำหรับภาพ Overlay (RGBA 4 channels)
    // ขนาด 640x640 เพื่อให้ตรงกับ output ของโมเดล
    const overlayBufferSize = INPUT_SIZE * INPUT_SIZE * 4; // RGBA = 4 bytes per pixel
    const overlayBuffer = new Uint8Array(overlayBufferSize);

    let detectedPixels = 0;

    for (let i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
        const probability = resultData[i];
        // คำนวณ index ของ buffer RGBA (1 pixel มี 4 ช่อง)
        const offset = i * 4;

        if (probability > THRESHOLD) {
            detectedPixels++;
            // ถ้าความน่าจะเป็นสูง ให้ระบายสีแดงกึ่งโปร่งใส
            overlayBuffer[offset] = 255; // R (แดง)
            overlayBuffer[offset + 1] = 0;   // G
            overlayBuffer[offset + 2] = 0;   // B
            // Alpha (ความโปร่งใส): ยิ่งมั่นใจมาก ยิ่งทึบมาก (สูงสุด 200 จาก 255)
            overlayBuffer[offset + 3] = Math.floor(probability * 200);
        } else {
            // ถ้าต่ำกว่า Threshold ให้เป็นสีใสๆ (Transparent)
            overlayBuffer[offset] = 0;
            overlayBuffer[offset + 1] = 0;
            overlayBuffer[offset + 2] = 0;
            overlayBuffer[offset + 3] = 0; // Alpha = 0 (ใสแจ๋ว)
        }
    }

    console.log(`   Detected approx. ${detectedPixels} text pixels.`);

    if (detectedPixels === 0) {
        console.log("   ไม่พบข้อความที่มั่นใจพอ ไม่มีการสร้างไฟล์ผลลัพธ์");
        return;
    }

    // สร้าง Object ภาพจาก Raw Buffer ที่เราสร้างขึ้น
    const overlayImage = sharp(overlayBuffer, {
        raw: {
            width: INPUT_SIZE,
            height: INPUT_SIZE,
            channels: 4 // RGBA
        }
    });

    console.log(`5. Resizing overlay back to original size (${originalMetadata.width}x${originalMetadata.height})...`);
    // Resize ภาพ Overlay ให้กลับไปเท่ากับภาพต้นฉบับ
    // ใช้ 'fill' เพื่อให้ยืดกลับไปตรงตำแหน่งเดิมเป๊ะๆ
    const resizedOverlay = await overlayImage
        .resize(originalMetadata.width, originalMetadata.height, { fit: 'fill' })
        .png() // แปลงเป็น PNG buffer เพื่อให้ composite เข้าใจ format
        .toBuffer();

    console.log("6. Compositing and saving...");
    // โหลดภาพต้นฉบับ และเอา Overlay วางทับ (Composite)
    await sharp(IMAGE_PATH)
        .composite([{ input: resizedOverlay }])
        .toFile(OUTPUT_PATH);

    console.log(`\n✅ เสร็จสิ้น! บันทึกภาพผลลัพธ์ไว้ที่: ${OUTPUT_PATH}`);
    console.log("เปิดไฟล์ผลลัพธ์ดูได้เลยครับ พื้นที่สีแดงคือบริเวณที่โมเดลคิดว่าเป็นข้อความ");
}

main().catch(error => {
    console.error("เกิดข้อผิดพลาด:", error);
});
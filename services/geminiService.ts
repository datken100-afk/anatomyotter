import { GoogleGenerativeAI } from "@google/generative-ai";
import { Difficulty, GeneratedMCQResponse, MentorResponse } from "../types";

// --- Lấy API Key ---
const apiKey = import.meta.env.VITE_GEMINI_API_KEY || '';

// Khởi tạo Client
const genAI = new GoogleGenerativeAI(apiKey);

// --- CẤU HÌNH MODEL (Dùng tên mã an toàn nhất) ---
// Sử dụng hậu tố '-latest' để tự động chọn bản phù hợp nhất với Key của bạn
const MODEL_NAME = "gemini-1.5-flash-latest";

const generationConfig = {
  temperature: 1,
  topP: 0.95,
  topK: 64,
  maxOutputTokens: 8192,
  responseMimeType: "application/json",
};

interface ContentFile {
    content: string;
    isText: boolean;
}

const LIMIT_THEORY_CHARS = 200000; 
const LIMIT_CLINICAL_CHARS = 100000; 
const LIMIT_SAMPLE_CHARS = 50000; 

// --- HÀM RETRY ---
const wait = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

async function retryGeminiCall<T>(call: () => Promise<T>, retries = 3, delay = 2000): Promise<T> {
  try {
    return await call();
  } catch (error: any) {
    // Nếu lỗi 404 (Model not found) hoặc 429/503 (Quá tải) -> Thử lại
    const msg = error.message || "";
    if (retries > 0 && (msg.includes("429") || msg.includes("503") || msg.includes("404"))) {
      console.warn(`Gemini Error (${msg}). Retrying in ${delay}ms...`);
      await wait(delay);
      return retryGeminiCall(call, retries - 1, delay * 2);
    }
    throw error;
  }
}

export const generateMCQQuestions = async (
  topic: string,
  count: number,
  difficulties: Difficulty[],
  files: { theory?: ContentFile[]; clinical?: ContentFile[]; sample?: ContentFile[] } = {}
): Promise<GeneratedMCQResponse> => {
  if (!apiKey) throw new Error("API Key is missing");

  const model = genAI.getGenerativeModel({ 
    model: MODEL_NAME,
    systemInstruction: `Bạn là giáo sư Y khoa. Tạo ${count} câu hỏi trắc nghiệm giải phẫu chủ đề "${topic}". Độ khó: ${difficulties.join(', ')}. Output JSON: { "questions": [...] }`
  });

  const parts: any[] = [];
  const addFiles = (list: ContentFile[] | undefined, label: string, limit: number) => {
    if (!list) return;
    let chars = 0;
    parts.push({ text: `\n=== ${label} ===\n` });
    for (const f of list) {
        if (chars >= limit) break;
        if (f.isText) {
            parts.push({ text: f.content.substring(0, limit - chars) });
            chars += f.content.length;
        } else {
            const base64 = f.content.includes('base64,') ? f.content.split('base64,')[1] : f.content;
            parts.push({ inlineData: { mimeType: "application/pdf", data: base64 }});
            chars += 10000;
        }
    }
  };

  addFiles(files.theory, "LÝ THUYẾT", LIMIT_THEORY_CHARS);
  addFiles(files.clinical, "LÂM SÀNG", LIMIT_CLINICAL_CHARS);
  addFiles(files.sample, "ĐỀ MẪU", LIMIT_SAMPLE_CHARS);
  parts.push({ text: `Tạo ${count} câu hỏi JSON.` });

  try {
    const result = await retryGeminiCall(() => model.generateContent({
        contents: [{ role: 'user', parts }],
        generationConfig
    }));
    const text = result.response.text().replace(/```json|```/g, '').trim();
    return JSON.parse(text);
  } catch (e: any) {
    throw new Error("Lỗi AI: " + e.message);
  }
};

export const generateStationQuestionFromImage = async (base64Image: string, topic?: string): Promise<any> => {
    const model = genAI.getGenerativeModel({ 
        model: MODEL_NAME, 
        generationConfig: { responseMimeType: "application/json" }
    });
    const cleanBase64 = base64Image.includes('base64,') ? base64Image.split('base64,')[1] : base64Image;
    try {
        const result = await retryGeminiCall(() => model.generateContent([
            topic ? `Chủ đề ${topic}. Tạo câu hỏi trạm.` : "Tạo câu hỏi trạm.",
            { inlineData: { mimeType: "image/jpeg", data: cleanBase64 } }
        ]));
        return JSON.parse(result.response.text().replace(/```json|```/g, '').trim());
    } catch (e) { return { isValid: false, questions: [] }; }
};

export const analyzeResultWithOtter = async (topic: string, stats: any): Promise<MentorResponse> => {
    const model = genAI.getGenerativeModel({ 
        model: MODEL_NAME, 
        generationConfig: { responseMimeType: "application/json" }
    });
    try {
        const result = await retryGeminiCall(() => model.generateContent(
            `Phân tích kết quả thi ${topic}: ${JSON.stringify(stats)}. Output JSON mentor.`
        ));
        return JSON.parse(result.response.text().replace(/```json|```/g, '').trim());
    } catch (e) { return { analysis: "Lỗi...", strengths: [], weaknesses: [], roadmap: [] }; }
};

export const chatWithOtter = async (history: any[], message: string, image?: string): Promise<string> => {
    const model = genAI.getGenerativeModel({ model: MODEL_NAME });
    const chat = model.startChat({
        history: history.map(h => ({ role: h.role === 'model' ? 'model' : 'user', parts: [{ text: h.text }] }))
    });
    try {
        let result;
        if (image) {
             const cleanBase64 = image.includes('base64,') ? image.split('base64,')[1] : image;
             result = await model.generateContent([message, { inlineData: { mimeType: "image/jpeg", data: cleanBase64 } }]);
        } else {
             result = await chat.sendMessage(message);
        }
        return result.response.text();
    } catch (e) { return "Rái cá đang bận... 🦦"; }
};

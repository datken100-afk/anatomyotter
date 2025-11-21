import { GoogleGenerativeAI, SchemaType } from "@google/generative-ai";
import { Difficulty, GeneratedMCQResponse, GeneratedStationResponse, MentorResponse } from "../types";

// --- Lấy API Key ---
const apiKey = import.meta.env.VITE_GEMINI_API_KEY || '';

// Khởi tạo Client (Chuẩn ổn định)
const genAI = new GoogleGenerativeAI(apiKey);

// Cấu hình Model
const modelId = "gemini-1.5-flash";
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

// Limits
const LIMIT_THEORY_CHARS = 200000; 
const LIMIT_CLINICAL_CHARS = 100000; 
const LIMIT_SAMPLE_CHARS = 50000; 

// --- HÀM RETRY ---
const wait = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

async function retryGeminiCall<T>(
  call: () => Promise<T>,
  retries: number = 3,
  initialDelay: number = 2000
): Promise<T> {
  let lastError: any;
  for (let i = 0; i < retries; i++) {
    try {
      return await call();
    } catch (error: any) {
      lastError = error;
      const msg = error.message || "";
      if (msg.includes("429") || msg.includes("503") || msg.includes("Overloaded")) {
        console.warn(`Gemini Busy. Retrying in ${initialDelay}ms...`);
        await wait(initialDelay);
        initialDelay *= 2; 
      } else {
        throw error; 
      }
    }
  }
  throw lastError;
}

export const generateMCQQuestions = async (
  topic: string,
  count: number,
  difficulties: Difficulty[],
  files: { theory?: ContentFile[]; clinical?: ContentFile[]; sample?: ContentFile[] } = {}
): Promise<GeneratedMCQResponse> => {
  if (!apiKey) throw new Error("API Key is missing");

  const model = genAI.getGenerativeModel({ 
    model: modelId,
    systemInstruction: `
    Bạn là giáo sư Y khoa. Tạo ${count} câu hỏi trắc nghiệm giải phẫu chủ đề "${topic}".
    Độ khó: ${difficulties.join(', ')}.
    Output JSON format: { "questions": [{ "question": "...", "options": ["A", "B", "C", "D"], "correctAnswer": "...", "explanation": "...", "difficulty": "..." }] }
    `
  });

  // Xử lý file input
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
            // Base64 Image/PDF
            const base64 = f.content.includes('base64,') ? f.content.split('base64,')[1] : f.content;
            parts.push({ inlineData: { mimeType: "application/pdf", data: base64 }});
            chars += 10000;
        }
    }
  };

  addFiles(files.theory, "LÝ THUYẾT", LIMIT_THEORY_CHARS);
  addFiles(files.clinical, "LÂM SÀNG", LIMIT_CLINICAL_CHARS);
  addFiles(files.sample, "ĐỀ MẪU", LIMIT_SAMPLE_CHARS);

  parts.push({ text: `Hãy tạo ${count} câu hỏi trắc nghiệm JSON.` });

  try {
    const result = await retryGeminiCall(() => model.generateContent({
        contents: [{ role: 'user', parts }],
        generationConfig
    }));
    
    const responseText = result.response.text();
    const jsonText = responseText.replace(/```json|```/g, '').trim();
    return JSON.parse(jsonText) as GeneratedMCQResponse;
  } catch (e: any) {
    console.error(e);
    throw new Error("Lỗi tạo đề: " + e.message);
  }
};

// --- Spot Test (Vision) ---
export const generateStationQuestionFromImage = async (base64Image: string, topic?: string): Promise<any> => {
    const model = genAI.getGenerativeModel({ 
        model: "gemini-1.5-flash",
        generationConfig: { responseMimeType: "application/json" }
    });

    const prompt = topic ? `Chủ đề: ${topic}. Kiểm tra hình và tạo câu hỏi trạm.` : "Tạo câu hỏi trạm giải phẫu.";
    const cleanBase64 = base64Image.includes('base64,') ? base64Image.split('base64,')[1] : base64Image;

    try {
        const result = await retryGeminiCall(() => model.generateContent([
            prompt,
            { inlineData: { mimeType: "image/jpeg", data: cleanBase64 } }
        ]));
        return JSON.parse(result.response.text().replace(/```json|```/g, '').trim());
    } catch (e) {
        return { isValid: false, questions: [] };
    }
};

// --- Mentor ---
export const analyzeResultWithOtter = async (topic: string, stats: any): Promise<MentorResponse> => {
    const model = genAI.getGenerativeModel({ 
        model: "gemini-1.5-flash",
        generationConfig: { responseMimeType: "application/json" }
    });
    
    try {
        const result = await retryGeminiCall(() => model.generateContent(`
            Đóng vai Rái cá nhỏ mentor. Phân tích kết quả thi chủ đề ${topic}: ${JSON.stringify(stats)}.
            Output JSON: { "analysis": "...", "strengths": [], "weaknesses": [], "roadmap": [{ "step": "...", "details": "..." }] }
        `));
        return JSON.parse(result.response.text().replace(/```json|```/g, '').trim()) as MentorResponse;
    } catch (e) {
        return { analysis: "Lỗi kết nối...", strengths: [], weaknesses: [], roadmap: [] };
    }
};

// --- Chat ---
export const chatWithOtter = async (history: any[], message: string, image?: string): Promise<string> => {
    const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
    
    // Convert history format
    const chatHistory = history.map(h => ({
        role: h.role === 'model' ? 'model' : 'user',
        parts: [{ text: h.text }]
    }));

    const chat = model.startChat({ history: chatHistory });
    
    try {
        let result;
        if (image) {
             const cleanBase64 = image.includes('base64,') ? image.split('base64,')[1] : image;
             result = await model.generateContent([message, { inlineData: { mimeType: "image/jpeg", data: cleanBase64 } }]);
        } else {
             result = await chat.sendMessage(message);
        }
        return result.response.text();
    } catch (e) {
        return "Rái cá đang bận... thử lại sau nhé!";
    }
};

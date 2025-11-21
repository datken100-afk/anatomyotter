import { GoogleGenAI, Type, Schema, GenerateContentResponse } from "@google/genai";
import { Difficulty, GeneratedMCQResponse, GeneratedStationResponse, MentorResponse, StationItem } from "../types";

// --- Lấy API Key theo chuẩn Vite ---
const apiKey = import.meta.env.VITE_GEMINI_API_KEY || '';

// Khởi tạo Gemini Client
const ai = new GoogleGenAI({ apiKey });

// Sử dụng bản Flash ổn định nhất hiện nay
const modelId = "gemini-1.5-flash";

interface ContentFile {
    content: string;
    isText: boolean;
}

// Giới hạn Token (Ước tính 1 token = 4 ký tự)
const LIMIT_THEORY_CHARS = 2400000; 
const LIMIT_CLINICAL_CHARS = 1000000; 
const LIMIT_SAMPLE_CHARS = 200000; 

// --- HÀM RETRY (Thử lại khi lỗi mạng) ---
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
      
      const isRateLimit = 
        error.status === 429 || 
        error.status === 503 ||
        (error.message && (
          error.message.includes("429") || 
          error.message.includes("quota") || 
          error.message.includes("RESOURCE_EXHAUSTED") ||
          error.message.includes("Overloaded")
        ));

      if (isRateLimit) {
        if (i === retries - 1) break;
        console.warn(`Gemini Rate Limit. Retrying in ${initialDelay}ms...`);
        await wait(initialDelay);
        initialDelay *= 2; 
      } else {
        throw error; 
      }
    }
  }
  
  const cleanMsg = lastError?.message || "Unknown error";
  if (cleanMsg.includes("quota") || cleanMsg.includes("RESOURCE_EXHAUSTED")) {
      throw new Error("Đã hết hạn mức sử dụng AI (Quota Exceeded). Vui lòng kiểm tra gói cước hoặc thử lại vào ngày mai.");
  }
  throw new Error("Hệ thống AI đang quá tải. Vui lòng thử lại sau vài giây.");
}

export const generateMCQQuestions = async (
  topic: string,
  count: number,
  difficulties: Difficulty[],
  files: { theory?: ContentFile[]; clinical?: ContentFile[]; sample?: ContentFile[] } = {}
): Promise<GeneratedMCQResponse> => {
  if (!apiKey) throw new Error("API Key is missing");

  // 1. Tạo câu lệnh (Prompt)
  let systemInstruction = `
    Bạn là một giáo sư Y khoa hàng đầu. Nhiệm vụ của bạn là tạo đề thi trắc nghiệm giải phẫu học chất lượng cao.
    
    QUY TẮC PHÂN TÍCH TÀI LIỆU:
    1. DỮ LIỆU LÝ THUYẾT (Theory): CHỈ được sử dụng để tạo các câu hỏi thuộc mức độ: 
       - ${Difficulty.REMEMBER} (Ghi nhớ)
       - ${Difficulty.UNDERSTAND} (Hiểu)
       - ${Difficulty.APPLY} (Vận dụng thấp)
       AI cần phân biệt rõ ba mức độ này dựa trên độ sâu của kiến thức.

    2. DỮ LIỆU LÂM SÀNG (Clinical): CHỈ được sử dụng để tạo câu hỏi mức độ:
       - ${Difficulty.CLINICAL} (Lâm sàng/Ca bệnh)
       Câu hỏi lâm sàng bắt buộc phải là các Case Study (tình huống bệnh nhân) cụ thể.

    3. ĐỀ THI MẪU: Nếu có, hãy học phong cách đặt câu hỏi từ đó.

    CẤU TRÚC ĐỀ THI:
    - Tổng số câu: ${count} câu.
    - Chủ đề: "${topic}".
    - Các mức độ khó: ${difficulties.join(', ')}.
    - Mỗi câu hỏi có 4 lựa chọn, 1 đáp án đúng.
    - Giải thích: Phải cực kỳ chi tiết.
  `;

  const schema: Schema = {
    type: Type.OBJECT,
    properties: {
      questions: {
        type: Type.ARRAY,
        items: {
          type: Type.OBJECT,
          properties: {
            question: { type: Type.STRING },
            options: { type: Type.ARRAY, items: { type: Type.STRING } },
            correctAnswer: { type: Type.STRING },
            explanation: { type: Type.STRING },
            difficulty: { type: Type.STRING },
          },
          required: ["question", "options", "correctAnswer", "explanation", "difficulty"],
        },
      },
    },
    required: ["questions"],
  };

  // 2. Xử lý file đính kèm
  const parts: any[] = [];

  const addContentParts = (fileItems: ContentFile[] | undefined, sectionTitle: string, usageInstruction: string, charLimit: number) => {
    if (!fileItems || fileItems.length === 0) return;

    parts.push({ text: `\n=== BẮT ĐẦU PHẦN: ${sectionTitle} ===\nCHỈ DẪN: ${usageInstruction}\n` });
    
    let currentChars = 0;

    for (const item of fileItems) {
        if (currentChars >= charLimit) {
             parts.push({ text: `\n[Đã ngưng tải thêm tài liệu do quá lớn]\n` });
             break;
        }

        if (item.content) {
            if (item.isText) {
                let textToAdd = item.content;
                const remaining = charLimit - currentChars;
                if (textToAdd.length > remaining) {
                    textToAdd = textToAdd.substring(0, remaining) + "\n\n[...]";
                }
                parts.push({ text: `\n--- FILE CONTENT ---\n${textToAdd}\n` });
                currentChars += textToAdd.length;
            } else {
                const base64Data = item.content.includes('base64,') ? item.content.split('base64,')[1] : item.content;
                parts.push({
                    inlineData: {
                        mimeType: "application/pdf", 
                        data: base64Data
                    }
                });
                currentChars += 50000; 
            }
        }
    }
    parts.push({ text: `=== KẾT THÚC PHẦN: ${sectionTitle} ===\n` });
  };

  addContentParts(files.theory, "LÝ THUYẾT", "Dùng cho câu hỏi Ghi nhớ/Hiểu/Vận dụng.", LIMIT_THEORY_CHARS);
  addContentParts(files.clinical, "LÂM SÀNG", "Dùng cho câu hỏi Lâm sàng.", LIMIT_CLINICAL_CHARS);
  addContentParts(files.sample, "ĐỀ MẪU", "Tham khảo.", LIMIT_SAMPLE_CHARS);

  parts.push({ text: `Hãy soạn thảo ${count} câu hỏi trắc nghiệm về chủ đề "${topic}" theo đúng định dạng JSON.` });

  try {
    const response = await retryGeminiCall<GenerateContentResponse>(() => ai.models.generateContent({
      model: modelId,
      contents: { parts: parts },
      config: {
        systemInstruction: systemInstruction,
        responseMimeType: "application/json",
        responseSchema: schema,
      },
    }));

    let text = response.text;
    if (!text) throw new Error("No response from AI");
    
    const jsonBlockMatch = text.match(/```json\s*([\s\S]*?)\s*```/);
    if (jsonBlockMatch) {
        text = jsonBlockMatch[1];
    } else {
        text = text.replace(/```json/g, '').replace(/```/g, '');
    }
    
    return JSON.parse(text.trim()) as GeneratedMCQResponse;

  } catch (error: any) {
    console.error("Gemini API Error:", error);
    if (error.message && error.message.includes("token count exceeds")) {
        throw new Error("Tài liệu quá lớn. Vui lòng bớt file lại.");
    }
    throw error;
  }
};

// --- Spot Test (Vision) ---
export interface StationQuestionResponse {
    isValid: boolean;
    questions?: {
        questionText: string;
        correctAnswer: string;
        explanation: string;
    }[];
}

export const generateStationQuestionFromImage = async (base64Image: string, topic?: string): Promise<StationQuestionResponse> => {
    const systemInstruction = `
    Bạn là giám khảo thi chạy trạm (Spot Test) Giải phẫu học.
    
    1. KIỂM TRA:
       - Hình ảnh phải rõ ràng và liên quan đến chủ đề: "${topic || 'Giải phẫu'}".
       - Nếu sai chủ đề hoặc không phải giải phẫu -> isValid = false.

    2. RA ĐỀ (Nếu isValid = true):
       - Chọn MỘT cấu trúc trong hình.
       - Đặt câu hỏi định danh (Cấu trúc này là gì?).
       - Đáp án chính xác (Tiếng Việt).

    Output JSON: { "isValid": boolean, "questions": [...] }
    `;

    const prompt = topic 
        ? `Kiểm tra hình này có thuộc chủ đề "${topic}" không. Nếu có, tạo câu hỏi.` 
        : "Kiểm tra hình giải phẫu và tạo câu hỏi.";

    try {
        const cleanBase64 = base64Image.includes('base64,') ? base64Image.split('base64,')[1] : base64Image;
        
        const response = await retryGeminiCall<GenerateContentResponse>(() => ai.models.generateContent({
            model: "gemini-1.5-flash", 
            contents: { 
                role: 'user', 
                parts: [
                    { text: prompt },
                    { inlineData: { mimeType: 'image/jpeg', data: cleanBase64 } }
                ] 
            },
            config: {
                systemInstruction: systemInstruction,
                responseMimeType: "application/json",
                responseSchema: {
                    type: Type.OBJECT,
                    properties: {
                        isValid: { type: Type.BOOLEAN },
                        questions: {
                            type: Type.ARRAY,
                            items: {
                                type: Type.OBJECT,
                                properties: {
                                    questionText: { type: Type.STRING },
                                    correctAnswer: { type: Type.STRING },
                                    explanation: { type: Type.STRING }
                                },
                                required: ["questionText", "correctAnswer", "explanation"]
                            }
                        }
                    },
                    required: ["isValid"]
                }
            }
        }));

        let text = response.text || "";
        text = text.replace(/```json/g, '').replace(/```/g, '').trim();
        return JSON.parse(text) as StationQuestionResponse;
    } catch (e: any) {
        return { isValid: false, questions: [] };
    }
};

// --- Rái cá Mentor ---
export const analyzeResultWithOtter = async (
    topic: string,
    stats: Record<string, { correct: number, total: number }>
): Promise<MentorResponse> => {
    const statsDescription = Object.entries(stats)
        .map(([diff, val]) => `- ${diff}: ${val.correct}/${val.total} câu`)
        .join('\n');

    const prompt = `
    Đóng vai "Rái cá nhỏ" (Little Otter) 🦦 gia sư giải phẫu.
    Học viên vừa thi chủ đề: "${topic}". Kết quả:
    ${statsDescription}
    
    Hãy đưa ra:
    1. Nhận xét dí dỏm.
    2. Điểm mạnh/Yếu.
    3. Lộ trình cải thiện (4 bước cụ thể).

    Output JSON: { "analysis": "...", "strengths": [], "weaknesses": [], "roadmap": [{ "step": "...", "details": "..." }] }
    `;

    try {
        const response = await retryGeminiCall<GenerateContentResponse>(() => ai.models.generateContent({
            model: "gemini-1.5-flash",
            contents: { role: 'user', parts: [{ text: prompt }] },
            config: { responseMimeType: "application/json" }
        }));

        let text = response.text || "";
        text = text.replace(/```json/g, '').replace(/```/g, '').trim();
        return JSON.parse(text) as MentorResponse;
    } catch (e) {
        return {
            analysis: "Rái cá đang bận bắt cá rồi! 🦦",
            strengths: [], weaknesses: [], roadmap: []
        };
    }
};

// --- Chatbot ---
export const chatWithOtter = async (history: {role: 'user' | 'model', text: string, image?: string}[], message: string, image?: string): Promise<string> => {
    const model = "gemini-1.5-flash"; 
    const systemInstruction = `Bạn là Rái cá nhỏ 🦦 chuyên về Giải phẫu. Trả lời ngắn gọn, vui vẻ, chính xác.`;

    const contents = history.map(msg => {
        const parts: any[] = [{ text: msg.text }];
        if (msg.image) {
             try {
                 const base64Data = msg.image.includes('base64,') ? msg.image.split('base64,')[1] : msg.image;
                 const mimeType = msg.image.match(/data:([^;]+);base64,/)?.[1] ||

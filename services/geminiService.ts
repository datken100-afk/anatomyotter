import { GoogleGenAI, Type, Schema, GenerateContentResponse } from "@google/genai";
import { Difficulty, GeneratedMCQResponse, GeneratedStationResponse, MentorResponse, StationItem } from "../types";

// --- SỬA Ở ĐÂY: Đổi sang cách lấy key của Vite ---
const apiKey = import.meta.env.VITE_GEMINI_API_KEY || '';

// Initialize Gemini Client
const ai = new GoogleGenAI({ apiKey });

// UPGRADE: Use Gemini 3 Pro for superior reasoning, thinking capabilities, and context handling
const modelId = "gemini-2.0-flash-thinking-exp-1219"; // Mình đổi tạm sang bản Flash Thinking ổn định hơn, bản Pro Preview đôi khi cần quyền truy cập đặc biệt

interface ContentFile {
    content: string;
    isText: boolean;
}

// Token Limits (Approximate 1 token = 4 chars)
// Limit total text input to ~3.5M characters (~875k tokens) to be safe under the 1M token limit
const LIMIT_THEORY_CHARS = 2400000; // ~600k tokens
const LIMIT_CLINICAL_CHARS = 1000000; // ~250k tokens
const LIMIT_SAMPLE_CHARS = 200000; // ~50k tokens

// --- RETRY LOGIC HELPER ---
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
      
      // Check for Rate Limit (429) or Quota Exceeded or Service Unavailable (503)
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
        if (i === retries - 1) break; // Don't wait on the last fail
        console.warn(`Gemini Rate Limit/Overload hit. Retrying in ${initialDelay}ms... (Attempt ${i + 1}/${retries})`);
        await wait(initialDelay);
        initialDelay *= 2; // Exponential backoff
      } else {
        throw error; // Not a rate limit error, throw immediately
      }
    }
  }
  
  // If we get here, we exhausted retries
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

  // 1. Construct the prompt with STRICT instructions for file usage
  let systemInstruction = `
    Bạn là một giáo sư Y khoa hàng đầu. Nhiệm vụ của bạn là tạo đề thi trắc nghiệm giải phẫu học chất lượng cao.
    
    QUY TẮC PHÂN TÍCH TÀI LIỆU (TUÂN THỦ TUYỆT ĐỐI):
    1. DỮ LIỆU LÝ THUYẾT (Theory): CHỈ được sử dụng để tạo các câu hỏi thuộc mức độ: 
       - ${Difficulty.REMEMBER} (Ghi nhớ)
       - ${Difficulty.UNDERSTAND} (Hiểu)
       - ${Difficulty.APPLY} (Vận dụng thấp)
       AI cần phân biệt rõ ba mức độ này dựa trên độ sâu của

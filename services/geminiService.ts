
import { GoogleGenAI, Type, Schema, GenerateContentResponse } from "@google/genai";
import { Difficulty, GeneratedMCQResponse, GeneratedStationResponse, MentorResponse, StationItem } from "../types";

const apiKey = process.env.API_KEY || '';

// Initialize Gemini Client
const ai = new GoogleGenAI({ apiKey });

// UPGRADE: Use Gemini 3 Pro for superior reasoning, thinking capabilities, and context handling
const modelId = "gemini-3-pro-preview";

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
       AI cần phân biệt rõ ba mức độ này dựa trên độ sâu của kiến thức lý thuyết.

    2. DỮ LIỆU LÂM SÀNG (Clinical): CHỈ được sử dụng để tạo câu hỏi mức độ:
       - ${Difficulty.CLINICAL} (Lâm sàng/Ca bệnh)
       Câu hỏi lâm sàng bắt buộc phải là các Case Study (tình huống bệnh nhân) cụ thể, yêu cầu chẩn đoán, tiên lượng hoặc giải phẫu ứng dụng thực tế.

    3. ĐỀ THI MẪU: Nếu có, hãy học phong cách đặt câu hỏi và format từ đó.

    CẤU TRÚC ĐỀ THI:
    - Tổng số câu: ${count} câu.
    - Chủ đề: "${topic}".
    - Các mức độ khó yêu cầu: ${difficulties.join(', ')}.
    - Mỗi câu hỏi có 4 lựa chọn, 1 đáp án đúng.
    - Giải thích: Phải cực kỳ chi tiết, trích dẫn lý do tại sao đúng/sai.
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
            difficulty: { type: Type.STRING, description: "Mức độ khó chính xác (Ghi nhớ, Hiểu, Vận dụng thấp, Lâm sàng)" },
          },
          required: ["question", "options", "correctAnswer", "explanation", "difficulty"],
        },
      },
    },
    required: ["questions"],
  };

  // 2. Construct Multimodal Parts with Explicit Context Separation
  const parts: any[] = [];

  // Helper to add and truncate content parts
  const addContentParts = (fileItems: ContentFile[] | undefined, sectionTitle: string, usageInstruction: string, charLimit: number) => {
    if (!fileItems || fileItems.length === 0) return;

    parts.push({ text: `\n=== BẮT ĐẦU PHẦN: ${sectionTitle} ===\nCHỈ DẪN: ${usageInstruction}\n` });
    
    let currentChars = 0;

    for (const item of fileItems) {
        // Stop adding files if limit is reached
        if (currentChars >= charLimit) {
             parts.push({ text: `\n[CẢNH BÁO: Đã ngưng tải thêm tài liệu phần này do vượt quá giới hạn bộ nhớ cho phép]\n` });
             break;
        }

        if (item.content) {
            if (item.isText) {
                // Case 1: Extracted Text
                let textToAdd = item.content;
                const remaining = charLimit - currentChars;

                if (textToAdd.length > remaining) {
                    textToAdd = textToAdd.substring(0, remaining) + "\n\n[...Nội dung file này đã bị cắt bớt do giới hạn bộ nhớ AI...]";
                }
                
                parts.push({ text: `\n--- FILE CONTENT ---\n${textToAdd}\n` });
                currentChars += textToAdd.length;
            } else {
                // Case 2: Base64 PDF/Image (Only for small files < 20MB)
                // Cannot easily count chars for binary, but assume it takes up context.
                // Check mimeType if available, default to pdf assumption for now.
                const base64Data = item.content.includes('base64,') ? item.content.split('base64,')[1] : item.content;
                parts.push({
                    inlineData: {
                        mimeType: "application/pdf", 
                        data: base64Data
                    }
                });
                // Arbitrary penalty for binary file to avoid infinite loop if mixed
                currentChars += 50000; 
            }
        }
    }
    parts.push({ text: `=== KẾT THÚC PHẦN: ${sectionTitle} ===\n` });
  };

  // Add files with strict limits
  addContentParts(
    files.theory, 
    "TÀI LIỆU LÝ THUYẾT", 
    `Dùng cho câu hỏi mức độ ${Difficulty.REMEMBER}, ${Difficulty.UNDERSTAND}, ${Difficulty.APPLY}.`,
    LIMIT_THEORY_CHARS
  );
  
  addContentParts(
    files.clinical, 
    "TÀI LIỆU LÂM SÀNG", 
    `CHỈ Dùng cho câu hỏi mức độ ${Difficulty.CLINICAL} (Case Study).`,
    LIMIT_CLINICAL_CHARS
  );
  
  addContentParts(
    files.sample, 
    "ĐỀ THI MẪU", 
    "Tham khảo cách đặt câu hỏi.",
    LIMIT_SAMPLE_CHARS
  );

  // Add the final trigger prompt
  parts.push({ text: `Hãy "Suy nghĩ" (Thinking) kỹ về phân phối câu hỏi, sau đó soạn thảo ${count} câu hỏi trắc nghiệm về chủ đề "${topic}" theo đúng định dạng JSON đã yêu cầu.` });

  try {
    const response = await retryGeminiCall<GenerateContentResponse>(() => ai.models.generateContent({
      model: modelId,
      contents: {
        parts: parts,
      },
      config: {
        systemInstruction: systemInstruction,
        responseMimeType: "application/json",
        responseSchema: schema,
        // Thinking Budget: Allows the model to plan the question distribution and validate clinical logic
        thinkingConfig: { thinkingBudget: 2048 }, 
      },
    }));

    let text = response.text;
    if (!text) throw new Error("No response from AI");
    
    // Robust JSON Cleaning
    const jsonBlockMatch = text.match(/```json\s*([\s\S]*?)\s*```/);
    if (jsonBlockMatch) {
        text = jsonBlockMatch[1];
    } else {
        text = text.replace(/```json/g, '').replace(/```/g, '');
    }
    
    text = text.trim();

    let parsed: any;
    try {
      parsed = JSON.parse(text);
    } catch (e) {
      console.error("Failed to parse JSON:", text);
      throw new Error("AI returned invalid JSON format. Please try again.");
    }

    if (!parsed || typeof parsed !== 'object') {
       throw new Error("Invalid response structure");
    }

    if (!Array.isArray(parsed.questions)) {
        throw new Error("Response missing 'questions' array");
    }

    return parsed as GeneratedMCQResponse;

  } catch (error: any) {
    console.error("Gemini API Error:", error);
    
    // Pass through the specific rate limit error thrown by retryGeminiCall
    if (error.message && (error.message.includes("quá tải") || error.message.includes("hết hạn mức"))) {
        throw error;
    }

    // Enhance token error
    if (error.message && error.message.includes("token count exceeds")) {
        throw new Error("Tổng dung lượng tài liệu quá lớn vượt quá giới hạn của AI. Vui lòng bớt file hoặc dùng file nhỏ hơn.");
    }
    
    throw error;
  }
};

// --- Generate Spot Test Question from Image (Vision) ---
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
    Bạn là giám khảo thi chạy trạm (Spot Test) Giải phẫu học cực kỳ nghiêm túc.
    Bạn sẽ được cung cấp một hình ảnh từ tài liệu PDF.
    
    NHIỆM VỤ 1: KIỂM TRA TÍNH HỢP LỆ & ĐÚNG CHỦ ĐỀ (QUAN TRỌNG NHẤT)
    - Hình ảnh HỢP LỆ (isValid = true) PHẢI THỎA MÃN CẢ 2 ĐIỀU KIỆN:
       1. Là hình giải phẫu minh họa rõ ràng, có đường chỉ dẫn (leader lines) hoặc số chú thích.
       2. NỘI DUNG HÌNH ẢNH PHẢI LIÊN QUAN ĐẾN CHỦ ĐỀ: "${topic || 'Giải phẫu học'}".
          - Nếu chủ đề là "Tim", nhưng hình là "Xương đùi" -> isValid = false.
          - Nếu chủ đề là "Thần kinh", nhưng hình chỉ có "Cơ bắp" -> isValid = false.

    - Hình ảnh KHÔNG HỢP LỆ (isValid = false): 
       + Trang sách chỉ toàn chữ (Text-only).
       + Mục lục, bìa sách.
       + Hình ảnh sai chủ đề.
       + Hình ảnh quá mờ.

    NHIỆM VỤ 2: RA ĐỀ (Chỉ khi isValid = true)
    
    Quy tắc ra đề:
    1. Chọn MỘT cấu trúc giải phẫu quan trọng nhất trong hình LIÊN QUAN ĐẾN CHỦ ĐỀ "${topic}".
    2. Đặt câu hỏi định danh trực tiếp. Ví dụ: "Cấu trúc được chỉ định là gì?", "Chi tiết số X là gì?".
    3. Đáp án phải là Tên giải phẫu chính xác (Tiếng Việt).
    4. Giải thích ngắn gọn.

    Output JSON format:
    {
      "isValid": boolean,
      "questions": [
        {
          "questionText": "Câu hỏi...",
          "correctAnswer": "Tên cấu trúc",
          "explanation": "Giải thích..."
        }
      ]
    }
    `;

    const prompt = topic 
        ? `Kiểm tra xem hình này có chứa cấu trúc giải phẫu thuộc chủ đề "${topic}" không. Nếu có, hãy tạo 1 câu hỏi trạm.` 
        : "Kiểm tra xem đây có phải là hình giải phẫu hợp lệ không. Nếu có, hãy tạo 1 câu hỏi trạm.";

    try {
        // Remove header if present to get raw base64
        const cleanBase64 = base64Image.includes('base64,') ? base64Image.split('base64,')[1] : base64Image;
        
        const response = await retryGeminiCall<GenerateContentResponse>(() => ai.models.generateContent({
            model: "gemini-2.5-flash", // Use Flash for Vision speed/efficiency
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
        const parsed = JSON.parse(text) as StationQuestionResponse;
        
        return parsed;
    } catch (e: any) {
        console.error("Vision API Error", e);
        // IMPORTANT: If we hit a rate limit or quota limit even after retries, we MUST throw 
        // to stop the loop in StationMode.
        if (e.message && (e.message.includes("quá tải") || e.message.includes("quota") || e.message.includes("429"))) {
            throw e;
        }
        // For other errors (e.g. bad image format), just return invalid so we skip this page
        return { isValid: false, questions: [] };
    }
};

export const analyzeResultWithOtter = async (
    topic: string,
    stats: Record<string, { correct: number, total: number }>
): Promise<MentorResponse> => {
    // Format stats into a readable string for the prompt
    const statsDescription = Object.entries(stats)
        .map(([diff, val]) => {
             const pct = val.total > 0 ? Math.round((val.correct / val.total) * 100) : 0;
             return `- ${diff}: ${val.correct}/${val.total} câu (${pct}%)`;
        })
        .join('\n');

    const prompt = `
    Đóng vai là "Rái cá nhỏ" (Little Otter) - một gia sư AI giải phẫu học cực kỳ thông minh, hài hước, hay dùng emoji 🦦.
    
    Học viên vừa làm bài thi về chủ đề: "${topic}".
    
    DỮ LIỆU KẾT QUẢ (STATS):
    ${statsDescription}
    
    NHIỆM VỤ CỦA BẠN (Yêu cầu độ chi tiết cao):
    1. PHÂN TÍCH SÂU (Deep Analysis): 
       - Dựa vào stats, nhận xét về năng lực hiện tại.
       - Đưa ra lời nhận xét dí dỏm nhưng thấm thía.

    2. ĐÁNH GIÁ CHI TIẾT:
       - Điểm mạnh: Các phần làm tốt.
       - Điểm yếu: Các phần hay sai.

    3. LỘ TRÌNH CẢI THIỆN (Actionable Roadmap - CỰC KỲ QUAN TRỌNG):
       - Hãy thiết kế 4 bước hành động cụ thể để khắc phục điểm yếu nhất.
       - KHÔNG ĐƯỢC viết chung chung như "Học lại lý thuyết" hay "Đọc thêm sách".
       - HÃY VIẾT CÁC KỸ THUẬT CỤ THỂ, ví dụ: 
         + "Vẽ lại sơ đồ đám rối thần kinh cánh tay 3 lần bằng trí nhớ (Active Recall)."
         + "So sánh nguyên ủy/bám tận của nhóm cơ gấp và duỗi (Comparative Study)."
         + "Giải thích cơ chế bệnh sinh của ca lâm sàng X cho người khác nghe (Feynman Technique)."
         + "Tạo Flashcard Anki cho các nhánh bên động mạch."
       - Mục "details" phải dài khoảng 2-3 câu, hướng dẫn cách làm chi tiết.

    Output JSON format:
    {
      "analysis": "Lời nhận xét chung...",
      "strengths": ["Điểm mạnh 1", "Điểm mạnh 2"],
      "weaknesses": ["Điểm yếu 1", "Điểm yếu 2"],
      "roadmap": [
         { "step": "Tên phương pháp (VD: Kỹ thuật Vẽ hồi tưởng)", "details": "Hướng dẫn chi tiết cách thực hiện..." }
      ]
    }
    `;

    try {
        const response = await retryGeminiCall<GenerateContentResponse>(() => ai.models.generateContent({
            model: "gemini-3-pro-preview",
            contents: { role: 'user', parts: [{ text: prompt }] },
            config: {
                responseMimeType: "application/json",
                thinkingConfig: { thinkingBudget: 2048 } // Increased budget for detailed roadmap planning
            }
        }));

        let text = response.text || "";
        text = text.replace(/```json/g, '').replace(/```/g, '').trim();
        return JSON.parse(text) as MentorResponse;
    } catch (e) {
        console.error(e);
        return {
            analysis: "Úi cha! Rái cá đang bận bắt cá nên không phân tích được rồi. Thử lại sau nhé! 🦦",
            strengths: [],
            weaknesses: [],
            roadmap: []
        };
    }
};

export const chatWithOtter = async (history: {role: 'user' | 'model', text: string, image?: string}[], message: string, image?: string): Promise<string> => {
    // Use Flash for speed in chat
    const model = "gemini-2.5-flash"; 
    
    const systemInstruction = `Bạn là "Rái cá nhỏ" (Little Otter) 🦦 - một trợ lý ảo chuyên về GIẢI PHẪU HỌC (Anatomy).
    
    TÍNH CÁCH & PHONG CÁCH TRẢ LỜI:
    - Vui vẻ, thân thiện, nhưng cực kỳ chuyên nghiệp về kiến thức y khoa.
    - Dùng emoji (🦦, 🦴, 🧠) hợp lý để tạo cảm giác gần gũi.
    
    QUY TẮC ĐỊNH DẠNG VĂN BẢN (QUAN TRỌNG):
    1. TRÌNH BÀY GỌN GÀNG:
       - Sử dụng **in đậm** (bold) CHỈ cho các từ khóa chính (thuật ngữ giải phẫu).
       - HẠN CHẾ DÙNG quá nhiều ký tự # (header) nếu đoạn văn ngắn.
       - Sử dụng gạch đầu dòng (-) để liệt kê ý.
       - KHÔNG dùng quá nhiều ký tự đặc biệt gây rối mắt (** không cần thiết thì đừng dùng).
    
    2. CẤU TRÚC:
       - Tách đoạn ngắn, dễ đọc.
       - Tập trung vào thông tin chính xác, tránh lan man.
    
    NHIỆM VỤ:
    - Giải đáp mọi câu hỏi về cấu trúc giải phẫu, chức năng sinh lý, liên hệ lâm sàng.
    - Phân tích hình ảnh giải phẫu nếu người dùng gửi.
    - Từ chối khéo léo các câu hỏi không liên quan đến Y học.
    `;

    // Construct Gemini content format
    const contents = history.map(msg => {
        const parts: any[] = [{ text: msg.text }];
        if (msg.image) {
             // Simple base64 extraction assuming data URL
             try {
                 const base64Data = msg.image.includes('base64,') ? msg.image.split('base64,')[1] : msg.image;
                 const mimeType = msg.image.match(/data:([^;]+);base64,/)?.[1] || 'image/jpeg';
                 parts.push({ inlineData: { mimeType, data: base64Data }});
             } catch (e) {
                 console.warn("Could not process history image", e);
             }
        }
        return { role: msg.role, parts };
    });

    const currentParts: any[] = [{ text: message }];
    if (image) {
        try {
            const base64Data = image.includes('base64,') ? image.split('base64,')[1] : image;
            const mimeType = image.match(/data:([^;]+);base64,/)?.[1] || 'image/jpeg';
            currentParts.push({ inlineData: { mimeType, data: base64Data }});
        } catch (e) {
             console.warn("Could not process current image", e);
        }
    }
    contents.push({ role: 'user', parts: currentParts });

    try {
        const response = await retryGeminiCall<GenerateContentResponse>(() => ai.models.generateContent({
            model,
            contents,
            config: { systemInstruction }
        }));
        return response.text || "Rái cá đang bơi đi đâu mất rồi, không trả lời được... 🦦";
    } catch (e) {
        console.error(e);
        return "Úi! Mạng bị nghẽn rồi, Rái cá không nghe rõ. Bạn hỏi lại nhé? 🦦";
    }
};

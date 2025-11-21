
export enum AppMode {
  HOME = 'HOME',
  MCQ = 'MCQ', // Trắc nghiệm
  STATION = 'STATION', // Chạy trạm
}

export interface UserProfile {
  fullName: string;
  studentId: string;
  avatar?: string; // Base64 image string
}

export interface MCQQuestion {
  id: string;
  question: string;
  options: string[];
  correctAnswer: string; // Should match one of the options
  explanation: string;
  difficulty?: string; // Mức độ khó của câu hỏi
}

export interface StationItem {
  id: string;
  imageUri: string; // Base64 of the PDF page
  questions: StationQuestion[];
}

export interface StationQuestion {
  id: string;
  questionText: string; // e.g., "Chi tiết số 1 là gì?"
  correctAnswer: string;
  explanation?: string;
}

export enum Difficulty {
  REMEMBER = 'Ghi nhớ',
  UNDERSTAND = 'Hiểu',
  APPLY = 'Vận dụng thấp',
  CLINICAL = 'Lâm sàng',
}

// Gemini Response Schemas
export interface GeneratedMCQResponse {
  questions: {
    question: string;
    options: string[];
    correctAnswer: string;
    explanation: string;
    difficulty: string;
  }[];
}

export interface GeneratedStationResponse {
  questions: {
    questionText: string;
    correctAnswer: string;
    explanation: string;
  }[];
}

// Otter Mentor Response
export interface MentorResponse {
    analysis: string; // Lời nhận xét của Rái cá
    strengths: string[]; // Điểm mạnh
    weaknesses: string[]; // Điểm yếu
    roadmap: {
        step: string;
        details: string;
    }[]; // Các bước cải thiện
}

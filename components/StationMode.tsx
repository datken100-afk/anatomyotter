
import React, { useState, useRef, useEffect } from 'react';
import { generateStationQuestionFromImage, analyzeResultWithOtter } from '../services/geminiService';
import { StationItem, MentorResponse } from '../types';
import { Play, Timer, ArrowRight, CheckCircle, Eye, EyeOff, Activity, FileText, Crosshair, Database, Sparkles, FileUp, Loader2, ZoomIn, ZoomOut, RotateCcw, Check, X, ThumbsUp, ShieldAlert, AlertCircle, Lightbulb, List, Search, Book, ChevronLeft, Edit3 } from 'lucide-react';

// Declare pdfjsLib globally
declare const pdfjsLib: any;

interface StationModeProps {
  onBack: () => void;
}

enum StationStep {
  SETUP,
  RUNNING,
  SUMMARY
}

// --- GRAY'S ANATOMY FLASHCARDS MAPPING LOGIC ---
// Formula: PDF_Page (y) = 2 * Card_Number (x) + Offset (b)
interface SectionMap {
    id: string;
    name: string;
    range: [number, number]; // [Start Card, End Card]
    b: number; // Offset constant
    keywords: string[];
}

const GRAYS_SECTIONS: SectionMap[] = [
    { id: 'overview', name: '1. Overview (Tổng quan)', range: [1, 8], b: 12, keywords: ['overview', 'tổng quan', 'nhập môn'] },
    { id: 'back', name: '2. Back (Lưng & Cột sống)', range: [9, 34], b: 14, keywords: ['back', 'lưng', 'cột sống', 'đốt sống', 'vertebra', 'spine'] },
    { id: 'thorax', name: '3. Thorax (Ngực)', range: [35, 74], b: 16, keywords: ['thorax', 'ngực', 'tim', 'phổi', 'trung thất', 'heart', 'lung', 'mediastinum'] },
    { id: 'abdomen', name: '4. Abdomen (Bụng)', range: [75, 113], b: 18, keywords: ['abdomen', 'bụng', 'dạ dày', 'gan', 'ruột', 'thận', 'stomach', 'liver', 'kidney'] },
    { id: 'pelvis', name: '5. Pelvis (Chậu hông)', range: [114, 133], b: 20, keywords: ['pelvis', 'chậu', 'sinh dục', 'tiết niệu', 'perineum', 'đáy chậu'] },
    { id: 'lower', name: '6. Lower Limb (Chi dưới)', range: [134, 191], b: 22, keywords: ['lower', 'chi dưới', 'chân', 'đùi', 'cẳng chân', 'bàn chân', 'leg', 'foot', 'femur'] },
    { id: 'upper', name: '7. Upper Limb (Chi trên)', range: [192, 258], b: 24, keywords: ['upper', 'chi trên', 'tay', 'cánh tay', 'vai', 'arm', 'hand', 'shoulder'] },
    { id: 'head', name: '8. Head & Neck (Đầu Mặt Cổ)', range: [259, 349], b: 28, keywords: ['head', 'neck', 'đầu', 'mặt', 'cổ', 'sọ', 'thần kinh sọ', 'cranial', 'skull', 'face'] },
    { id: 'surface', name: '9. Surface Anatomy (Bề mặt)', range: [350, 369], b: 30, keywords: ['surface', 'bề mặt'] },
    { id: 'nervous', name: '10. Nervous System (Thần kinh)', range: [370, 377], b: 32, keywords: ['nervous', 'thần kinh', 'não', 'tủy', 'brain', 'spinal'] },
    { id: 'imaging', name: '11. Imaging (Hình ảnh học)', range: [378, 391], b: 34, keywords: ['imaging', 'hình ảnh', 'x-quang', 'ct', 'mri', 'radiograph'] },
];

// Helper to format text
const formatText = (text: string) => {
  if (!text) return "";
  return text.replace(/->/g, ' → ').replace(/=>/g, ' ⇒ ').replace(/<-/g, ' ← ');
};

// --- STRICT GRADING LOGIC ---
const normalizeString = (str: string) => {
    return str.trim().toLowerCase()
        .replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]/g, "") // Remove punctuation
        .replace(/\s{2,}/g, " "); // Remove extra spaces
};

const checkAnswer = (userAns: string, correctAns: string): boolean => {
    const u = normalizeString(userAns);
    const c = normalizeString(correctAns);

    if (!u) return false; // Empty answer is wrong

    // 1. Exact Match
    if (u === c) return true;

    // 2. Safety check for very short answers (prevents "s" matching "spinal cord")
    if (u.length < 3) return false; 

    // 3. Fuzzy Logic:
    if (c.includes(u)) {
        return u.length >= c.length * 0.5;
    }
    
    if (u.includes(c)) {
        return true;
    }

    return false;
};

export const StationMode: React.FC<StationModeProps> = ({ onBack }) => {
  const [step, setStep] = useState<StationStep>(StationStep.SETUP);
  const [stations, setStations] = useState<StationItem[]>([]);
  const [userAnswers, setUserAnswers] = useState<Record<string, string>>({});
  
  // Configuration State
  const [selectedSection, setSelectedSection] = useState<SectionMap | null>(null);
  const [topic, setTopic] = useState(''); // Now represents the SPECIFIC topic input by user
  const [questionCount, setQuestionCount] = useState(5); // 1-10
  const [timePerQuestion, setTimePerQuestion] = useState(30); // 10-120s
  const [pdfFile, setPdfFile] = useState<File | null>(null);
  
  const [loading, setLoading] = useState(false);
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [loadingText, setLoadingText] = useState('');
  const [error, setError] = useState<string | null>(null);

  // Mentor State
  const [mentorLoading, setMentorLoading] = useState(false);
  const [mentorData, setMentorData] = useState<MentorResponse | null>(null);
  const [showMentor, setShowMentor] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const mentorSectionRef = useRef<HTMLDivElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
          if (file.type !== 'application/pdf') {
              setError("Vui lòng chọn file PDF.");
              return;
          }
          setPdfFile(file);
          setError(null);
      }
  };

  // Core Logic: Mathematical Mapping
  const handleGenerate = async () => {
    if (!pdfFile) {
        setError("Vui lòng tải lên file Flashcard PDF.");
        return;
    }
    if (!selectedSection) {
        setError("Vui lòng chọn chương sách.");
        return;
    }
    // If topic is empty, default to the section name for AI context
    const processingTopic = topic.trim() || selectedSection.name;

    setLoading(true);
    setLoadingProgress(0);
    setError(null);
    setStations([]);
    setMentorData(null);
    setShowMentor(false);

    try {
        // 1. Identify Target Pages based on Selected Section
        setLoadingText(`Đang truy xuất dữ liệu chương: ${selectedSection.name}`);
        setLoadingProgress(10);

        const [startCard, endCard] = selectedSection.range;
        const b = selectedSection.b;
        
        // Generate random unique card numbers within the range
        const potentialCards = Array.from({length: endCard - startCard + 1}, (_, i) => startCard + i);
        
        // Shuffle and pick 'questionCount' cards (or fewer if range is small)
        for (let i = potentialCards.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [potentialCards[i], potentialCards[j]] = [potentialCards[j], potentialCards[i]];
        }
        
        const selectedCards = potentialCards.slice(0, Math.min(questionCount * 2, potentialCards.length)); // Pick double just in case some fail

        // Convert Card Number (x) to PDF Page (y) using y = 2x + b
        let targetPageNums = selectedCards.map(x => (2 * x) + b);
        
        console.log(`Section: ${selectedSection.name}. Context: ${processingTopic}. Selected Cards: ${selectedCards.join(', ')}. PDF Pages: ${targetPageNums.join(', ')}`);

        // 2. Load PDF Document
        setLoadingText("Đang mở tài liệu...");
        const arrayBuffer = await pdfFile.arrayBuffer();
        const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
        const totalPages = pdf.numPages;

        // Filter out pages that are out of bounds
        targetPageNums = targetPageNums.filter(p => p <= totalPages && p > 0);

        if (targetPageNums.length === 0) {
             throw new Error("Lỗi tính toán trang PDF. Vui lòng kiểm tra lại file.");
        }

        // 3. Extract Images & Generate Questions
        const newStations: StationItem[] = [];
        setLoadingText(`Đang trích xuất hình ảnh từ ${targetPageNums.length} vị trí...`);
        setLoadingProgress(30);
        
        let hasQuotaError = false;

        // Process one by one to maintain order and check quota
        for (let i = 0; i < targetPageNums.length; i++) {
            if (newStations.length >= questionCount) break;
            if (hasQuotaError) break;

            const pageNum = targetPageNums[i];
            
            try {
                setLoadingText(`Đang phân tích trạm ${newStations.length + 1}... (Trang PDF ${pageNum})`);
                
                const page = await pdf.getPage(pageNum);
                const viewport = page.getViewport({ scale: 1.5 }); // High quality for small details
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                canvas.height = viewport.height;
                canvas.width = viewport.width;
                
                if (!ctx) continue;
                
                await page.render({ canvasContext: ctx, viewport }).promise;
                const base64 = canvas.toDataURL('image/jpeg', 0.7); 

                // Call AI to generate question with the specific topic context
                const res = await generateStationQuestionFromImage(base64, processingTopic);
                
                if (res.isValid && res.questions && res.questions.length > 0) {
                    newStations.push({
                        id: `st-${Date.now()}-${i}`,
                        imageUri: base64,
                        questions: res.questions.map((q: any) => ({
                            ...q, 
                            id: `q-${Date.now()}-${i}`
                        }))
                    });
                    
                    // Update progress
                    const progress = 30 + Math.round((newStations.length / questionCount) * 70);
                    setLoadingProgress(Math.min(progress, 99));
                } else {
                    console.warn(`Page ${pageNum} rejected by AI (Not anatomical or unclear).`);
                }
            } catch (e: any) {
                if (e.message && (e.message.includes("quota") || e.message.includes("429") || e.message.includes("hết hạn mức"))) {
                    hasQuotaError = true;
                    console.error("Quota Exceeded during loop");
                } else {
                    console.warn(`Error processing page ${pageNum}:`, e);
                }
            }
        }
        
        if (newStations.length > 0) {
             if (hasQuotaError && newStations.length < questionCount) {
                 setError("Đã đạt giới hạn AI (Quota). Đang hiển thị các trạm đã tạo được.");
             }
             
             setLoadingProgress(100);
             setLoadingText("Hoàn tất! Chuẩn bị vào trạm...");
             
             setTimeout(() => {
                setStations(newStations);
                setUserAnswers({});
                setStep(StationStep.RUNNING);
                setLoading(false);
             }, 800);
        } else {
            if (hasQuotaError) {
                 throw new Error("Đã hết hạn mức sử dụng AI (Quota Exceeded). Vui lòng thử lại sau.");
            } else {
                 throw new Error("Không thể tạo trạm nào từ các trang đã chọn. Hình ảnh có thể không rõ ràng.");
            }
        }

    } catch (err: any) {
        console.error("Station generation error", err);
        setError(err.message || "Có lỗi xảy ra khi xử lý.");
        setLoading(false);
    }
  };

  // Mentor Logic for Station Mode
  const handleConsultMentor = async () => {
    if (mentorData) {
        setShowMentor(true);
        mentorSectionRef.current?.scrollIntoView({ behavior: 'smooth' });
        return;
    }

    setMentorLoading(true);
    setShowMentor(true);

    // Calculate stats for Station Mode using Strict Grading
    const stats: Record<string, { correct: number, total: number }> = {
        "Thực hành chạy trạm": { correct: 0, total: 0 }
    };

    stations.forEach(s => {
        stats["Thực hành chạy trạm"].total++;
        const userAns = userAnswers[s.id] || "";
        const correctAns = s.questions[0].correctAnswer;
        
        if (checkAnswer(userAns, correctAns)) {
            stats["Thực hành chạy trạm"].correct++;
        }
    });

    try {
        const response = await analyzeResultWithOtter(topic || selectedSection?.name || "Giải phẫu học", stats);
        setMentorData(response);
    } catch (e) {
        console.error("Mentor Error", e);
    } finally {
        setMentorLoading(false);
        setTimeout(() => {
            mentorSectionRef.current?.scrollIntoView({ behavior: 'smooth' });
        }, 200);
    }
  };


  if (loading) {
    return (
        <div className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-slate-50/95 dark:bg-slate-950/95 backdrop-blur-md transition-all duration-500">
            <div className="w-full max-w-2xl p-8 relative">
                {/* Title */}
                <h3 className="text-3xl font-black text-center text-transparent bg-clip-text bg-gradient-to-r from-teal-500 to-emerald-600 mb-16 animate-pulse">
                    RÁI CÁ ĐANG CHẠY TRẠM...
                </h3>

                {/* THE TRACK */}
                <div className="relative w-full h-4 bg-slate-200 dark:bg-slate-800 rounded-full overflow-visible">
                    
                    {/* Progress Fill */}
                    <div 
                        className="absolute top-0 left-0 h-full bg-gradient-to-r from-teal-400 to-emerald-500 rounded-full transition-all duration-500 ease-out shadow-[0_0_20px_rgba(20,184,166,0.5)]"
                        style={{ width: `${loadingProgress}%` }}
                    >
                        {/* Shimmer Effect inside bar */}
                        <div className="absolute inset-0 w-full h-full bg-white/30 animate-[shimmer_1s_infinite]"></div>
                    </div>

                    {/* THE RUNNING OTTER */}
                    <div 
                        className="absolute top-1/2 -translate-y-1/2 transition-all duration-500 ease-out z-20"
                        style={{ left: `${loadingProgress}%`, transform: 'translate(-50%, -50%)' }}
                    >
                        <div className="relative">
                            {/* The Emoji - Flipped horizontally to run Right */}
                            <div className="text-7xl transform -scale-x-100 animate-[bounce_0.4s_infinite] filter drop-shadow-lg">
                                🦦
                            </div>
                            
                            {/* Dust/Wind Effect behind */}
                            <div className="absolute bottom-1 right-full mr-2 text-2xl opacity-0 animate-[fade-left_0.6s_infinite]">
                                💨
                            </div>
                            
                            {/* Sweat Drop */}
                            <div className="absolute -top-2 left-full text-xl animate-bounce">
                                💦
                            </div>
                        </div>
                        
                        {/* Percentage Tooltip */}
                        <div className="absolute -top-12 left-1/2 -translate-x-1/2 bg-slate-900 text-white text-xs font-bold px-2 py-1 rounded-md whitespace-nowrap opacity-80">
                            {loadingProgress}%
                        </div>
                    </div>
                    
                    {/* Finish Line Flag */}
                    <div className="absolute right-0 top-1/2 -translate-y-1/2 translate-x-4 text-2xl">
                        🏁
                    </div>
                </div>

                {/* Status Text */}
                <div className="mt-20 text-center space-y-3">
                    <p className="text-xl font-bold text-slate-700 dark:text-slate-200 animate-fade-up" key={loadingText}>
                        {loadingText}
                    </p>
                    <div className="flex justify-center gap-4 text-sm text-slate-400 italic">
                        <span className="flex items-center gap-1"><Book className="w-3 h-3" /> Gray's Anatomy</span>
                        <span className="flex items-center gap-1"><Search className="w-3 h-3" /> Smart Map</span>
                        <span className="flex items-center gap-1"><Eye className="w-3 h-3" /> Vision AI</span>
                    </div>
                </div>
            </div>
        </div>
    );
  }

  if (step === StationStep.SETUP) {
    return (
      <div className="max-w-4xl mx-auto pb-20 px-4">
        {/* Back Button */}
        <div className="flex items-center mb-6">
            <button onClick={onBack} className="mr-4 p-2 rounded-full hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-500 transition-colors">
                <ArrowRight className="w-6 h-6 rotate-180" />
            </button>
            <h2 className="text-xl font-medium text-slate-500 dark:text-slate-400">Quay lại</h2>
        </div>

        {/* GRID LAYOUT */}
        <div className="space-y-8">

            {/* CARD 1: HEADER */}
            <div className="relative overflow-hidden rounded-3xl bg-gradient-to-r from-teal-600 to-emerald-600 p-8 text-white shadow-xl animate-fade-up" style={{ animationDelay: '0ms' }}>
                <div className="absolute top-0 right-0 w-64 h-64 bg-white/10 rounded-full -mr-16 -mt-16 blur-3xl"></div>
                <div className="relative z-10 flex items-center gap-6">
                    <div className="w-20 h-20 bg-white/20 backdrop-blur-md rounded-2xl flex items-center justify-center border border-white/30 shadow-inner">
                        <Activity className="w-10 h-10 text-white drop-shadow-lg" />
                    </div>
                    <div>
                        <h1 className="text-3xl md:text-4xl font-bold mb-2 text-glow-white">Chạy Trạm (Spot Test)</h1>
                        <p className="text-teal-100 text-lg">Hệ thống sử dụng thuật toán ánh xạ trực tiếp vào file PDF Gray's Anatomy.</p>
                    </div>
                </div>
            </div>

            {/* CARD 2: INPUT & SOURCE */}
            <div className="relative rounded-3xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-8 shadow-lg animate-fade-up" style={{ animationDelay: '100ms' }}>
                <div className="grid md:grid-cols-2 gap-8">
                    
                    {/* LEFT: UPLOAD SOURCE */}
                    <div>
                        <h3 className="text-lg font-bold text-slate-800 dark:text-white mb-4 flex items-center gap-2">
                            <Database className="w-5 h-5 text-teal-500" />
                            Nguồn dữ liệu gốc (PDF)
                        </h3>
                        
                        <div 
                            onClick={() => fileInputRef.current?.click()}
                            className={`border-2 border-dashed rounded-2xl p-6 flex flex-col items-center justify-center text-center cursor-pointer transition-all duration-300 group
                            ${pdfFile ? 'border-teal-500 bg-teal-50 dark:bg-teal-900/20' : 'border-slate-300 dark:border-slate-700 hover:border-teal-400 hover:bg-slate-50 dark:hover:bg-slate-800'}`}
                        >
                            <input 
                                type="file" 
                                ref={fileInputRef} 
                                onChange={handleFileChange} 
                                accept="application/pdf" 
                                className="hidden" 
                            />
                            
                            {pdfFile ? (
                                <div className="flex flex-col items-center gap-2 animate-in zoom-in">
                                    <div className="w-12 h-12 bg-teal-100 dark:bg-teal-800 rounded-full flex items-center justify-center text-teal-600 dark:text-teal-300">
                                        <CheckCircle className="w-6 h-6" />
                                    </div>
                                    <p className="font-bold text-slate-700 dark:text-slate-200 text-sm break-all px-2">{pdfFile.name}</p>
                                    <span className="text-xs text-teal-600 dark:text-teal-400">Nhấn để thay đổi file</span>
                                </div>
                            ) : (
                                <div className="flex flex-col items-center gap-3 text-slate-400 group-hover:text-slate-600 dark:group-hover:text-slate-300">
                                    <div className="w-12 h-12 bg-slate-100 dark:bg-slate-800 rounded-full flex items-center justify-center">
                                        <FileUp className="w-6 h-6" />
                                    </div>
                                    <div>
                                        <p className="font-bold text-sm">Tải lên Flashcard PDF</p>
                                        <p className="text-xs opacity-70">Gray's Anatomy (3rd Ed)</p>
                                    </div>
                                </div>
                            )}
                        </div>
                        
                    </div>

                     {/* RIGHT: TOPIC SELECTION (TWO STEP) */}
                     <div>
                        <h3 className="text-lg font-bold text-slate-800 dark:text-white mb-4 flex items-center gap-2">
                            <FileText className="w-5 h-5 text-blue-500" />
                            Chủ đề (Chọn chương sách)
                        </h3>
                        
                        {!selectedSection ? (
                            // STEP 1: SELECT CHAPTER
                            <div className="space-y-3 max-h-60 overflow-y-auto pr-2 custom-scrollbar">
                                {GRAYS_SECTIONS.map((sec) => (
                                    <button
                                        key={sec.id}
                                        onClick={() => {
                                            setSelectedSection(sec);
                                            setTopic(''); // Reset specific topic for user input
                                        }} 
                                        className="w-full text-left px-4 py-3 rounded-xl border border-slate-200 dark:border-slate-700 text-sm font-medium text-slate-700 dark:text-slate-300 hover:border-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 transition-all"
                                    >
                                        {sec.name}
                                    </button>
                                ))}
                            </div>
                        ) : (
                            // STEP 2: INPUT SPECIFIC TOPIC
                            <div className="animate-in fade-in slide-in-from-right-4">
                                <div className="bg-blue-50 dark:bg-blue-900/10 border border-blue-200 dark:border-blue-800 rounded-xl p-4 mb-4">
                                    <div className="flex justify-between items-center mb-2">
                                        <span className="text-xs font-bold text-blue-600 dark:text-blue-400 uppercase tracking-wider">Đang chọn chương</span>
                                        <button 
                                            onClick={() => setSelectedSection(null)}
                                            className="text-xs text-slate-400 hover:text-blue-500 flex items-center gap-1"
                                        >
                                            <RotateCcw className="w-3 h-3" /> Chọn lại
                                        </button>
                                    </div>
                                    <p className="font-bold text-slate-800 dark:text-white text-lg">{selectedSection.name}</p>
                                </div>

                                <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                                    Chủ đề chính (để AI tạo câu hỏi):
                                </label>
                                <div className="relative">
                                    <input
                                        type="text"
                                        value={topic}
                                        onChange={(e) => setTopic(e.target.value)}
                                        placeholder={`Ví dụ: ${selectedSection.keywords.slice(0, 3).join(', ')}...`}
                                        className="w-full p-3 pl-10 bg-slate-50 dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all text-slate-900 dark:text-white"
                                    />
                                    <Edit3 className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
                                </div>
                                <p className="text-xs text-slate-500 dark:text-slate-400 mt-2 italic">
                                    * Nhập cụ thể (VD: "Van tim", "Cơ vùng đùi") để câu hỏi tập trung hơn, hoặc để trống để AI tự chọn trong chương.
                                </p>
                            </div>
                        )}
                    </div>
                </div>

                {error && (
                    <div className="mt-6 p-4 bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 rounded-xl flex items-center gap-2 text-sm animate-pulse border border-red-200 dark:border-red-800">
                        <AlertCircle className="w-5 h-5 flex-shrink-0" />
                        {error}
                    </div>
                )}
            </div>

            {/* CARD 3: SETTINGS */}
            <div className="relative rounded-3xl bg-gradient-to-br from-midnight-950 to-midnight-900 border border-slate-700 p-8 shadow-2xl animate-fade-up" style={{ animationDelay: '200ms' }}>
                <div className="absolute top-0 right-0 w-full h-1 bg-gradient-to-r from-transparent via-teal-500 to-transparent opacity-50"></div>
                
                <div className="grid md:grid-cols-2 gap-12 mb-10">
                    {/* Question Count Slider */}
                    <div className="relative">
                        <div className="flex justify-between mb-4">
                            <label className="text-sm font-bold text-slate-300">Số lượng trạm (1-10)</label>
                            <span className="text-2xl font-bold text-teal-400">{questionCount}</span>
                        </div>
                        <div className="relative h-10 flex items-center">
                            <input
                                type="range"
                                min="1"
                                max="10"
                                value={questionCount}
                                onChange={(e) => setQuestionCount(Number(e.target.value))}
                                className="liquid-slider w-full"
                                style={{ '--range-progress': `${((questionCount - 1) / 9) * 100}%` } as React.CSSProperties}
                            />
                        </div>
                    </div>

                    {/* Time Slider */}
                    <div className="relative">
                        <div className="flex justify-between mb-4">
                            <label className="text-sm font-bold text-slate-300">Thời gian (giây/trạm)</label>
                            <span className="text-2xl font-bold text-purple-400">{timePerQuestion}s</span>
                        </div>
                        <div className="relative h-10 flex items-center">
                            <input
                                type="range"
                                min="10"
                                max="120"
                                step="5"
                                value={timePerQuestion}
                                onChange={(e) => setTimePerQuestion(Number(e.target.value))}
                                className="liquid-slider w-full"
                                style={{ '--range-progress': `${((timePerQuestion - 10) / 110) * 100}%` } as React.CSSProperties}
                            />
                        </div>
                    </div>
                </div>

                {/* Start Button */}
                <button
                    onClick={handleGenerate}
                    disabled={!selectedSection || !pdfFile}
                    className="w-full bg-gradient-to-r from-teal-600 to-emerald-600 hover:from-teal-500 hover:to-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-bold py-5 rounded-2xl shadow-[0_0_20px_rgba(20,184,166,0.3)] hover:shadow-[0_0_30px_rgba(20,184,166,0.5)] transition-all flex items-center justify-center space-x-3 text-lg active:scale-95 relative overflow-hidden group"
                >
                    <Play className="w-6 h-6 fill-current" />
                    <span>Bắt đầu thi Chạy Trạm</span>
                </button>
            </div>

        </div>
      </div>
    );
  }

  if (step === StationStep.RUNNING) {
    return (
        <StationRunner 
            stations={stations} 
            timePerStation={timePerQuestion} 
            userAnswers={userAnswers}
            setUserAnswers={setUserAnswers}
            onFinish={() => setStep(StationStep.SUMMARY)} 
        />
    );
  }

  if (step === StationStep.SUMMARY) {
    return (
        <StationSummary 
            stations={stations} 
            userAnswers={userAnswers}
            onRestart={() => setStep(StationStep.SETUP)} 
            handleConsultMentor={handleConsultMentor}
            mentorLoading={mentorLoading}
            mentorData={mentorData}
            showMentor={showMentor}
            mentorSectionRef={mentorSectionRef}
        />
    );
  }

  return null;
};

// ---------------------------------------------------------------------------
// Sub-component: Station Runner (Visual Layout)
// ---------------------------------------------------------------------------
interface StationRunnerProps {
    stations: StationItem[];
    timePerStation: number;
    userAnswers: Record<string, string>;
    setUserAnswers: React.Dispatch<React.SetStateAction<Record<string, string>>>;
    onFinish: () => void;
}

const StationRunner: React.FC<StationRunnerProps> = ({ stations, timePerStation, userAnswers, setUserAnswers, onFinish }) => {
    const [currentIndex, setCurrentIndex] = useState(0);
    const [timeLeft, setTimeLeft] = useState(timePerStation);
    const [isPaused, setIsPaused] = useState(false);
    const [zoomLevel, setZoomLevel] = useState(1);
    
    // Reset timer & zoom when station changes
    useEffect(() => {
        setTimeLeft(timePerStation);
        setZoomLevel(1);
    }, [currentIndex, timePerStation]);

    useEffect(() => {
        if (isPaused) return;
        if (timeLeft <= 0) {
            handleNext();
            return;
        }
        const timer = setInterval(() => setTimeLeft(prev => prev - 1), 1000);
        return () => clearInterval(timer);
    }, [timeLeft, isPaused]);

    const handleNext = () => {
        if (currentIndex >= stations.length - 1) {
            onFinish();
        } else {
            setCurrentIndex(prev => prev + 1);
        }
    };

    const handleZoomIn = () => setZoomLevel(prev => Math.min(prev + 0.5, 3));
    const handleZoomOut = () => setZoomLevel(prev => Math.max(prev - 0.5, 1));
    const handleResetZoom = () => setZoomLevel(1);

    const handleAnswerChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
        const currentStationId = stations[currentIndex].id;
        setUserAnswers(prev => ({
            ...prev,
            [currentStationId]: e.target.value
        }));
    };

    const currentStation = stations[currentIndex];
    const currentQuestion = currentStation?.questions[0]; 
    const currentAnswer = userAnswers[currentStation.id] || "";

    if (!currentStation) return null; 

    const progress = ((timePerStation - timeLeft) / timePerStation) * 100;

    return (
        <div className="fixed inset-0 bg-slate-950 text-white z-50 flex flex-col">
            {/* Top Bar: Timer & Progress */}
            <div className="h-2 bg-slate-800 w-full">
                <div 
                    className="h-full bg-teal-500 transition-all duration-1000 ease-linear shadow-[0_0_10px_rgba(20,184,166,0.8)]"
                    style={{ width: `${100 - progress}%` }}
                ></div>
            </div>
            
            <div className="flex items-center justify-between px-4 md:px-8 py-4 bg-slate-900 border-b border-slate-800">
                <div className="flex items-center gap-4">
                    <div className="bg-teal-500/20 text-teal-400 px-3 py-1 rounded-lg font-bold border border-teal-500/30 text-sm md:text-base">
                        TRẠM {currentIndex + 1}
                    </div>
                    <span className="text-slate-400 text-sm">/ {stations.length}</span>
                </div>
                
                <div className={`text-4xl md:text-5xl font-mono font-bold tracking-widest ${timeLeft < 10 ? 'text-red-500 animate-pulse scale-110' : 'text-white'} transition-all`}>
                    {timeLeft}
                </div>

                <button 
                    onClick={() => setIsPaused(!isPaused)}
                    className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition-colors text-sm"
                >
                    {isPaused ? "Tiếp tục" : "Tạm dừng"}
                </button>
            </div>

            {/* Main Content: Split View (Image Left / Input Right) */}
            <div className="flex-1 flex flex-col md:flex-row overflow-hidden">
                
                {/* Image Area - Interactive Zoom - LEFT SIDE */}
                <div className="flex-1 bg-black/80 relative overflow-hidden flex items-center justify-center group select-none order-1">
                    {/* Zoom Toolbar */}
                    <div className="absolute top-6 left-1/2 -translate-x-1/2 z-30 flex gap-2 bg-slate-800/80 backdrop-blur-md p-1.5 rounded-xl border border-slate-700 shadow-lg opacity-0 group-hover:opacity-100 transition-opacity">
                        <button onClick={handleZoomOut} className="p-2 hover:bg-slate-700 rounded-lg text-white"><ZoomOut className="w-5 h-5" /></button>
                        <button onClick={handleResetZoom} className="p-2 hover:bg-slate-700 rounded-lg text-white font-mono text-xs min-w-[3ch]">{Math.round(zoomLevel * 100)}%</button>
                        <button onClick={handleZoomIn} className="p-2 hover:bg-slate-700 rounded-lg text-white"><ZoomIn className="w-5 h-5" /></button>
                        <div className="w-px bg-slate-600 mx-1"></div>
                        <button onClick={handleResetZoom} className="p-2 hover:bg-slate-700 rounded-lg text-white"><RotateCcw className="w-5 h-5" /></button>
                    </div>

                    <div 
                        className="w-full h-full flex items-center justify-center overflow-auto p-4"
                        style={{ cursor: zoomLevel > 1 ? 'grab' : 'default' }}
                    >
                         <div 
                            style={{ transform: `scale(${zoomLevel})`, transition: 'transform 0.2s ease-out' }}
                            className="origin-center w-full h-full flex items-center justify-center"
                         >
                             <img 
                                src={currentStation.imageUri} 
                                alt="Anatomy Spot" 
                                className="max-h-full max-w-full object-contain shadow-2xl rounded-lg"
                                draggable={false}
                             />
                         </div>
                    </div>
                </div>

                {/* Question Area - RIGHT SIDE */}
                <div className="md:w-[420px] w-full bg-slate-900 border-l border-slate-800 p-6 flex flex-col justify-between shadow-2xl relative z-10 order-2">
                     <div className="space-y-6">
                         <div className="flex items-center gap-2 text-teal-400 font-bold uppercase tracking-wider text-xs">
                             <Crosshair className="w-4 h-4" /> Câu hỏi định danh
                         </div>
                         
                         <h2 className="text-2xl font-bold leading-snug text-white">
                             {formatText(currentQuestion?.questionText)}
                         </h2>
                         
                         <div className="p-3 bg-slate-800/50 rounded-xl border border-slate-700 text-slate-400 italic text-xs">
                             Lưu ý: Quan sát kỹ hình ảnh bên trái. Có thể phóng to (Zoom) để xem rõ chi tiết.
                         </div>
                     </div>

                     <div className="mt-8 md:mt-0 space-y-4 flex-1 flex flex-col justify-end">
                         <div className="space-y-2">
                             <label className="text-sm font-bold text-slate-300">Câu trả lời của bạn:</label>
                             <textarea
                                value={currentAnswer}
                                onChange={handleAnswerChange}
                                placeholder="Nhập tên cấu trúc giải phẫu..."
                                className="w-full h-32 bg-slate-800 border border-slate-700 rounded-xl p-4 text-white focus:border-teal-500 focus:ring-1 focus:ring-teal-500 outline-none resize-none text-lg placeholder-slate-600"
                             />
                         </div>

                         <button 
                            onClick={handleNext}
                            className="w-full bg-teal-600 hover:bg-teal-500 text-white px-6 py-4 rounded-2xl font-bold flex items-center justify-center gap-3 shadow-lg hover:shadow-teal-500/30 transition-all text-lg mt-4"
                         >
                            <span>{currentIndex === stations.length - 1 ? "Nộp bài" : "Tiếp theo"}</span>
                            <ArrowRight className="w-6 h-6" />
                         </button>
                     </div>
                </div>
            </div>
        </div>
    );
};

// ---------------------------------------------------------------------------
// Sub-component: Station Summary (Shows Score & Comparison)
// ---------------------------------------------------------------------------
interface StationSummaryProps {
    stations: StationItem[];
    userAnswers: Record<string, string>;
    onRestart: () => void;
    handleConsultMentor: () => void;
    mentorLoading: boolean;
    mentorData: MentorResponse | null;
    showMentor: boolean;
    mentorSectionRef: React.RefObject<HTMLDivElement>;
}

const StationSummary: React.FC<StationSummaryProps> = ({ 
    stations, userAnswers, onRestart, handleConsultMentor, mentorLoading, mentorData, showMentor, mentorSectionRef 
}) => {
    // STRICT SCORING
    let correctCount = 0;
    const gradedStations = stations.map(s => {
        const userAns = userAnswers[s.id] || "";
        const correctAns = s.questions[0].correctAnswer;
        
        const isCorrect = checkAnswer(userAns, correctAns);
        
        if (isCorrect) correctCount++;
        
        return { ...s, isCorrect };
    });

    const scorePercentage = Math.round((correctCount / stations.length) * 100);
    // Piggy math for circle chart
    const radius = 28;
    const circumference = 2 * Math.PI * radius;
    const dashOffset = circumference - (scorePercentage / 100) * circumference;
    const pigAngle = (scorePercentage / 100) * 2 * Math.PI;
    const pigX = 35 + 28 * Math.cos(pigAngle);
    const pigY = 35 + 28 * Math.sin(pigAngle);

    return (
        <div className="max-w-4xl mx-auto pb-20 px-4">
            
            {/* Score Header */}
            <div className="bg-white dark:bg-slate-900 rounded-3xl p-8 shadow-xl border border-slate-200 dark:border-slate-700 mb-10 flex flex-col md:flex-row items-center gap-8 animate-fade-up">
                 {/* Circle Chart */}
                 <div className="relative w-32 h-32 flex-shrink-0 group">
                        <svg className="w-full h-full transform -rotate-90" viewBox="0 0 70 70">
                        <circle cx="35" cy="35" r={radius} fill="none" stroke="currentColor" className="text-slate-100 dark:text-slate-800" strokeWidth="6" />
                        <circle cx="35" cy="35" r={radius} fill="none" stroke="#14b8a6" strokeWidth="6" strokeDasharray={circumference} strokeDashoffset={dashOffset} strokeLinecap="round" className="transition-all duration-1000 ease-out" />
                        <foreignObject x={pigX - 7} y={pigY - 7} width="14" height="14">
                            <div className="w-full h-full rounded-full bg-pink-100 border border-pink-300 flex items-center justify-center shadow-sm rotate-90">
                                <span className="text-[9px]">🐷</span>
                            </div>
                        </foreignObject>
                        </svg>
                        <div className="absolute inset-0 flex items-center justify-center flex-col">
                            <span className="text-3xl font-black text-teal-600 dark:text-teal-400">{scorePercentage}%</span>
                        </div>
                </div>
                
                <div className="text-center md:text-left flex-1">
                    <h2 className="text-3xl font-bold text-slate-900 dark:text-white mb-2">Kết quả Chạy Trạm</h2>
                    <p className="text-slate-500 dark:text-slate-400 mb-4">Heo con đã chấm bài xong! Bạn làm đúng <strong className="text-teal-600 dark:text-teal-400">{correctCount}/{stations.length}</strong> trạm.</p>
                    
                    <div className="flex gap-3 justify-center md:justify-start">
                        <button 
                            onClick={onRestart}
                            className="px-6 py-2.5 bg-slate-900 dark:bg-white text-white dark:text-slate-900 rounded-xl font-bold hover:opacity-90 transition-all shadow-md flex items-center gap-2"
                        >
                            <Activity className="w-4 h-4" />
                            Thi lại
                        </button>
                        
                         {/* Mentor Button */}
                         {!showMentor && (
                            <button 
                                onClick={handleConsultMentor}
                                disabled={mentorLoading}
                                className="px-6 py-2.5 bg-amber-100 dark:bg-amber-900/30 text-amber-800 dark:text-amber-300 border border-amber-200 dark:border-amber-800/50 hover:bg-amber-200 dark:hover:bg-amber-900/50 transition-all flex items-center gap-2 font-bold rounded-xl"
                            >
                                <span>🦦</span>
                                <span>Hỏi Rái cá</span>
                            </button>
                        )}
                    </div>
                </div>
            </div>

            {/* MENTOR SECTION */}
            {showMentor && (
                <div ref={mentorSectionRef} className="mb-12 animate-in slide-in-from-bottom-10 duration-700">
                    {mentorLoading ? (
                        <div className="w-full bg-white dark:bg-slate-900 rounded-[2rem] p-8 shadow-xl border border-amber-200 dark:border-amber-900/30 text-center flex flex-col items-center gap-4">
                                <div className="text-6xl animate-bounce">🦦</div>
                                <p className="text-slate-600 dark:text-slate-300 font-medium animate-pulse">Rái cá nhỏ đang xem bài làm của bạn...</p>
                        </div>
                    ) : mentorData ? (
                        <div className="relative bg-gradient-to-b from-amber-50 to-white dark:from-slate-900 dark:to-slate-900 rounded-[2.5rem] p-8 md:p-10 shadow-2xl border border-amber-200 dark:border-slate-700 overflow-hidden">
                                <div className="absolute top-0 right-0 w-64 h-64 bg-amber-300/20 rounded-full blur-3xl -mr-20 -mt-20"></div>
                                
                                <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 relative z-10">
                                    <div className="lg:col-span-5 flex flex-col gap-6">
                                        <div className="flex items-center gap-6">
                                            <div className="w-28 h-28 bg-gradient-to-br from-amber-100 to-white dark:from-slate-800 dark:to-slate-700 rounded-full flex items-center justify-center shadow-lg border-4 border-white dark:border-slate-600 relative flex-shrink-0">
                                                    <span className="text-6xl animate-[wiggle_3s_infinite]">🦦</span>
                                            </div>
                                            <div>
                                                <h3 className="text-xl font-bold text-slate-900 dark:text-white">Rái cá nhỏ</h3>
                                                <p className="text-xs text-amber-600 dark:text-amber-400 font-bold uppercase tracking-wide">Góc nhìn chuyên gia</p>
                                            </div>
                                        </div>
                                        <div className="bg-white dark:bg-slate-800 p-6 rounded-2xl rounded-tl-none shadow-sm border border-amber-100 dark:border-slate-700 relative">
                                            <p className="text-slate-700 dark:text-slate-300 leading-relaxed italic text-lg">"{mentorData.analysis}"</p>
                                        </div>
                                    </div>
                                    <div className="lg:col-span-7 flex flex-col gap-4">
                                        <div className="bg-green-50/80 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-2xl p-4">
                                            <div className="flex items-center gap-2 mb-2 text-green-700 dark:text-green-400 font-bold text-sm uppercase"><ThumbsUp className="w-4 h-4" /> Điểm mạnh</div>
                                            <ul className="space-y-1">{mentorData.strengths?.map((s, i) => <li key={i} className="text-xs text-slate-600 dark:text-slate-300">• {s}</li>)}</ul>
                                        </div>
                                        <div className="bg-red-50/80 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-2xl p-4">
                                            <div className="flex items-center gap-2 mb-2 text-red-700 dark:text-red-400 font-bold text-sm uppercase"><ShieldAlert className="w-4 h-4" /> Điểm yếu</div>
                                            <ul className="space-y-1">{mentorData.weaknesses?.map((w, i) => <li key={i} className="text-xs text-slate-600 dark:text-slate-300">• {w}</li>)}</ul>
                                        </div>
                                    </div>
                                </div>
                                <div className="mt-12 pt-8 border-t border-amber-200/50 dark:border-slate-700/50">
                                        <h4 className="text-center text-lg font-bold text-amber-600 dark:text-amber-400 uppercase tracking-widest mb-8 flex items-center justify-center gap-2"><Sparkles className="w-5 h-5" /> Lộ trình cải thiện</h4>
                                        <div className="space-y-6">{mentorData.roadmap?.map((step, idx) => (
                                            <div key={idx} className="bg-white dark:bg-slate-800 p-6 rounded-2xl shadow-sm border border-slate-100 dark:border-slate-700">
                                                <h5 className="text-lg font-bold text-medical-600 dark:text-medical-400 mb-2">Bước {idx+1}: {step.step}</h5>
                                                <p className="text-slate-600 dark:text-slate-300 text-sm">{step.details}</p>
                                            </div>
                                        ))}</div>
                                </div>
                        </div>
                    ) : null}
                </div>
            )}

            <div className="space-y-8">
                {gradedStations.map((station, idx) => {
                    const userAns = userAnswers[station.id];
                    
                    return (
                    <div 
                        key={station.id} 
                        className={`bg-white dark:bg-slate-800 rounded-3xl shadow-sm border overflow-hidden transition-all hover:shadow-md
                        ${station.isCorrect ? 'border-green-200 dark:border-green-900/30' : 'border-red-200 dark:border-red-900/30'}`}
                    >
                        {/* Image Header */}
                        <div className="w-full h-64 bg-slate-950 flex items-center justify-center relative overflow-hidden group">
                             <img src={station.imageUri} alt={`Station ${idx+1}`} className="h-full w-auto object-contain" />
                             <div className={`absolute top-4 left-4 text-white text-xs font-bold px-3 py-1 rounded-full shadow-lg flex items-center gap-1 ${station.isCorrect ? 'bg-green-500' : 'bg-red-500'}`}>
                                 {station.isCorrect ? <Check className="w-3 h-3" /> : <X className="w-3 h-3" />}
                                 Trạm {idx + 1}
                             </div>
                        </div>

                        <div className="p-6">
                            <div className="mb-6">
                                <h3 className="text-lg font-bold text-slate-900 dark:text-white">
                                    {formatText(station.questions[0].questionText)}
                                </h3>
                            </div>

                            {/* Comparison Grid */}
                            <div className="grid md:grid-cols-2 gap-4 mb-6">
                                <div className={`p-4 rounded-2xl border ${station.isCorrect ? 'bg-green-50 dark:bg-green-900/10 border-green-100 dark:border-green-900/30' : 'bg-red-50 dark:bg-red-900/10 border-red-100 dark:border-red-900/30'}`}>
                                    <label className="text-xs font-bold uppercase tracking-wider mb-1 block opacity-70">Đáp án của bạn</label>
                                    <p className={`font-medium text-lg ${station.isCorrect ? 'text-green-700 dark:text-green-400' : 'text-red-700 dark:text-red-400'}`}>
                                        {userAns || "(Bỏ trống)"}
                                    </p>
                                </div>
                                
                                <div className="p-4 rounded-2xl bg-blue-50 dark:bg-blue-900/10 border border-blue-100 dark:border-blue-900/30">
                                    <label className="text-xs font-bold text-blue-600 dark:text-blue-400 uppercase tracking-wider mb-1 block opacity-70">Đáp án đúng</label>
                                    <p className="font-medium text-lg text-blue-800 dark:text-blue-300">
                                        {formatText(station.questions[0].correctAnswer)}
                                    </p>
                                </div>
                            </div>

                            {/* Explanation - HIGHLIGHTED */}
                            <div className="bg-yellow-50 dark:bg-yellow-900/10 border-l-4 border-yellow-400 dark:border-yellow-600 p-5 rounded-r-xl shadow-sm">
                                <div className="flex items-center gap-2 mb-2">
                                    <Lightbulb className="w-5 h-5 text-yellow-600 dark:text-yellow-400" />
                                    <span className="font-bold text-yellow-700 dark:text-yellow-300 uppercase text-xs tracking-wide">Giải thích chi tiết</span>
                                </div>
                                <p className="text-slate-700 dark:text-slate-300 text-sm leading-relaxed pl-7">
                                    {formatText(station.questions[0].explanation)}
                                </p>
                            </div>
                        </div>
                    </div>
                )})}
            </div>
        </div>
    );
};

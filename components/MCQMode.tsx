
import React, { useState, useRef, useEffect } from 'react';
import { generateMCQQuestions, analyzeResultWithOtter } from '../services/geminiService';
import { Difficulty, MCQQuestion, MentorResponse } from '../types';
import { CheckCircle2, CheckCircle, XCircle, BrainCircuit, RefreshCw, ArrowRight, AlertCircle, BookOpen, FileText, Activity, Clock, FileCheck, UploadCloud, Trash, Plus, File as FileIcon, X, Check, Sparkles, Loader2, ArrowUpCircle, Timer, AlertTriangle, ChevronDown, Zap, Trophy, ThumbsUp, ShieldAlert } from 'lucide-react';

// Declare pdfjsLib globally
declare const pdfjsLib: any;

interface UploadedFile {
    name: string;
    data: string; // Stores extracted Text for PDFs, or Base64 for images
    type: 'text' | 'base64'; 
}

// INCREASED LIMIT because we now extract text instead of loading full PDF into RAM
const MAX_FILE_SIZE = 200 * 1024 * 1024; // 200MB
const MAX_FILES_PER_CATEGORY = 3;

// Helper to format text (replace arrows with professional unicode characters)
const formatText = (text: string) => {
  if (!text) return "";
  return text.replace(/->/g, ' → ').replace(/=>/g, ' ⇒ ').replace(/<-/g, ' ← ');
};

interface FileCategoryProps {
  icon: React.ReactNode;
  title: string;
  desc: string;
  bgGradient: string;
  iconColor: string;
  glowClass: string;
  files: UploadedFile[];
  onRemove: (index: number) => void;
  onAdd: () => void;
}

const FileCategory: React.FC<FileCategoryProps> = ({ 
    icon, title, desc, bgGradient, iconColor, glowClass, files, onRemove, onAdd 
}) => {
    return (
        <div className={`group relative rounded-2xl border border-slate-200 dark:border-slate-700 p-4 transition-all duration-300 ${bgGradient} ${glowClass}`}>
            {/* Header */}
            <div className="flex items-center gap-4 mb-3 relative z-10">
                <div className={`w-12 h-12 rounded-xl bg-white dark:bg-slate-800 shadow-sm flex items-center justify-center ${iconColor} ring-1 ring-black/5 dark:ring-white/5`}>
                    {icon}
                </div>
                <div className="flex-1">
                    <h4 className="font-bold text-slate-800 dark:text-slate-200 text-sm uppercase tracking-wide">{title}</h4>
                    <p className="text-xs text-slate-500 dark:text-slate-400 font-medium">{desc}</p>
                </div>
                <button 
                    onClick={onAdd}
                    className="w-8 h-8 rounded-full bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-600 flex items-center justify-center hover:bg-medical-50 dark:hover:bg-medical-900/30 hover:border-medical-200 dark:hover:border-medical-500/50 text-slate-400 hover:text-medical-600 transition-all shadow-sm active:scale-90"
                    title="Thêm tài liệu"
                >
                    <Plus className="w-5 h-5" />
                </button>
            </div>

            {/* Files List */}
            {files.length > 0 ? (
                <div className="space-y-2 relative z-10">
                    {files.map((file, idx) => (
                        <div key={idx} className="flex items-center gap-2 bg-white/60 dark:bg-slate-800/60 p-2 rounded-lg border border-slate-100 dark:border-slate-700/50 text-xs animate-in fade-in slide-in-from-left-2">
                            <FileIcon className="w-3 h-3 text-slate-400 flex-shrink-0" />
                            <span className="flex-1 truncate font-medium text-slate-700 dark:text-slate-300" title={file.name}>{file.name}</span>
                            <button onClick={() => onRemove(idx)} className="text-slate-400 hover:text-red-500 p-1 rounded-md hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors">
                                <Trash className="w-3 h-3" />
                            </button>
                        </div>
                    ))}
                </div>
            ) : (
                <div className="relative z-10 mt-2 border-2 border-dashed border-slate-300 dark:border-slate-700 rounded-xl p-3 text-center hover:border-medical-400 dark:hover:border-medical-600 transition-colors cursor-pointer bg-white/30 dark:bg-black/10" onClick={onAdd}>
                    <p className="text-xs text-slate-400 font-medium">Chưa có file nào. <br/> Nhấn + để thêm.</p>
                </div>
            )}
        </div>
    );
};

// --- NEW COMPONENT: SKILL SCANNER BAR ---
const SkillBar: React.FC<{ label: string; correct: number; total: number; icon: React.ReactNode; color: string }> = ({ label, correct, total, icon, color }) => {
    const percentage = total > 0 ? Math.round((correct / total) * 100) : 0;
    
    // Determine color based on percentage
    let barColor = "bg-red-500";
    let statusText = "Cần ôn gấp";
    if (percentage >= 80) { barColor = "bg-green-500"; statusText = "Thượng thừa"; }
    else if (percentage >= 50) { barColor = "bg-yellow-400"; statusText = "Khá ổn"; }

    return (
        <div className="mb-4">
            <div className="flex justify-between items-end mb-1">
                <div className="flex items-center gap-2 text-sm font-bold text-slate-700 dark:text-slate-300">
                    {icon}
                    <span>{label}</span>
                </div>
                <div className="text-xs font-medium">
                    <span className={`${percentage >= 80 ? 'text-green-600 dark:text-green-400' : percentage >= 50 ? 'text-yellow-600 dark:text-yellow-400' : 'text-red-600 dark:text-red-400'} mr-2`}>
                        {statusText}
                    </span>
                    <span className="text-slate-400">({correct}/{total})</span>
                </div>
            </div>
            <div className="h-3 w-full bg-slate-100 dark:bg-slate-700 rounded-full overflow-hidden shadow-inner">
                <div 
                    className={`h-full ${barColor} transition-all duration-1000 ease-out rounded-full relative`}
                    style={{ width: `${percentage}%` }}
                >
                    <div className="absolute inset-0 bg-white/30 w-full h-full animate-[shimmer_2s_infinite]"></div>
                </div>
            </div>
        </div>
    );
};

export const MCQMode: React.FC<{ onBack: () => void }> = ({ onBack }) => {
  // Configuration State
  const [topic, setTopic] = useState('');
  const [count, setCount] = useState(10); // Default 10
  const [timeLimit, setTimeLimit] = useState(15); // Default 15 mins
  const [difficulties, setDifficulties] = useState<Difficulty[]>([Difficulty.UNDERSTAND]);
  
  // File States (Arrays of files)
  const [theoryFiles, setTheoryFiles] = useState<UploadedFile[]>([]);
  const [clinicalFiles, setClinicalFiles] = useState<UploadedFile[]>([]);
  const [sampleFiles, setSampleFiles] = useState<UploadedFile[]>([]);
  
  // Processing State
  const [isProcessingFile, setIsProcessingFile] = useState(false);

  // Quiz State
  const [loading, setLoading] = useState(false);
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [loadingText, setLoadingText] = useState('');
  const [questions, setQuestions] = useState<MCQQuestion[]>([]);
  const [userAnswers, setUserAnswers] = useState<Record<string, string>>({});
  const [showResult, setShowResult] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Timer State
  const [timeLeft, setTimeLeft] = useState(0); // In seconds

  // Mentor "Rái Cá Nhỏ" State
  const [showMentor, setShowMentor] = useState(false);
  const [mentorLoading, setMentorLoading] = useState(false);
  const [mentorData, setMentorData] = useState<MentorResponse | null>(null);
  const [mentorStats, setMentorStats] = useState<Record<string, { correct: number, total: number }> | null>(null);

  // Refs for hidden file inputs
  const theoryInputRef = useRef<HTMLInputElement>(null);
  const clinicalInputRef = useRef<HTMLInputElement>(null);
  const sampleInputRef = useRef<HTMLInputElement>(null);
  const mentorSectionRef = useRef<HTMLDivElement>(null);

  const extractTextFromPDF = async (file: File): Promise<string> => {
      const arrayBuffer = await file.arrayBuffer();
      // Using global pdfjsLib from CDN
      const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
      let fullText = "";
      
      // Extract text from all pages
      for (let i = 1; i <= pdf.numPages; i++) {
          const page = await pdf.getPage(i);
          const textContent = await page.getTextContent();
          const pageText = textContent.items.map((item: any) => item.str).join(" ");
          fullText += `\n--- Page ${i} ---\n${pageText}`;
      }
      
      return fullText;
  };

  // Helper to handle multi-file upload with TEXT EXTRACTION
  const handleFileChange = async (
    e: React.ChangeEvent<HTMLInputElement>, 
    setFiles: React.Dispatch<React.SetStateAction<UploadedFile[]>>,
    currentFiles: UploadedFile[]
  ) => {
    const files = e.target.files;
    setError(null);

    if (files) {
      const fileArray = Array.from(files) as File[];
      
      if (currentFiles.length + fileArray.length > MAX_FILES_PER_CATEGORY) {
          setError(`Chỉ được tải tối đa ${MAX_FILES_PER_CATEGORY} file cho mỗi mục.`);
          return;
      }

      setIsProcessingFile(true); // Show spinner

      for (const file of fileArray) {
          if (file.size > MAX_FILE_SIZE) {
              setError(`File ${file.name} quá lớn (>200MB).`);
              continue;
          }

          try {
            if (file.type === 'application/pdf') {
                // EXTRACT TEXT to avoid Out of Memory
                const text = await extractTextFromPDF(file);
                if (!text || text.trim().length < 100) {
                   // Warning if text layer is missing (scanned PDF)
                   setError(`File ${file.name} có vẻ là bản scan (ảnh). AI chỉ đọc được văn bản có thể copy. Vui lòng dùng file có Text Layer.`);
                }
                setFiles(prev => [...prev, { name: file.name, data: text, type: 'text' }]);
            } else {
                // Handle other types normally if needed (though we restrict to PDF mostly)
                const reader = new FileReader();
                reader.readAsDataURL(file);
                reader.onloadend = () => {
                    setFiles(prev => [...prev, { name: file.name, data: reader.result as string, type: 'base64' }]);
                };
            }
          } catch (err) {
              console.error("File processing error", err);
              setError(`Lỗi khi đọc file ${file.name}. File có thể bị hỏng.`);
          }
      }
      
      setIsProcessingFile(false);
    }
    // Reset input value
    if (e.target) e.target.value = '';
  };

  const removeFile = (index: number, setFiles: React.Dispatch<React.SetStateAction<UploadedFile[]>>) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const toggleDifficulty = (diff: Difficulty) => {
    setDifficulties(prev => {
      if (prev.includes(diff)) {
        // Allow deselecting even if it's the last one (we disable submit button instead)
        return prev.filter(d => d !== diff);
      } else {
        return [...prev, diff];
      }
    });
  };

  // Use Effect for Loading Animation text
  useEffect(() => {
    if (!loading) return;
    
    const messages = [
        "Đang tải dữ liệu...",
        "Phân tích cấu trúc PDF...",
        "Rái cá đang lọc ý chính...",
        "Xử lý các case lâm sàng...",
        "Rái cá nhỏ đang suy nghĩ...",
        "Cấu trúc hoá kiến thức...",
        "Đang soạn câu hỏi...",
        "Kiểm tra độ chính xác đáp án..."
    ];
    
    let msgIndex = 0;
    setLoadingText(messages[0]);

    const textInterval = setInterval(() => {
        msgIndex = (msgIndex + 1) % messages.length;
        setLoadingText(messages[msgIndex]);
    }, 2500);

    // Simulate progress
    const progressInterval = setInterval(() => {
        setLoadingProgress(prev => {
            if (prev >= 95) return prev; // Hold at 95% until finish
            // Slow down as it gets higher
            const increment = prev > 80 ? 0.2 : 1.5;
            return prev + increment;
        });
    }, 200);

    return () => {
        clearInterval(textInterval);
        clearInterval(progressInterval);
    };
  }, [loading]);

  // Timer Effect
  useEffect(() => {
    if (questions.length === 0 || showResult || loading) return;

    // Auto submit if time runs out
    if (timeLeft <= 0) {
        if (timeLeft === 0 && !loading && questions.length > 0) {
            setShowResult(true);
        }
        return;
    }

    const timerInterval = setInterval(() => {
        setTimeLeft(prev => {
            if (prev <= 0) return 0;
            return prev - 1;
        });
    }, 1000);

    return () => clearInterval(timerInterval);
  }, [timeLeft, questions.length, showResult, loading]);

  const formatTime = (seconds: number) => {
      const m = Math.floor(seconds / 60);
      const s = seconds % 60;
      return `${m < 10 ? '0' + m : m}:${s < 10 ? '0' + s : s}`;
  };

  const handleGenerate = async () => {
    if (!topic.trim() || difficulties.length === 0) return;
    
    // 1. Set Loading State immediately
    setLoading(true);
    setError(null);
    setLoadingProgress(5);
    setLoadingText("Gọi Rái cá nhỏ...");
    setMentorData(null);
    setMentorStats(null);
    setShowMentor(false);

    // 2. Defer the heavy processing to the next event loop tick
    setTimeout(async () => {
        try {
            const files = {
                theory: theoryFiles.map(f => ({ content: f.data, isText: f.type === 'text' })),
                clinical: clinicalFiles.map(f => ({ content: f.data, isText: f.type === 'text' })),
                sample: sampleFiles.map(f => ({ content: f.data, isText: f.type === 'text' })),
            };
            
            // API Call
            const response = await generateMCQQuestions(topic, count, difficulties, files);
            
            // Success
            const newQuestions: MCQQuestion[] = response.questions.map((q, idx) => ({
                ...q,
                id: `q-${Date.now()}-${idx}`
            }));
            
            // Jump to 100%
            setLoadingProgress(100);
            setLoadingText("Hoàn tất!");
            
            // Short delay to show 100% before switching view
            setTimeout(() => {
                // Initialize Timer here
                setTimeLeft(timeLimit * 60);
                setQuestions(newQuestions);
                setUserAnswers({});
                setShowResult(false);
                setLoading(false);
            }, 500);

        } catch (err: any) {
            console.error(err);
            let errMsg = "Không thể tạo câu hỏi. Vui lòng thử lại.";
            if (err.message && err.message.includes("Too Large")) {
                errMsg = "Dữ liệu quá lớn vượt quá giới hạn của Rái cá. Hãy thử giảm bớt số lượng file.";
            } else if (err.message) {
                errMsg = err.message;
            }
            setError(errMsg);
            setLoading(false);
        }
    }, 200);
  };

  const handleAnswer = (questionId: string, option: string) => {
    if (showResult) return;
    setUserAnswers(prev => ({ ...prev, [questionId]: option }));
  };

  const calculateScore = () => {
    let score = 0;
    questions.forEach(q => {
      if (userAnswers[q.id] === q.correctAnswer) score++;
    });
    return score;
  };

  // Mentor Logic
  const handleConsultMentor = async () => {
      if (mentorData) {
          setShowMentor(true);
          mentorSectionRef.current?.scrollIntoView({ behavior: 'smooth' });
          return;
      }

      setMentorLoading(true);
      setShowMentor(true);

      // 1. Detailed Stats Calculation per Difficulty
      const stats: Record<string, { correct: number, total: number }> = {};
      
      // Initialize all difficulties to ensure they appear even if 0 (Fix for missing 'Ghi nhớ')
      Object.values(Difficulty).forEach(d => {
          stats[d] = { correct: 0, total: 0 };
      });

      // Initialize all potential difficulties that are present in the quiz
      questions.forEach(q => {
          const diff = q.difficulty || "Khác";
          // Safety check if diff wasn't in enum
          if (!stats[diff]) stats[diff] = { correct: 0, total: 0 };
          
          stats[diff].total++;
          if (userAnswers[q.id] === q.correctAnswer) {
              stats[diff].correct++;
          }
      });
      
      setMentorStats(stats);

      // 2. Call AI
      try {
          const response = await analyzeResultWithOtter(topic, stats);
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

  // Difficulty Color Mapping for Capsules
  const getDifficultyColor = (diff: string | undefined) => {
    switch(diff) {
        case Difficulty.REMEMBER: return 'bg-blue-500/20 border-blue-400/30 text-blue-600 dark:text-blue-300';
        case Difficulty.UNDERSTAND: return 'bg-teal-500/20 border-teal-400/30 text-teal-600 dark:text-teal-300';
        case Difficulty.APPLY: return 'bg-amber-500/20 border-amber-400/30 text-amber-600 dark:text-amber-300';
        case Difficulty.CLINICAL: return 'bg-purple-500/20 border-purple-400/30 text-purple-600 dark:text-purple-300';
        default: return 'bg-slate-500/20 border-slate-400/30 text-slate-600 dark:text-slate-300';
    }
  };
  
  // Icon mapping for Skill Scanner
  const getDifficultyIcon = (diff: string) => {
      switch(diff) {
          case Difficulty.REMEMBER: return <BookOpen className="w-4 h-4" />;
          case Difficulty.UNDERSTAND: return <BrainCircuit className="w-4 h-4" />;
          case Difficulty.APPLY: return <Zap className="w-4 h-4" />;
          case Difficulty.CLINICAL: return <Activity className="w-4 h-4" />;
          default: return <Trophy className="w-4 h-4" />;
      }
  };

  // Result calculation helpers
  const correctCount = calculateScore();
  const totalCount = questions.length;
  const wrongCount = totalCount - correctCount;
  const scorePercentage = Math.round((correctCount / totalCount) * 100);
  
  // SVG Circle calculation
  const radius = 28;
  const circumference = 2 * Math.PI * radius;
  const dashOffset = circumference - (scorePercentage / 100) * circumference;

  // Piggy Math (Calculate position on circle)
  // SVG Container is rotated -90deg.
  // Center (35, 35), Radius 28.
  // Angle = Percentage converted to radians (0 to 2*PI)
  const pigAngle = (scorePercentage / 100) * 2 * Math.PI;
  const pigX = 35 + 28 * Math.cos(pigAngle);
  const pigY = 35 + 28 * Math.sin(pigAngle);


  // --- LOADING VIEW ---
  if (loading) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-100/80 dark:bg-slate-950/80 backdrop-blur-sm">
        {/* Glassmorphism Card */}
        <div className="w-full max-w-md bg-white/40 dark:bg-slate-900/60 backdrop-blur-xl border border-white/50 dark:border-slate-700 rounded-[2rem] p-8 shadow-2xl relative overflow-hidden flex flex-col items-center text-center animate-in zoom-in-95 duration-300">
            
            {/* Decorative Glow */}
            <div className="absolute -top-20 -left-20 w-60 h-60 bg-blue-500/20 rounded-full blur-[80px] animate-pulse"></div>
            <div className="absolute -bottom-20 -right-20 w-60 h-60 bg-teal-500/20 rounded-full blur-[80px] animate-pulse delay-700"></div>

            {/* Icon - Rái cá nhỏ (UPDATED: Ocean Waves & Sun Rays) */}
            <div className="relative mb-6">
                <div className="w-28 h-28 rounded-3xl shadow-[0_0_30px_rgba(14,165,233,0.3)] border border-white/50 dark:border-slate-600 relative overflow-hidden bg-gradient-to-b from-sky-300 to-blue-500 dark:from-sky-800 dark:to-blue-900 flex items-center justify-center group">
                    {/* Sun Glow (Top Left) */}
                    <div className="absolute -top-10 -left-10 w-40 h-40 bg-white/40 blur-2xl rounded-full animate-pulse"></div>
                    
                    {/* Sun Rays (Diagonal) */}
                    <div className="absolute inset-0 bg-[linear-gradient(135deg,transparent_40%,rgba(255,255,255,0.2)_45%,transparent_50%,rgba(255,255,255,0.2)_55%,transparent_60%)] opacity-80"></div>

                    {/* Ocean Waves (Bottom) */}
                    <div className="absolute bottom-0 left-0 right-0 h-12 bg-gradient-to-t from-blue-600/50 to-transparent opacity-50"></div>
                    {/* Wave 1 */}
                    <div className="absolute -bottom-4 left-[-20%] w-[140%] h-12 bg-blue-100/20 dark:bg-blue-400/20 rounded-[50%] animate-[bounce_4s_infinite]"></div>
                    {/* Wave 2 */}
                    <div className="absolute -bottom-6 left-[-20%] w-[140%] h-12 bg-blue-50/30 dark:bg-blue-300/30 rounded-[50%] animate-[bounce_3s_infinite_reverse]"></div>

                    {/* The Otter */}
                    <span className="text-6xl relative z-10 animate-[bounce_3s_infinite] drop-shadow-xl transform -translate-y-1">🦦</span>
                    
                    {/* Badge */}
                    <div className="absolute top-2 right-2 w-6 h-6 bg-white/20 backdrop-blur-sm rounded-full flex items-center justify-center border border-white/40 shadow-sm z-20">
                        <BrainCircuit className="w-3 h-3 text-white" />
                    </div>
                </div>
            </div>

            {/* Title with Gradient - Updated Text */}
            <h3 className="text-2xl font-extrabold mb-2 text-transparent bg-clip-text bg-gradient-to-r from-blue-500 via-cyan-500 to-teal-500">
                Rái cá nhỏ đang thiết kế đề thi...
            </h3>

            {/* Dynamic Status Text */}
            <div className="h-6 mb-8 flex items-center justify-center">
                <p className="text-slate-600 dark:text-slate-300 text-sm font-medium animate-fade-up key={loadingText}">
                    {loadingText} <span className="inline-block w-[3ch] text-left">{Math.floor(loadingProgress)}%</span>
                </p>
            </div>

            {/* Custom Progress Bar */}
            <div className="w-full h-4 bg-slate-200 dark:bg-slate-800 rounded-full overflow-hidden shadow-inner relative">
                {/* Moving gradient fill */}
                <div 
                    className="h-full rounded-full bg-gradient-to-r from-midnight-900 via-blue-600 to-teal-400 transition-all duration-300 ease-out shadow-[0_0_10px_rgba(14,165,233,0.5)]"
                    style={{ width: `${loadingProgress}%` }}
                ></div>
                
                {/* Shimmer effect overlay */}
                <div className="absolute inset-0 w-full h-full bg-gradient-to-r from-transparent via-white/30 to-transparent -translate-x-full animate-[shimmer_1.5s_infinite]"></div>
            </div>
            
            <p className="text-xs text-slate-400 mt-6 italic">
                Đang xử lý {theoryFiles.length + clinicalFiles.length + sampleFiles.length} tài liệu và cấu trúc hoá kiến thức...
            </p>
        </div>
      </div>
    );
  }

  // --- CONFIG VIEW ---
  if (questions.length === 0) {
    return (
      <div className="max-w-6xl mx-auto pb-20">
        <div className="flex items-center mb-6">
            <button onClick={onBack} className="mr-4 p-2 rounded-full hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-500 transition-colors">
                <ArrowRight className="w-6 h-6 rotate-180" />
            </button>
            <h2 className="text-xl font-medium text-slate-500 dark:text-slate-400">Quay lại</h2>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            
            {/* CARD 1: HEADER */}
            <div className="lg:col-span-12 bg-gradient-to-r from-medical-600 to-blue-700 rounded-3xl p-8 text-white shadow-xl relative overflow-hidden">
                <div className="absolute top-0 right-0 w-64 h-64 bg-white/10 rounded-full -mr-16 -mt-16 blur-3xl"></div>
                <div className="relative z-10 flex items-center gap-6">
                    <div className="w-20 h-20 bg-white/20 backdrop-blur-md rounded-2xl flex items-center justify-center border border-white/30 shadow-inner">
                        <BrainCircuit className="w-10 h-10 text-white drop-shadow-lg" />
                    </div>
                    <div>
                        <h1 className="text-3xl md:text-4xl font-bold mb-2 text-glow-white">Thiết lập đề thi</h1>
                        <p className="text-blue-100 text-lg">Tùy chỉnh thông số và cung cấp tài liệu để AI tạo đề sát thực tế nhất.</p>
                    </div>
                </div>
            </div>

            {/* CARD 2: UPLOADS (Left Column) */}
            <div className="lg:col-span-4 space-y-6">
                <div className="bg-white dark:bg-slate-900 rounded-3xl p-6 shadow-lg border border-slate-100 dark:border-slate-800 h-full flex flex-col relative overflow-hidden">
                    <h3 className="text-lg font-bold text-slate-800 dark:text-white mb-6 flex items-center gap-2 relative z-10">
                        <UploadCloud className="w-5 h-5 text-medical-500" />
                        Tài liệu tham khảo (PDF)
                    </h3>
                    
                    {isProcessingFile && (
                        <div className="absolute inset-0 z-50 bg-white/80 dark:bg-slate-900/80 backdrop-blur-sm flex items-center justify-center flex-col gap-2 rounded-3xl">
                             <Loader2 className="w-8 h-8 text-medical-500 animate-spin" />
                             <span className="text-sm font-bold text-medical-600">Đang đọc file PDF...</span>
                        </div>
                    )}
                    
                    <div className="space-y-4 flex-1 relative z-10">
                        <FileCategory 
                            icon={<BookOpen className="w-6 h-6" />}
                            title="Lý thuyết"
                            desc="Kiến thức nền (Ghi nhớ, Hiểu)"
                            bgGradient="bg-gradient-to-br from-blue-50/80 to-white/40 dark:from-blue-900/20 dark:to-slate-900/40"
                            iconColor="text-blue-600 dark:text-blue-400"
                            glowClass="hover:shadow-[0_0_25px_rgba(14,165,233,0.4)] dark:hover:shadow-[0_0_25px_rgba(14,165,233,0.2)] hover:border-blue-300 dark:hover:border-blue-500/50"
                            files={theoryFiles}
                            onRemove={(idx) => removeFile(idx, setTheoryFiles)}
                            onAdd={() => theoryInputRef.current?.click()}
                        />
                        <input type="file" ref={theoryInputRef} onChange={(e) => handleFileChange(e, setTheoryFiles, theoryFiles)} accept="application/pdf" multiple className="hidden" />

                        <FileCategory 
                            icon={<Activity className="w-6 h-6" />}
                            title="Lâm sàng"
                            desc="Case Study (Mức độ khó cao)"
                            bgGradient="bg-gradient-to-br from-teal-50/80 to-white/40 dark:from-teal-900/20 dark:to-slate-900/40"
                            iconColor="text-teal-600 dark:text-teal-400"
                            glowClass="hover:shadow-[0_0_25px_rgba(20,184,166,0.4)] dark:hover:shadow-[0_0_25px_rgba(20,184,166,0.2)] hover:border-teal-300 dark:hover:border-teal-500/50"
                            files={clinicalFiles}
                            onRemove={(idx) => removeFile(idx, setClinicalFiles)}
                            onAdd={() => clinicalInputRef.current?.click()}
                        />
                        <input type="file" ref={clinicalInputRef} onChange={(e) => handleFileChange(e, setClinicalFiles, clinicalFiles)} accept="application/pdf" multiple className="hidden" />

                        <FileCategory 
                            icon={<FileCheck className="w-6 h-6" />}
                            title="Đề thi mẫu"
                            desc="Để AI học format đề"
                            bgGradient="bg-gradient-to-br from-purple-50/80 to-white/40 dark:from-purple-900/20 dark:to-slate-900/40"
                            iconColor="text-purple-600 dark:text-purple-400"
                            glowClass="hover:shadow-[0_0_25px_rgba(168,85,247,0.4)] dark:hover:shadow-[0_0_25px_rgba(168,85,247,0.2)] hover:border-purple-300 dark:hover:border-purple-500/50"
                            files={sampleFiles}
                            onRemove={(idx) => removeFile(idx, setSampleFiles)}
                            onAdd={() => sampleInputRef.current?.click()}
                        />
                        <input type="file" ref={sampleInputRef} onChange={(e) => handleFileChange(e, setSampleFiles, sampleFiles)} accept="application/pdf" multiple className="hidden" />
                    </div>
                    <p className="text-xs text-slate-400 mt-4 text-center italic">
                        Chế độ xử lý văn bản: Hỗ trợ file PDF lớn (Textbooks).<br/>Không hỗ trợ hình ảnh trong PDF.
                    </p>
                </div>
            </div>

            {/* CARD 3: SETTINGS (Main Area) */}
            <div className="lg:col-span-8 space-y-6">
                <div className="bg-white dark:bg-slate-900 rounded-3xl p-8 shadow-lg border border-slate-100 dark:border-slate-800">
                    
                    {/* Topic Input */}
                    <div className="mb-10">
                        <label className="block text-sm font-bold text-slate-700 dark:text-slate-300 mb-3 uppercase tracking-wider">Chủ đề ôn tập</label>
                        <div className="relative">
                            <input
                                type="text"
                                value={topic}
                                onChange={(e) => setTopic(e.target.value)}
                                placeholder="Ví dụ: Giải phẫu tim, Hệ thần kinh trung ương..."
                                className="w-full text-xl p-4 pl-12 bg-slate-50 dark:bg-slate-800 border-2 border-slate-200 dark:border-slate-700 rounded-2xl focus:border-medical-500 focus:ring-4 focus:ring-medical-500/20 outline-none transition-all text-slate-900 dark:text-white placeholder-slate-400"
                            />
                            <FileText className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400 w-6 h-6" />
                        </div>
                    </div>

                    {/* Sliders Row */}
                    <div className="grid md:grid-cols-2 gap-10 mb-12">
                        {/* Question Count Slider */}
                        <div className="relative group">
                            <div className="flex justify-between mb-4">
                                <label className="flex items-center gap-2 font-bold text-slate-700 dark:text-slate-300">
                                    <BrainCircuit className="w-5 h-5 text-medical-500" />
                                    Số câu hỏi
                                </label>
                                <span className="text-2xl font-bold text-medical-600 dark:text-medical-400">{count}</span>
                            </div>
                            <div className="relative h-12 flex items-center">
                                <input
                                    type="range"
                                    min="1"
                                    max="50"
                                    value={count}
                                    onChange={(e) => setCount(Number(e.target.value))}
                                    className="liquid-slider"
                                    style={{ '--range-progress': `${((count - 1) / 49) * 100}%` } as React.CSSProperties}
                                />
                            </div>
                            <div className="flex justify-between text-xs text-slate-400 mt-1">
                                <span>1</span>
                                <span>50</span>
                            </div>
                        </div>

                        {/* Time Slider */}
                        <div className="relative group">
                            <div className="flex justify-between mb-4">
                                <label className="flex items-center gap-2 font-bold text-slate-700 dark:text-slate-300">
                                    <Clock className="w-5 h-5 text-purple-500" />
                                    Thời gian (phút)
                                </label>
                                <span className="text-2xl font-bold text-purple-600 dark:text-purple-400">{timeLimit}</span>
                            </div>
                            <div className="relative h-12 flex items-center">
                                <input
                                    type="range"
                                    min="1"
                                    max="90"
                                    step="1"
                                    value={timeLimit}
                                    onChange={(e) => setTimeLimit(Number(e.target.value))}
                                    className="liquid-slider"
                                    style={{ '--range-progress': `${((timeLimit - 1) / 89) * 100}%` } as React.CSSProperties}
                                />
                            </div>
                            <div className="flex justify-between text-xs text-slate-400 mt-1">
                                <span>1</span>
                                <span>90</span>
                            </div>
                        </div>
                    </div>

                    {/* Difficulty Cards */}
                    <div className="mb-10">
                        <label className="block text-sm font-bold text-slate-700 dark:text-slate-300 mb-4 uppercase tracking-wider">Mức độ câu hỏi (Có thể chọn nhiều)</label>
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                            {Object.entries(Difficulty).map(([key, value]) => {
                                const isSelected = difficulties.includes(value);
                                
                                return (
                                    <div 
                                        key={key}
                                        onClick={() => toggleDifficulty(value)}
                                        className={`relative p-5 rounded-2xl bg-gradient-to-br from-midnight-950 to-midnight-800 text-white border transition-all duration-300 cursor-pointer group overflow-hidden hover:shadow-[0_0_20px_rgba(56,189,248,0.3)] ${isSelected ? 'border-medical-500 shadow-glow-blue' : 'border-slate-700 hover:border-medical-500/50'}`}
                                    >
                                        {isSelected && <div className="absolute inset-0 bg-medical-500/10"></div>}
                                        <div className="relative z-10 flex flex-col items-center text-center h-full justify-between">
                                            <h4 className={`font-bold mb-3 group-hover:text-white transition-colors ${isSelected ? 'text-white' : 'text-slate-300'}`}>{value}</h4>
                                            <div className={`liquid-switch ${isSelected ? 'active' : ''}`}>
                                                <div className="liquid-switch-thumb"></div>
                                            </div>
                                        </div>
                                    </div>
                                )
                            })}
                        </div>
                    </div>
                    
                    {/* Error Message */}
                    {error && (
                        <div className="bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 p-4 rounded-xl flex items-center gap-2 mb-6 animate-pulse border border-red-200 dark:border-red-800">
                        <AlertCircle className="w-5 h-5 flex-shrink-0" />
                        <span className="text-sm font-medium">{error}</span>
                        </div>
                    )}

                    {/* Submit Button - Updated Text */}
                    <button
                        onClick={handleGenerate}
                        disabled={!topic.trim() || difficulties.length === 0}
                        className="w-full bg-medical-600 hover:bg-medical-700 disabled:bg-slate-300 dark:disabled:bg-slate-700 disabled:cursor-not-allowed text-white font-bold py-5 rounded-2xl shadow-lg hover:shadow-glow-blue transition-all flex items-center justify-center space-x-3 text-lg active:scale-95"
                    >
                        <BrainCircuit className="w-6 h-6" />
                        <span>Khởi tạo đề thi Rái cá nhỏ</span>
                    </button>

                </div>
            </div>
        </div>
      </div>
    );
  }

  // --- QUIZ VIEW ---
  return (
    <div className={`mx-auto px-4 transition-all duration-500 ${showResult ? 'lg:pl-80 max-w-7xl pb-20' : 'max-w-4xl pb-60'}`}>
      {/* 0) FLOATING TIMER */}
      {!showResult && (
         <div className={`fixed top-20 right-4 md:right-8 z-50 flex items-center gap-2 px-4 py-2 rounded-full shadow-lg backdrop-blur-xl border transition-all duration-500 ${timeLeft <= 15 ? 'bg-red-500 border-red-400 text-white animate-pulse' : 'bg-white/80 dark:bg-slate-800/80 border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-200'}`}>
            {timeLeft <= 15 ? <AlertTriangle className="w-5 h-5 animate-bounce" /> : <Timer className="w-5 h-5 text-blue-500" />}
            <span className={`font-mono font-bold text-xl ${timeLeft <= 15 ? '' : 'text-slate-900 dark:text-white'}`}>
                {formatTime(timeLeft)}
            </span>
         </div>
      )}

      {/* 1) HEADER CARD */}
      <div className="bg-white dark:bg-slate-900 rounded-3xl p-8 shadow-lg border border-slate-200 dark:border-slate-700 mb-10 relative overflow-hidden animate-fade-up">
        <div className="absolute top-0 right-0 w-32 h-32 bg-medical-500/10 rounded-full -mr-10 -mt-10 blur-2xl"></div>
        <div className="relative z-10 flex flex-col md:flex-row justify-between items-start md:items-center gap-6">
            <div>
                <div className="flex items-center gap-3 mb-2">
                    <span className="bg-medical-100 dark:bg-medical-900/30 text-medical-700 dark:text-medical-300 text-xs font-bold px-3 py-1 rounded-full uppercase tracking-wide border border-medical-200 dark:border-medical-800">
                        Trắc nghiệm
                    </span>
                </div>
                <h2 className="text-3xl font-bold text-slate-900 dark:text-white leading-tight">{topic}</h2>
            </div>

            <div className="flex gap-3 w-full md:w-auto">
                <button onClick={onBack} className="flex-1 md:flex-none px-6 py-3 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-xl font-bold transition-colors border border-transparent hover:border-slate-200 dark:hover:border-slate-700">
                    Thoát
                </button>
                <button
                    onClick={() => { setQuestions([]); setUserAnswers({}); setShowResult(false); setShowMentor(false); }}
                    className="flex-1 md:flex-none flex items-center justify-center space-x-2 px-6 py-3 bg-slate-900 dark:bg-white text-white dark:text-slate-900 rounded-xl hover:opacity-90 transition-all shadow-md font-bold"
                >
                    <RefreshCw className="w-4 h-4" />
                    <span>Làm đề khác</span>
                </button>
            </div>
        </div>
      </div>

      {/* 2) QUESTION CARDS LIST */}
      <div className="space-y-10">
        {questions.map((q, index) => (
            <div 
                key={q.id} 
                className="relative bg-white dark:bg-slate-900 p-8 rounded-3xl shadow-xl border border-slate-100 dark:border-slate-800 transition-all hover:shadow-2xl animate-fade-up"
                style={{ animationDelay: `${index * 100}ms` }}
            >
                <div className={`absolute -top-4 right-8 backdrop-blur-md border shadow-lg text-xs font-bold px-4 py-1.5 rounded-full z-10 flex items-center gap-2 ${getDifficultyColor(q.difficulty)} bg-white/80 dark:bg-slate-950/80`}>
                    <span className="w-2 h-2 rounded-full bg-current animate-pulse"></span>
                    {q.difficulty || "Câu hỏi"}
                </div>

                <div className="flex gap-6">
                    <div className="flex-shrink-0">
                         <div className="w-12 h-12 flex items-center justify-center bg-slate-100 dark:bg-slate-800 text-slate-900 dark:text-white font-bold text-xl rounded-2xl shadow-inner border border-slate-200 dark:border-slate-700">
                            {index + 1}
                         </div>
                    </div>
                    
                    <div className="flex-1">
                        <h3 className="text-xl font-bold text-slate-900 dark:text-white mb-8 leading-relaxed pr-4">
                            {formatText(q.question)}
                        </h3>
                        
                        <div className="grid grid-cols-1 gap-4">
                            {q.options.map((option, optIdx) => {
                                const isSelected = userAnswers[q.id] === option;
                                const isCorrect = option === q.correctAnswer;
                                
                                let optionClass = "relative p-5 rounded-2xl border-2 transition-all duration-300 cursor-pointer flex items-center group overflow-hidden ";
                                
                                if (showResult) {
                                    if (isCorrect) {
                                        optionClass += "bg-emerald-50 dark:bg-emerald-900/20 border-emerald-500 text-emerald-800 dark:text-emerald-300 shadow-[0_0_15px_rgba(16,185,129,0.2)] ";
                                    } else if (isSelected && !isCorrect) {
                                        optionClass += "bg-red-50 dark:bg-red-900/20 border-red-500 text-red-800 dark:text-red-300 opacity-80 ";
                                    } else {
                                        optionClass += "bg-slate-50 dark:bg-slate-800/50 border-transparent text-slate-400 dark:text-slate-500 opacity-50 ";
                                    }
                                } else {
                                    if (isSelected) {
                                        optionClass += "bg-gradient-to-r from-midnight-950 to-midnight-800 border-medical-500 text-white shadow-glow-blue scale-[1.01] ";
                                    } else {
                                        optionClass += "bg-slate-50 dark:bg-slate-800/50 border-transparent text-slate-700 dark:text-slate-300 hover:bg-white dark:hover:bg-slate-800 hover:border-slate-300 dark:hover:border-slate-600 hover:shadow-md ";
                                    }
                                }

                                return (
                                    <button
                                        key={optIdx}
                                        onClick={() => handleAnswer(q.id, option)}
                                        disabled={showResult}
                                        className={optionClass}
                                    >
                                        <div className={`w-8 h-8 rounded-lg flex items-center justify-center text-sm font-bold mr-4 transition-colors ${isSelected && !showResult ? 'bg-white/20 text-white' : 'bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-400'}`}>
                                            {String.fromCharCode(65 + optIdx)}
                                        </div>
                                        <span className="text-lg font-medium text-left flex-1">{formatText(option)}</span>
                                        {showResult && isCorrect && <CheckCircle2 className="w-6 h-6 text-emerald-500 ml-2" />}
                                        {showResult && isSelected && !isCorrect && <XCircle className="w-6 h-6 text-red-500 ml-2" />}
                                    </button>
                                );
                            })}
                        </div>

                        {showResult && (
                            <div className="mt-6 p-6 bg-blue-50/50 dark:bg-blue-900/10 text-slate-700 dark:text-slate-300 rounded-2xl border border-blue-100 dark:border-blue-800/30 animate-in fade-in slide-in-from-top-2">
                                <div className="flex items-center gap-2 mb-2 text-blue-700 dark:text-blue-400 font-bold uppercase text-xs tracking-wider">
                                    <BrainCircuit className="w-4 h-4" />
                                    Giải thích chi tiết
                                </div>
                                <p className="leading-relaxed">{formatText(q.explanation)}</p>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        ))}
      </div>

      {/* OTTER MENTOR SECTION (Shown when toggled) */}
      {showMentor && (
          <div ref={mentorSectionRef} className="mt-12 mb-32 animate-in slide-in-from-bottom-10 duration-700">
              {mentorLoading ? (
                   <div className="w-full bg-white dark:bg-slate-900 rounded-[2rem] p-8 shadow-xl border border-amber-200 dark:border-amber-900/30 text-center flex flex-col items-center gap-4">
                        <div className="text-6xl animate-bounce">🦦</div>
                        <p className="text-slate-600 dark:text-slate-300 font-medium animate-pulse">Rái cá nhỏ đang lặn tìm ngọc trai kiến thức...</p>
                   </div>
              ) : mentorData ? (
                   <div className="relative bg-gradient-to-b from-amber-50 to-white dark:from-slate-900 dark:to-slate-900 rounded-[2.5rem] p-8 md:p-10 shadow-2xl border border-amber-200 dark:border-slate-700 overflow-hidden">
                        {/* Background decoration */}
                        <div className="absolute top-0 right-0 w-64 h-64 bg-amber-300/20 rounded-full blur-3xl -mr-20 -mt-20"></div>
                        
                        {/* GRID LAYOUT FOR CONTENT */}
                        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 relative z-10">
                             
                             {/* COLUMN 1: OTTER PERSONA + ANALYSIS (5 cols) */}
                             <div className="lg:col-span-5 flex flex-col gap-6">
                                  <div className="flex items-center gap-6">
                                      <div className="w-28 h-28 bg-gradient-to-br from-amber-100 to-white dark:from-slate-800 dark:to-slate-700 rounded-full flex items-center justify-center shadow-lg border-4 border-white dark:border-slate-600 relative flex-shrink-0">
                                            <span className="text-6xl animate-[wiggle_3s_infinite]">🦦</span>
                                            <div className="absolute -bottom-2 bg-amber-500 text-white text-xs font-bold px-3 py-1 rounded-full border-2 border-white dark:border-slate-900">
                                                Cố vấn AI
                                            </div>
                                      </div>
                                      <div>
                                          <h3 className="text-xl font-bold text-slate-900 dark:text-white">Rái cá nhỏ</h3>
                                          <p className="text-xs text-amber-600 dark:text-amber-400 font-bold uppercase tracking-wide">Chuyên gia giải phẫu</p>
                                      </div>
                                  </div>
                                  
                                  {/* Analysis Bubble */}
                                  <div className="bg-white dark:bg-slate-800 p-6 rounded-2xl rounded-tl-none shadow-sm border border-amber-100 dark:border-slate-700 relative">
                                      <p className="text-slate-700 dark:text-slate-300 leading-relaxed italic text-lg">
                                        "{mentorData.analysis}"
                                      </p>
                                      <div className="absolute top-0 -left-3 w-0 h-0 border-t-[15px] border-t-white dark:border-t-slate-800 border-l-[15px] border-l-transparent"></div>
                                  </div>

                                  {/* NEW: Parallel Strength/Weakness Cards */}
                                  <div className="grid grid-cols-2 gap-4">
                                      {/* Strength Card */}
                                      <div className="bg-green-50/80 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-2xl p-4 flex flex-col h-full">
                                          <div className="flex items-center gap-2 mb-2 text-green-700 dark:text-green-400 font-bold text-sm uppercase">
                                              <ThumbsUp className="w-4 h-4" /> Điểm mạnh
                                          </div>
                                          <ul className="space-y-1">
                                              {mentorData.strengths && mentorData.strengths.length > 0 ? (
                                                  mentorData.strengths.map((s, i) => (
                                                      <li key={i} className="text-xs text-slate-600 dark:text-slate-300 flex items-start gap-1.5 leading-tight">
                                                          <span className="text-green-500 mt-0.5">•</span> {s}
                                                      </li>
                                                  ))
                                              ) : (
                                                  <li className="text-xs text-slate-400 italic">Chưa tìm thấy điểm mạnh nổi bật</li>
                                              )}
                                          </ul>
                                      </div>

                                      {/* Weakness Card */}
                                      <div className="bg-red-50/80 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-2xl p-4 flex flex-col h-full">
                                          <div className="flex items-center gap-2 mb-2 text-red-700 dark:text-red-400 font-bold text-sm uppercase">
                                              <ShieldAlert className="w-4 h-4" /> Điểm yếu
                                          </div>
                                          <ul className="space-y-1">
                                              {mentorData.weaknesses && mentorData.weaknesses.length > 0 ? (
                                                  mentorData.weaknesses.map((w, i) => (
                                                      <li key={i} className="text-xs text-slate-600 dark:text-slate-300 flex items-start gap-1.5 leading-tight">
                                                          <span className="text-red-500 mt-0.5">•</span> {w}
                                                      </li>
                                                  ))
                                              ) : (
                                                  <li className="text-xs text-slate-400 italic">Không có điểm yếu đáng kể</li>
                                              )}
                                          </ul>
                                      </div>
                                  </div>
                             </div>

                             {/* COLUMN 2: SKILL SCANNER CHART (7 cols) */}
                             <div className="lg:col-span-7 bg-white/50 dark:bg-slate-800/50 rounded-3xl p-6 border border-amber-100 dark:border-slate-700/50">
                                  <h4 className="text-sm font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-6 flex items-center gap-2">
                                      <Activity className="w-4 h-4" /> Hồ sơ năng lực
                                  </h4>
                                  
                                  {mentorStats && Object.entries(mentorStats).map(([diff, stat]) => {
                                    const s = stat as { correct: number; total: number };
                                    return (
                                      <SkillBar 
                                        key={diff}
                                        label={diff}
                                        correct={s.correct}
                                        total={s.total}
                                        icon={getDifficultyIcon(diff)}
                                        color="bg-amber-500" // Internal logic determines color
                                      />
                                    );
                                  })}
                                  
                                  {(!mentorStats || Object.keys(mentorStats).length === 0) && (
                                      <p className="text-center text-slate-400 text-sm italic py-4">Chưa đủ dữ liệu để phân tích.</p>
                                  )}
                             </div>
                        </div>

                        {/* ROADMAP SECTION */}
                        <div className="mt-12 pt-8 border-t border-amber-200/50 dark:border-slate-700/50">
                             <h4 className="text-center text-lg font-bold text-amber-600 dark:text-amber-400 uppercase tracking-widest mb-8 flex items-center justify-center gap-2">
                                 <Sparkles className="w-5 h-5" /> Lộ trình cải thiện
                             </h4>
                             
                             <div className="relative max-w-3xl mx-auto">
                                  {/* Connecting Line */}
                                  <div className="absolute left-8 md:left-1/2 top-0 bottom-0 w-1 bg-amber-200 dark:bg-slate-700 -translate-x-1/2 md:translate-x-0"></div>

                                  <div className="space-y-12">
                                      {mentorData.roadmap.map((step, idx) => (
                                          <div key={idx} className={`flex flex-col md:flex-row items-center gap-6 md:gap-10 relative group ${idx % 2 === 0 ? 'md:flex-row-reverse' : ''}`}>
                                              
                                              {/* Content Box */}
                                              <div className="flex-1 w-full md:w-auto pl-20 md:pl-0">
                                                   <div className={`bg-white dark:bg-slate-800 p-6 rounded-2xl shadow-md border border-slate-100 dark:border-slate-700 transition-transform hover:-translate-y-1 hover:shadow-lg ${idx % 2 === 0 ? 'md:text-right' : 'md:text-left'}`}>
                                                        <h5 className="text-lg font-bold text-slate-900 dark:text-white mb-3 flex items-center gap-2 text-medical-600 dark:text-medical-400" style={{ justifyContent: idx % 2 === 0 ? 'flex-end' : 'flex-start' }}>
                                                            {step.step}
                                                        </h5>
                                                        <p className="text-slate-600 dark:text-slate-300 text-sm leading-relaxed">
                                                            {step.details}
                                                        </p>
                                                   </div>
                                              </div>

                                              {/* Number Bubble */}
                                              <div className="absolute left-8 md:left-1/2 -translate-x-1/2 w-12 h-12 bg-amber-500 text-white rounded-full flex items-center justify-center font-bold text-xl shadow-[0_0_0_4px_white] dark:shadow-[0_0_0_4px_rgb(15,23,42)] z-10">
                                                  {idx + 1}
                                              </div>

                                              {/* Spacer for opposite side */}
                                              <div className="hidden md:block flex-1"></div>
                                          </div>
                                      ))}
                                  </div>
                             </div>
                        </div>
                   </div>
              ) : null}
          </div>
      )}

      {/* RESULT / SUBMIT BAR - FIXED BOTTOM OR SIDEBAR */}
      {!showResult ? (
        <div className="fixed bottom-8 left-0 right-0 px-4 flex justify-center z-50 pointer-events-none">
            <button
                onClick={() => setShowResult(true)}
                className="pointer-events-auto group relative flex items-center gap-3 px-8 py-4 rounded-full bg-gradient-to-r from-medical-600 to-blue-600 text-white font-bold shadow-[0_0_20px_rgba(14,165,233,0.5)] hover:shadow-[0_0_40px_rgba(14,165,233,0.8)] transition-all duration-300 hover:scale-105 active:scale-95 overflow-hidden"
            >
                {/* Shimmer overlay */}
                <div className="absolute inset-0 -translate-x-full group-hover:animate-[shimmer_1.5s_infinite] bg-gradient-to-r from-transparent via-white/30 to-transparent z-10" />

                <CheckCircle className="w-6 h-6 animate-bounce" />
                <span className="text-lg tracking-wide">Nộp bài & Xem điểm</span>
            </button>
        </div>
      ) : (
        /* Vertical Sidebar for Desktop, Bottom Sheet for Mobile */
        <div className="fixed z-40 transition-all duration-500 
            max-lg:bottom-0 max-lg:left-0 max-lg:right-0 max-lg:rounded-t-[2.5rem] max-lg:p-4 max-lg:bg-white/95 max-lg:dark:bg-slate-900/95 max-lg:backdrop-blur-xl max-lg:border-t max-lg:shadow-[0_-10px_40px_rgba(0,0,0,0.1)]
            lg:top-24 lg:left-8 lg:w-72 lg:h-[calc(100vh-8rem)] lg:bg-white/80 lg:dark:bg-slate-900/80 lg:backdrop-blur-xl lg:rounded-[2rem] lg:border lg:border-slate-200 lg:dark:border-slate-700/50 lg:p-6 lg:flex lg:flex-col lg:shadow-xl lg:overflow-y-auto scrollbar-hide
        ">
            {/* Inner Layout Wrapper */}
            <div className="flex flex-col items-center gap-6 h-full lg:justify-between">
                
                {/* Section 1: Chart & Score */}
                <div className="flex flex-row lg:flex-col items-center gap-5 w-full justify-center lg:justify-start pt-2 lg:pt-4">
                    <div className="relative w-20 h-20 lg:w-32 lg:h-32 flex-shrink-0 group">
                         <div className="absolute inset-0 bg-blue-500/20 rounded-full blur-xl group-hover:blur-2xl transition-all"></div>
                         <svg className="w-full h-full transform -rotate-90 relative z-10" viewBox="0 0 70 70">
                            <defs>
                                <linearGradient id="glitterGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                                    <stop offset="0%" stopColor="#0f172a" />
                                    <stop offset="50%" stopColor="#38bdf8" />
                                    <stop offset="100%" stopColor="#1e40af" />
                                </linearGradient>
                            </defs>

                            <circle
                                cx="35"
                                cy="35"
                                r={radius}
                                fill="none"
                                stroke="currentColor"
                                className="text-slate-200 dark:text-slate-800"
                                strokeWidth="6"
                            />
                            <circle
                                cx="35"
                                cy="35"
                                r={radius}
                                fill="none"
                                stroke="url(#glitterGradient)"
                                strokeWidth="6"
                                strokeDasharray={circumference}
                                strokeDashoffset={dashOffset}
                                strokeLinecap="round"
                                className="transition-all duration-1000 ease-out drop-shadow-[0_0_3px_rgba(56,189,248,0.8)]"
                            />
                            
                            {/* Piggy Grader Marker */}
                            {/* Calculated based on score percentage relative to rotated SVG coordinate system */}
                            <foreignObject x={pigX - 7} y={pigY - 7} width="14" height="14">
                                <div className="w-full h-full rounded-full bg-pink-100 border border-pink-300 flex items-center justify-center shadow-sm rotate-90 transition-transform hover:scale-125" title="Heo con chấm điểm">
                                    <span className="text-[9px]">🐷</span>
                                </div>
                            </foreignObject>
                         </svg>
                         <div className="absolute inset-0 flex items-center justify-center flex-col z-20">
                            <span className="text-xl lg:text-3xl font-black text-transparent bg-clip-text bg-gradient-to-br from-blue-900 via-blue-500 to-blue-900 dark:from-blue-300 dark:via-blue-100 dark:to-blue-300">
                                {scorePercentage}%
                            </span>
                         </div>
                    </div>
                    <div className="flex flex-col items-start lg:items-center">
                        <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">Heo con chấm</span>
                        <span className="text-2xl font-bold text-slate-900 dark:text-white">
                            {correctCount} <span className="text-slate-400 text-lg">/ {totalCount}</span>
                        </span>
                    </div>
                </div>

                {/* Section 2: Stats Detail */}
                <div className="grid grid-cols-2 lg:grid-cols-1 gap-4 w-full border-t border-b border-slate-100 dark:border-slate-800 py-4 lg:py-6">
                     <div className="flex items-center justify-center lg:justify-between gap-3 p-3 rounded-xl bg-green-50 dark:bg-green-900/10">
                        <div className="flex items-center gap-3">
                            <div className="w-8 h-8 rounded-full bg-green-100 dark:bg-green-900/30 flex items-center justify-center text-green-600 dark:text-green-400">
                                <Check className="w-4 h-4" />
                            </div>
                            <p className="text-sm font-medium text-slate-600 dark:text-slate-300">Đúng</p>
                        </div>
                        <p className="text-lg font-bold text-green-600 dark:text-green-400">{correctCount}</p>
                     </div>

                     <div className="flex items-center justify-center lg:justify-between gap-3 p-3 rounded-xl bg-red-50 dark:bg-red-900/10">
                        <div className="flex items-center gap-3">
                            <div className="w-8 h-8 rounded-full bg-red-100 dark:bg-red-900/30 flex items-center justify-center text-red-600 dark:text-red-400">
                                <X className="w-4 h-4" />
                            </div>
                            <p className="text-sm font-medium text-slate-600 dark:text-slate-300">Sai</p>
                        </div>
                        <p className="text-lg font-bold text-red-600 dark:text-red-400">{wrongCount}</p>
                     </div>
                </div>

                {/* Section 3: Actions */}
                <div className="flex flex-row lg:flex-col gap-3 w-full">
                     {!showMentor && (
                         <button 
                            onClick={handleConsultMentor}
                            disabled={mentorLoading}
                            className="relative w-full py-3 rounded-xl bg-amber-100 dark:bg-amber-900/30 text-amber-800 dark:text-amber-300 border border-amber-200 dark:border-amber-800/50 hover:bg-amber-200 dark:hover:bg-amber-900/50 transition-all flex items-center justify-center gap-2 font-bold group overflow-hidden"
                         >
                             <span className="text-xl group-hover:scale-125 transition-transform">🦦</span>
                             <span>Hỏi Rái cá</span>
                             {/* Liquid Blob Effect */}
                             <div className="absolute inset-0 rounded-xl animate-ping bg-amber-400/10 -z-10"></div>
                         </button>
                     )}

                     <button 
                        onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
                        className="flex-1 py-3 rounded-xl hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-400 transition-colors border border-slate-200 dark:border-slate-700 flex items-center justify-center gap-2 text-sm font-medium"
                     >
                         <ArrowUpCircle className="w-5 h-5" />
                         <span className="hidden lg:inline">Xem lại bài</span>
                     </button>
                     
                     <button 
                        onClick={() => { setQuestions([]); setUserAnswers({}); setShowResult(false); setShowMentor(false); }}
                        className="flex-1 py-3 bg-slate-900 dark:bg-white text-white dark:text-slate-900 rounded-xl font-bold hover:opacity-90 transition-all shadow-lg flex items-center justify-center gap-2 text-sm"
                     >
                         <RefreshCw className="w-4 h-4" />
                         <span>Thi lại</span>
                     </button>
                </div>
            </div>
        </div>
      )}
    </div>
  );
};

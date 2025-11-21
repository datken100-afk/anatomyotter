
import React, { useState, useEffect } from 'react';
import { Layout } from './components/Layout';
import { LoginScreen } from './components/LoginScreen';
import { MCQMode } from './components/MCQMode';
import { StationMode } from './components/StationMode';
import { AppMode, UserProfile } from './types';
import { BookOpen, Activity, ChevronRight } from 'lucide-react';

const App: React.FC = () => {
  const [user, setUser] = useState<UserProfile | null>(null);
  const [mode, setMode] = useState<AppMode>(AppMode.HOME);
  const [darkMode, setDarkMode] = useState(false);

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  const handleLogin = (loggedInUser: UserProfile) => {
    setUser(loggedInUser);
  };

  const handleLogout = () => {
    setUser(null);
    setMode(AppMode.HOME);
  };

  const handleUpdateUser = (updatedUser: UserProfile) => {
    setUser(updatedUser);
  };

  const currentYear = new Date().getFullYear();

  const renderContent = () => {
    switch (mode) {
      case AppMode.MCQ:
        return (
          // Removed animate-fade-up to prevent breaking fixed positioning of Timer and Result Bar
          <MCQMode onBack={() => setMode(AppMode.HOME)} />
        );
      case AppMode.STATION:
        return (
          // Removed animate-fade-up to prevent breaking fixed positioning of Station Runner
          <StationMode onBack={() => setMode(AppMode.HOME)} />
        );
      default:
        return (
          <div className="max-w-4xl mx-auto pt-10 animate-fade-up">
            <div className="text-center mb-16 space-y-4">
              <h1 className="text-4xl md:text-5xl font-extrabold text-slate-900 dark:text-white tracking-tight">
                Xin chào, <span className="text-amber-600 dark:text-amber-400">{user?.fullName}</span>
              </h1>
              <p className="text-lg text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
                Chọn chế độ để bắt đầu ôn luyện kiến thức Giải phẫu học ngay hôm nay.
              </p>
            </div>

            <div className="grid md:grid-cols-2 gap-8">
              {/* MCQ Card */}
              <button
                onClick={() => setMode(AppMode.MCQ)}
                className="group bg-white dark:bg-slate-800 p-8 rounded-3xl shadow-md hover:shadow-2xl border border-slate-100 dark:border-slate-700 transition-all duration-300 text-left relative overflow-hidden hover:-translate-y-2 hover:scale-[1.02]"
              >
                <div className="absolute top-0 right-0 w-32 h-32 bg-blue-50 dark:bg-blue-900/20 rounded-full -mr-10 -mt-10 group-hover:scale-150 transition-transform duration-500"></div>
                <div className="relative z-10">
                  <div 
                    className="w-16 h-16 bg-blue-100 dark:bg-blue-900/50 text-blue-600 dark:text-blue-400 rounded-2xl flex items-center justify-center mb-6 liquid-icon"
                    style={{ '--glow-color': 'rgba(37, 99, 235, 0.8)' } as React.CSSProperties}
                  >
                    <BookOpen className="w-8 h-8" />
                  </div>
                  <h3 className="text-2xl font-bold text-slate-900 dark:text-white mb-3 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">Trắc Nghiệm</h3>
                  <p className="text-slate-500 dark:text-slate-400 mb-6 leading-relaxed">
                    Tạo đề thi trắc nghiệm nhanh chóng theo chủ đề (Xương, Cơ, Thần kinh...). Giải thích chi tiết từng đáp án.
                  </p>
                  <div className="flex items-center text-blue-600 dark:text-blue-400 font-semibold group-hover:translate-x-2 transition-transform">
                    Bắt đầu ngay <ChevronRight className="w-5 h-5 ml-1" />
                  </div>
                </div>
              </button>

              {/* Station Card */}
              <button
                onClick={() => setMode(AppMode.STATION)}
                className="group bg-white dark:bg-slate-800 p-8 rounded-3xl shadow-md hover:shadow-2xl border border-slate-100 dark:border-slate-700 transition-all duration-300 text-left relative overflow-hidden hover:-translate-y-2 hover:scale-[1.02]"
              >
                <div className="absolute top-0 right-0 w-32 h-32 bg-teal-50 dark:bg-teal-900/20 rounded-full -mr-10 -mt-10 group-hover:scale-150 transition-transform duration-500"></div>
                <div className="relative z-10">
                  <div 
                    className="w-16 h-16 bg-teal-100 dark:bg-teal-900/50 text-teal-600 dark:text-teal-400 rounded-2xl flex items-center justify-center mb-6 liquid-icon"
                    style={{ '--glow-color': 'rgba(13, 148, 136, 0.8)' } as React.CSSProperties}
                  >
                    <Activity className="w-8 h-8" />
                  </div>
                  <h3 className="text-2xl font-bold text-slate-900 dark:text-white mb-3 group-hover:text-teal-600 dark:group-hover:text-teal-400 transition-colors">Chạy Trạm (Spot Test)</h3>
                  <p className="text-slate-500 dark:text-slate-400 mb-6 leading-relaxed">
                    Mô phỏng thi thực hành. Tải lên hình ảnh giải phẫu, AI sẽ tạo câu hỏi định danh và tính giờ tự động.
                  </p>
                  <div className="flex items-center text-teal-600 dark:text-teal-400 font-semibold group-hover:translate-x-2 transition-transform">
                    Tạo trạm thi <ChevronRight className="w-5 h-5 ml-1" />
                  </div>
                </div>
              </button>
            </div>

            {/* Footer Info */}
            <div className="mt-20 text-center border-t border-slate-200 dark:border-slate-800 pt-8 pb-8">
                <p className="text-slate-400 text-sm leading-relaxed">
                    © {currentYear} Lam Chan Dat (Y2025B - PNTU). All rights reserved.
                    <br />
                    Designed & Developed by Lam Chan Dat
                </p>
            </div>
          </div>
        );
    }
  };

  return (
    <>
      {!user ? (
        <div className="w-full h-full animate-fade-up">
          <LoginScreen 
            onLogin={handleLogin} 
            darkMode={darkMode} 
            toggleDarkMode={() => setDarkMode(!darkMode)} 
          />
        </div>
      ) : (
        <Layout 
          user={user} 
          onLogout={handleLogout} 
          onUpdateUser={handleUpdateUser}
          darkMode={darkMode}
          toggleDarkMode={() => setDarkMode(!darkMode)}
        >
          {renderContent()}
        </Layout>
      )}
    </>
  );
};

export default App;

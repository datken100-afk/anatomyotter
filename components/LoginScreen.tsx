
import React, { useState, useRef } from 'react';
import { ArrowRight, GraduationCap, User, Sun, Moon, Camera, Upload } from 'lucide-react';
import { UserProfile } from '../types';

interface LoginScreenProps {
  onLogin: (user: UserProfile) => void;
  darkMode: boolean;
  toggleDarkMode: () => void;
}

export const LoginScreen: React.FC<LoginScreenProps> = ({ onLogin, darkMode, toggleDarkMode }) => {
  const [fullName, setFullName] = useState('');
  const [studentId, setStudentId] = useState('');
  const [avatar, setAvatar] = useState<string | null>(null);
  const [error, setError] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!fullName.trim() || !studentId.trim()) {
      setError('Vui lòng nhập đầy đủ thông tin');
      return;
    }
    onLogin({
      fullName: fullName.trim(),
      studentId: studentId.trim(),
      avatar: avatar || undefined,
    });
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setAvatar(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 flex items-center justify-center p-4 transition-colors duration-300 relative">
      
      {/* Dark Mode Toggle (Absolute positioned) */}
      <div className="absolute top-4 right-4 sm:top-8 sm:right-8">
        <button 
          onClick={toggleDarkMode}
          className="liquid-icon relative rounded-xl text-slate-500 dark:text-slate-400 w-10 h-10 flex items-center justify-center overflow-hidden focus:outline-none bg-white dark:bg-slate-900 shadow-sm"
          style={{ '--glow-color': darkMode ? 'rgba(99, 102, 241, 0.5)' : 'rgba(245, 158, 11, 0.5)' } as React.CSSProperties}
          aria-label="Toggle Dark Mode"
        >
           <div className={`absolute transition-all duration-500 ease-[cubic-bezier(0.23,1,0.32,1)] ${darkMode ? 'rotate-0 scale-100 opacity-100' : '-rotate-90 scale-0 opacity-0'}`}>
              <Sun className="w-5 h-5" />
           </div>
           <div className={`absolute transition-all duration-500 ease-[cubic-bezier(0.23,1,0.32,1)] ${darkMode ? 'rotate-90 scale-0 opacity-0' : 'rotate-0 scale-100 opacity-100'}`}>
              <Moon className="w-5 h-5" />
           </div>
        </button>
      </div>

      <div className="max-w-md w-full bg-white dark:bg-slate-900 rounded-3xl shadow-2xl overflow-hidden border border-slate-100 dark:border-slate-800">
        {/* BRANDING HEADER: AMBER/ORANGE GRADIENT */}
        <div className="bg-gradient-to-br from-amber-400 to-orange-600 p-8 text-center relative overflow-hidden">
          <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-br from-amber-500 to-orange-700 opacity-50"></div>
          <div className="relative z-10">
            <div className="w-16 h-16 bg-white/20 rounded-2xl flex items-center justify-center mx-auto mb-4 backdrop-blur-sm">
              <span className="text-4xl leading-none drop-shadow-md">🦦</span>
            </div>
            <h1 className="text-2xl font-bold text-white text-glow-white">AnatomyOtter</h1>
            <p className="text-white/90 mt-2">Hệ thống ôn thi giải phẫu thông minh</p>
          </div>
        </div>

        <div className="p-8">
          <div className="flex justify-center -mt-16 mb-6 relative z-20">
            <div 
              className="relative group cursor-pointer"
              onClick={() => fileInputRef.current?.click()}
            >
              <div className={`w-24 h-24 rounded-full border-4 border-white dark:border-slate-800 shadow-lg flex items-center justify-center overflow-hidden bg-slate-100 dark:bg-slate-700 transition-all ${!avatar ? 'hover:bg-slate-200 dark:hover:bg-slate-600' : ''}`}>
                {avatar ? (
                  <img src={avatar} alt="Avatar" className="w-full h-full object-cover" />
                ) : (
                  <User className="w-10 h-10 text-slate-400" />
                )}
              </div>
              <div className="absolute bottom-0 right-0 bg-amber-500 text-white p-2 rounded-full shadow-md hover:bg-amber-600 transition-colors border-2 border-white dark:border-slate-900">
                {avatar ? <Camera className="w-4 h-4" /> : <Upload className="w-4 h-4" />}
              </div>
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleFileChange} 
                accept="image/*" 
                className="hidden" 
              />
            </div>
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">Họ và tên</label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                  <User className="h-5 w-5 text-slate-400" />
                </div>
                <input
                  type="text"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  className="w-full pl-11 pr-4 py-3 bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl focus:ring-2 focus:ring-amber-500 focus:border-transparent outline-none transition-all text-slate-900 dark:text-white placeholder-slate-400"
                  placeholder="Nguyễn Văn A"
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">Mã số sinh viên</label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                  <GraduationCap className="h-5 w-5 text-slate-400" />
                </div>
                <input
                  type="text"
                  value={studentId}
                  onChange={(e) => setStudentId(e.target.value)}
                  className="w-full pl-11 pr-4 py-3 bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl focus:ring-2 focus:ring-amber-500 focus:border-transparent outline-none transition-all text-slate-900 dark:text-white placeholder-slate-400"
                  placeholder="Y2024..."
                />
              </div>
            </div>

            {error && (
              <p className="text-red-500 text-sm text-center bg-red-50 dark:bg-red-900/20 py-2 rounded-lg">
                {error}
              </p>
            )}

            <button
              type="submit"
              className="w-full bg-gradient-to-r from-amber-500 to-orange-600 hover:from-amber-600 hover:to-orange-700 text-white font-bold py-4 rounded-xl shadow-lg btn-glow flex items-center justify-center gap-2 group"
            >
              <span>Bắt đầu học</span>
              <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </button>
          </form>
        </div>
        
        <div className="bg-slate-50 dark:bg-slate-800/50 p-4 text-center text-xs text-slate-500 dark:text-slate-400 border-t border-slate-100 dark:border-slate-800">
          © {new Date().getFullYear()} AnatomyOtter. Dành cho sinh viên Y khoa.
        </div>
      </div>
    </div>
  );
};

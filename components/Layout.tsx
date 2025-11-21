
import React, { useState, useRef } from 'react';
import { Moon, Sun, UserCircle, LogOut, Settings, Check, X, Camera, Sparkles } from 'lucide-react';
import { UserProfile } from '../types';
import { OtterChat } from './OtterChat';

interface LayoutProps {
  children: React.ReactNode;
  user: UserProfile;
  onLogout: () => void;
  onUpdateUser: (user: UserProfile) => void;
  darkMode: boolean;
  toggleDarkMode: () => void;
}

export const Layout: React.FC<LayoutProps> = ({ 
  children, 
  user, 
  onLogout, 
  onUpdateUser,
  darkMode,
  toggleDarkMode
}) => {
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const closeTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Edit Mode State
  const [isEditing, setIsEditing] = useState(false);
  const [editName, setEditName] = useState(user.fullName);
  const [editId, setEditId] = useState(user.studentId);
  const [editAvatar, setEditAvatar] = useState<string | undefined>(user.avatar);

  // Logo Focus Mode State
  const [isOtterMode, setIsOtterMode] = useState(false);

  const handleMouseEnter = () => {
    if (closeTimeoutRef.current) {
      clearTimeout(closeTimeoutRef.current);
      closeTimeoutRef.current = null;
    }
    setIsProfileOpen(true);
  };

  const handleMouseLeave = () => {
    closeTimeoutRef.current = setTimeout(() => {
      setIsProfileOpen(false);
      if (!isEditing) {
         // Only reset if not currently editing
         setIsEditing(false);
         setEditName(user.fullName);
         setEditId(user.studentId);
         setEditAvatar(user.avatar);
      }
    }, 300);
  };

  const handleSaveProfile = () => {
    if (editName.trim() && editId.trim()) {
        onUpdateUser({ 
            fullName: editName, 
            studentId: editId,
            avatar: editAvatar
        });
        setIsEditing(false);
    }
  };

  const cancelEdit = () => {
    setEditName(user.fullName);
    setEditId(user.studentId);
    setEditAvatar(user.avatar);
    setIsEditing(false);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setEditAvatar(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 dark:bg-slate-950 transition-colors duration-300">
      
      {/* OTTER FOCUS MODE OVERLAY - SMOOTH TRANSITION UPDATE */}
      {/* We keep this rendered to allow exit animations via CSS classes */}
      <div 
          className={`fixed inset-0 z-[100] flex items-center justify-center transition-all duration-500 ease-[cubic-bezier(0.34,1.56,0.64,1)]
          ${isOtterMode ? 'opacity-100 visible backdrop-blur-md bg-slate-900/60' : 'opacity-0 invisible backdrop-blur-none bg-transparent pointer-events-none'}`}
          onClick={() => setIsOtterMode(false)}
      >
          <div 
              className={`relative max-w-lg w-full mx-4 bg-gradient-to-br from-amber-400 to-orange-600 p-1 rounded-[3rem] shadow-[0_0_100px_rgba(245,158,11,0.4)] cursor-default group transition-all duration-500 ease-[cubic-bezier(0.34,1.56,0.64,1)]
              ${isOtterMode ? 'scale-100 translate-y-0 opacity-100' : 'scale-50 -translate-y-20 -translate-x-20 opacity-0'}`}
              onClick={(e) => e.stopPropagation()}
          >
              <div className="bg-white dark:bg-slate-900 rounded-[2.9rem] p-12 flex flex-col items-center text-center relative overflow-hidden">
                  {/* Background decoration */}
                  <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(251,191,36,0.15),transparent_70%)]"></div>
                  <div className="absolute top-0 right-0 w-40 h-40 bg-amber-400/10 rounded-full blur-3xl -mr-20 -mt-20"></div>
                  <div className="absolute bottom-0 left-0 w-40 h-40 bg-orange-500/10 rounded-full blur-3xl -ml-20 -mb-20"></div>
                  
                  {/* Large Animated Otter */}
                  <div 
                      className="relative z-10 mb-8 transform transition-transform duration-500 hover:scale-110 cursor-pointer" 
                      onClick={() => setIsOtterMode(false)}
                  >
                      <div className="text-[8rem] md:text-[10rem] leading-none animate-[bounce_3s_infinite] drop-shadow-2xl filter">
                          🦦
                      </div>
                      {/* Sparkles */}
                      <div className="absolute top-0 right-0 animate-pulse text-amber-400"><Sparkles className="w-8 h-8" /></div>
                      <div className="absolute bottom-4 left-4 animate-pulse delay-300 text-orange-400"><Sparkles className="w-6 h-6" /></div>
                  </div>

                  <h2 className="text-4xl md:text-5xl font-black text-transparent bg-clip-text bg-gradient-to-r from-amber-500 to-orange-600 mb-4 tracking-tight">
                      AnatomyOtter
                  </h2>
                  <p className="text-slate-600 dark:text-slate-300 text-lg font-medium mb-8 max-w-xs mx-auto leading-relaxed">
                      Người bạn đồng hành tin cậy trên hành trình chinh phục giải phẫu học.
                  </p>

                  <button 
                      onClick={() => setIsOtterMode(false)}
                      className="px-8 py-3 rounded-full bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400 text-sm font-bold hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors"
                  >
                      Đóng
                  </button>
              </div>
          </div>
      </div>

      <header className="bg-white dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800 sticky top-0 z-40 transition-colors duration-300">
        <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
          <div 
            className="flex items-center space-x-2 group cursor-pointer select-none transition-transform active:scale-95"
            onClick={() => setIsOtterMode(true)}
          >
            <div 
                className="w-10 h-10 bg-gradient-to-br from-amber-400 to-orange-500 rounded-xl flex items-center justify-center liquid-icon relative z-10"
                style={{ '--glow-color': 'rgba(245, 158, 11, 0.6)' } as React.CSSProperties}
            >
                <span className="text-2xl leading-none">🦦</span>
            </div>
            <h1 className="text-xl font-bold text-slate-800 dark:text-white tracking-tight transition-colors">
              Anatomy<span className="text-amber-500 text-glow">Otter</span>
            </h1>
          </div>
          
          <div className="flex items-center gap-4">
            <button 
              onClick={toggleDarkMode}
              className="liquid-icon relative rounded-xl text-slate-500 dark:text-slate-400 w-10 h-10 flex items-center justify-center overflow-hidden focus:outline-none bg-slate-100 dark:bg-slate-800"
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
            
            {/* Profile Section with Hover Popup */}
            <div 
                className="relative z-50"
                onMouseEnter={handleMouseEnter}
                onMouseLeave={handleMouseLeave}
            >
               {/* Unified Trigger Card */}
               <div 
                  className="flex items-center gap-3 px-3 py-1.5 rounded-2xl liquid-icon cursor-pointer bg-transparent transition-all duration-300 border border-transparent hover:border-amber-100 dark:hover:border-amber-900"
                  style={{ '--glow-color': 'rgba(245, 158, 11, 0.5)' } as React.CSSProperties}
               >
                  <div className="hidden md:block text-right">
                      <p className="text-sm font-bold text-slate-700 dark:text-slate-200 transition-colors">{user.fullName}</p>
                      <p className="text-xs text-slate-400 transition-colors">{user.studentId}</p>
                  </div>
                  <div className="text-slate-600 dark:text-slate-300 transition-colors relative">
                      {user.avatar ? (
                        <img 
                          src={user.avatar} 
                          alt="Profile" 
                          className="w-9 h-9 rounded-full object-cover border border-slate-200 dark:border-slate-700" 
                        />
                      ) : (
                        <UserCircle className="w-9 h-9" />
                      )}
                  </div>
               </div>

               {/* Dropdown Popup */}
               <div 
                  ref={dropdownRef}
                  className={`absolute top-full right-0 mt-2 w-80 bg-white dark:bg-slate-900 rounded-2xl shadow-2xl border border-slate-100 dark:border-slate-700 transform transition-all duration-300 origin-top-right overflow-hidden ${isProfileOpen ? 'opacity-100 translate-y-0 visible' : 'opacity-0 -translate-y-4 invisible'}`}
               >
                  <div className="p-4 border-b border-slate-100 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/50">
                    <p className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Thông tin cá nhân</p>
                  </div>
                  
                  <div className="p-4">
                    {isEditing ? (
                        <div className="space-y-3 animate-in fade-in slide-in-from-right-4 duration-200">
                            {/* Avatar Upload in Edit Mode */}
                            <div className="flex justify-center mb-4">
                                <div 
                                    className="relative group cursor-pointer"
                                    onClick={() => fileInputRef.current?.click()}
                                >
                                    <div className="w-16 h-16 rounded-full overflow-hidden border-2 border-amber-500 relative">
                                        {editAvatar ? (
                                            <img src={editAvatar} alt="Edit" className="w-full h-full object-cover" />
                                        ) : (
                                            <div className="w-full h-full bg-slate-100 dark:bg-slate-800 flex items-center justify-center">
                                                <UserCircle className="w-10 h-10 text-slate-400" />
                                            </div>
                                        )}
                                        {/* Overlay */}
                                        <div className="absolute inset-0 bg-black/40 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                                            <Camera className="w-6 h-6 text-white" />
                                        </div>
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

                            <div>
                                <label className="text-xs text-slate-500 dark:text-slate-400 mb-1 block">Họ và tên</label>
                                <input 
                                    value={editName} 
                                    onChange={(e) => setEditName(e.target.value)}
                                    className="w-full text-sm p-2 rounded border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-900 dark:text-white focus:ring-2 focus:ring-amber-500 outline-none"
                                />
                            </div>
                            <div>
                                <label className="text-xs text-slate-500 dark:text-slate-400 mb-1 block">Mã số sinh viên</label>
                                <input 
                                    value={editId} 
                                    onChange={(e) => setEditId(e.target.value)}
                                    className="w-full text-sm p-2 rounded border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-900 dark:text-white focus:ring-2 focus:ring-amber-500 outline-none"
                                />
                            </div>
                            <div className="flex gap-2 pt-2">
                                <button onClick={handleSaveProfile} className="flex-1 bg-amber-500 text-white py-1.5 rounded-lg text-sm font-medium hover:bg-amber-600 flex items-center justify-center gap-1">
                                    <Check className="w-4 h-4" /> Lưu
                                </button>
                                <button onClick={cancelEdit} className="flex-1 bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 py-1.5 rounded-lg text-sm font-medium hover:bg-slate-200 dark:hover:bg-slate-700 flex items-center justify-center gap-1">
                                    <X className="w-4 h-4" /> Hủy
                                </button>
                            </div>
                        </div>
                    ) : (
                        <div className="space-y-4">
                            <div className="flex items-center gap-3">
                                <div className="w-12 h-12 rounded-full flex items-center justify-center overflow-hidden border border-slate-100 dark:border-slate-700 bg-amber-100 dark:bg-amber-900/30 text-amber-600 dark:text-amber-400">
                                    {user.avatar ? (
                                        <img src={user.avatar} alt="User" className="w-full h-full object-cover" />
                                    ) : (
                                        <UserCircle className="w-7 h-7" />
                                    )}
                                </div>
                                <div>
                                    <p className="font-bold text-slate-900 dark:text-white">{user.fullName}</p>
                                    <p className="text-sm text-slate-500 dark:text-slate-400">{user.studentId}</p>
                                </div>
                            </div>
                            <button 
                                onClick={() => {
                                    setIsEditing(true);
                                    setEditName(user.fullName);
                                    setEditId(user.studentId);
                                    setEditAvatar(user.avatar);
                                }}
                                className="w-full py-2 border border-slate-200 dark:border-slate-700 rounded-xl text-sm font-medium text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors flex items-center justify-center gap-2"
                            >
                                <Settings className="w-4 h-4" />
                                Chỉnh sửa thông tin
                            </button>
                        </div>
                    )}
                  </div>

                  <div className="p-2 border-t border-slate-100 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/30">
                    <button 
                        onClick={onLogout}
                        className="w-full py-2 px-4 rounded-xl text-sm font-medium text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors flex items-center gap-2"
                    >
                        <LogOut className="w-4 h-4" />
                        Đăng xuất
                    </button>
                  </div>
               </div>
            </div>
          </div>
        </div>
      </header>
      <main className="flex-1 max-w-7xl mx-auto w-full px-4 py-8">
        {children}
      </main>
      
      {/* Otter Chat Widget */}
      <OtterChat />
    </div>
  );
};

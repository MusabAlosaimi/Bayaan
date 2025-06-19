import streamlit as st

# Page configuration MUST be first
st.set_page_config(
    page_title="أذكار المسلم الذكي - Smart Islamic Adhkar",
    page_icon="🕌",
    layout="wide",
    initial_sidebar_state="collapsed"
)

import pandas as pd
import numpy as np
from datetime import datetime
import random
import pickle
from collections import Counter
import re

# Try to import optional dependencies
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# Try to import scikit-learn, fallback gracefully if not available
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

# Premium Modern CSS Design with Advanced Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Noto+Kufi+Arabic:wght@300;400;500;600;700&display=swap');
    
    /* Advanced CSS Variables - Premium Color System */
    :root {
        /* Primary Colors - Islamic Green & Teal */
        --primary-50: #ecfdf5;
        --primary-100: #d1fae5;
        --primary-200: #a7f3d0;
        --primary-300: #6ee7b7;
        --primary-400: #34d399;
        --primary-500: #10b981;
        --primary-600: #059669;
        --primary-700: #047857;
        --primary-800: #065f46;
        --primary-900: #064e3b;
        
        /* Secondary Colors - Deep Blue */
        --secondary-50: #eff6ff;
        --secondary-100: #dbeafe;
        --secondary-200: #bfdbfe;
        --secondary-300: #93c5fd;
        --secondary-400: #60a5fa;
        --secondary-500: #3b82f6;
        --secondary-600: #2563eb;
        --secondary-700: #1d4ed8;
        --secondary-800: #1e40af;
        --secondary-900: #1e3a8a;
        
        /* Accent Colors - Gold */
        --accent-50: #fffbeb;
        --accent-100: #fef3c7;
        --accent-200: #fde68a;
        --accent-300: #fcd34d;
        --accent-400: #fbbf24;
        --accent-500: #f59e0b;
        --accent-600: #d97706;
        --accent-700: #b45309;
        --accent-800: #92400e;
        --accent-900: #78350f;
        
        /* Neutral Colors */
        --gray-50: #f9fafb;
        --gray-100: #f3f4f6;
        --gray-200: #e5e7eb;
        --gray-300: #d1d5db;
        --gray-400: #9ca3af;
        --gray-500: #6b7280;
        --gray-600: #4b5563;
        --gray-700: #374151;
        --gray-800: #1f2937;
        --gray-900: #111827;
        
        /* Status Colors */
        --success-50: #ecfdf5;
        --success-500: #10b981;
        --success-600: #059669;
        --warning-50: #fffbeb;
        --warning-500: #f59e0b;
        --error-50: #fef2f2;
        --error-500: #ef4444;
        
        /* Spacing & Shadows */
        --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
        --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
        --shadow-2xl: 0 25px 50px -12px rgb(0 0 0 / 0.25);
        
        --radius-sm: 6px;
        --radius-md: 8px;
        --radius-lg: 12px;
        --radius-xl: 16px;
        --radius-2xl: 20px;
        --radius-3xl: 24px;
    }
    
    /* Global Reset & Base Styles */
    * {
        box-sizing: border-box;
    }
    
    .stApp {
        background: linear-gradient(135deg, 
            var(--primary-50) 0%, 
            var(--secondary-50) 25%,
            var(--accent-50) 50%,
            var(--primary-50) 75%,
            var(--secondary-50) 100%);
        min-height: 100vh;
        font-family: 'Inter', 'Noto Kufi Arabic', sans-serif;
    }
    
    /* Hide Streamlit defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    
    /* Advanced Header with Glassmorphism */
    .premium-header {
        background: linear-gradient(135deg, 
            rgba(16, 185, 129, 0.95) 0%,
            rgba(59, 130, 246, 0.9) 50%,
            rgba(245, 158, 11, 0.95) 100%);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white;
        padding: 3rem 0;
        margin: -1rem -1rem 3rem -1rem;
        box-shadow: var(--shadow-2xl);
        position: relative;
        overflow: hidden;
    }
    
    .premium-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Ccircle cx='30' cy='30' r='1.5'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E") repeat;
        pointer-events: none;
    }
    
    .header-content {
        max-width: 1400px;
        margin: 0 auto;
        padding: 0 2rem;
        text-align: center;
        position: relative;
        z-index: 1;
    }
    
    .header-logo {
        margin-bottom: 1.5rem;
        animation: fadeInDown 0.8s ease-out;
    }
    
    .header-logo img {
        height: 100px;
        width: auto;
        filter: brightness(0) invert(1) drop-shadow(0 4px 8px rgba(0,0,0,0.3));
        transition: transform 0.3s ease;
    }
    
    .header-logo img:hover {
        transform: scale(1.05);
    }
    
    .header-title {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        font-family: 'Amiri', serif;
        text-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        background: linear-gradient(45deg, #ffffff, #f0f9ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: fadeInUp 0.8s ease-out 0.2s both;
        line-height: 1.2;
    }
    
    .header-subtitle {
        font-size: 1.4rem;
        color: rgba(255, 255, 255, 0.95);
        font-family: 'Inter', sans-serif;
        font-weight: 400;
        margin-bottom: 0.5rem;
        animation: fadeInUp 0.8s ease-out 0.4s both;
    }
    
    .ai-status {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(255, 255, 255, 0.2);
        padding: 8px 16px;
        border-radius: var(--radius-3xl);
        font-size: 0.9rem;
        font-weight: 500;
        margin-top: 1rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        animation: fadeInUp 0.8s ease-out 0.6s both;
    }
    
    /* Enhanced Navigation Tabs */
    .premium-tabs {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 12px;
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(20px);
        border-radius: var(--radius-2xl);
        padding: 12px;
        margin-bottom: 3rem;
        box-shadow: var(--shadow-xl);
        border: 1px solid rgba(255, 255, 255, 0.5);
    }
    
    .premium-tab {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
        padding: 16px 24px;
        background: transparent;
        border: 2px solid transparent;
        border-radius: var(--radius-xl);
        cursor: pointer;
        font-family: 'Inter', 'Noto Kufi Arabic', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        color: var(--gray-600);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .premium-tab::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
        transition: left 0.5s;
    }
    
    .premium-tab:hover::before {
        left: 100%;
    }
    
    .premium-tab:hover {
        background: linear-gradient(135deg, var(--primary-50), var(--secondary-50));
        color: var(--primary-700);
        border-color: var(--primary-200);
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
    }
    
    .premium-tab.active {
        background: linear-gradient(135deg, var(--primary-600), var(--secondary-600));
        color: white;
        border-color: var(--primary-500);
        box-shadow: var(--shadow-xl);
        transform: translateY(-2px);
    }
    
    .premium-tab .tab-icon {
        font-size: 1.2rem;
        transition: transform 0.3s ease;
    }
    
    .premium-tab:hover .tab-icon {
        transform: scale(1.1);
    }
    
    /* Advanced Search Container */
    .premium-search {
        position: relative;
        margin-bottom: 3rem;
    }
    
    .search-box {
        position: relative;
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: var(--radius-2xl);
        border: 2px solid rgba(16, 185, 129, 0.2);
        box-shadow: var(--shadow-xl);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        overflow: hidden;
    }
    
    .search-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--primary-500), var(--secondary-500), var(--accent-500));
        transform: scaleX(0);
        transition: transform 0.3s ease;
    }
    
    .search-box:focus-within::before {
        transform: scaleX(1);
    }
    
    .search-box:focus-within {
        border-color: var(--primary-500);
        box-shadow: 0 0 0 4px rgba(16, 185, 129, 0.1), var(--shadow-2xl);
        transform: translateY(-4px);
    }
    
    /* Enhanced Cards with Glassmorphism */
    .premium-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px) saturate(180%);
        border-radius: var(--radius-2xl);
        box-shadow: var(--shadow-lg);
        border: 1px solid rgba(255, 255, 255, 0.5);
        margin-bottom: 2rem;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }
    
    .premium-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, 
            rgba(16, 185, 129, 0.02) 0%,
            rgba(59, 130, 246, 0.02) 50%,
            rgba(245, 158, 11, 0.02) 100%);
        opacity: 0;
        transition: opacity 0.3s ease;
        pointer-events: none;
    }
    
    .premium-card:hover::before {
        opacity: 1;
    }
    
    .premium-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: var(--shadow-2xl);
        border-color: rgba(16, 185, 129, 0.3);
    }
    
    .card-glow {
        position: relative;
    }
    
    .card-glow::after {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, var(--primary-500), var(--secondary-500), var(--accent-500));
        border-radius: var(--radius-2xl);
        z-index: -1;
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .card-glow:hover::after {
        opacity: 0.5;
    }
    
    /* Enhanced Category Badges */
    .premium-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 8px 16px;
        border-radius: var(--radius-3xl);
        font-size: 0.875rem;
        font-weight: 600;
        font-family: 'Inter', 'Noto Kufi Arabic', sans-serif;
        text-transform: capitalize;
        border: 1px solid transparent;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .premium-badge::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
        transition: left 0.5s;
    }
    
    .premium-badge:hover::before {
        left: 100%;
    }
    
    .badge-morning { 
        background: linear-gradient(135deg, var(--accent-100), var(--accent-200));
        color: var(--accent-800);
        border-color: var(--accent-300);
    }
    .badge-evening { 
        background: linear-gradient(135deg, var(--secondary-100), var(--secondary-200));
        color: var(--secondary-800);
        border-color: var(--secondary-300);
    }
    .badge-general { 
        background: linear-gradient(135deg, var(--primary-100), var(--primary-200));
        color: var(--primary-800);
        border-color: var(--primary-300);
    }
    .badge-istighfar { 
        background: linear-gradient(135deg, var(--success-50), var(--success-100));
        color: var(--success-600);
        border-color: var(--success-200);
    }
    .badge-protection { 
        background: linear-gradient(135deg, var(--error-50), var(--error-100));
        color: var(--error-600);
        border-color: var(--error-200);
    }
    
    /* Enhanced Arabic Text */
    .premium-arabic {
        font-family: 'Amiri', 'Noto Kufi Arabic', serif;
        font-size: 2rem;
        line-height: 1.8;
        color: var(--gray-800);
        margin: 2rem 0;
        text-align: right;
        direction: rtl;
        background: linear-gradient(135deg, 
            rgba(16, 185, 129, 0.05) 0%,
            rgba(255, 255, 255, 0.8) 50%,
            rgba(59, 130, 246, 0.05) 100%);
        padding: 2rem;
        border-radius: var(--radius-xl);
        border-right: 6px solid var(--primary-500);
        border-left: 2px solid var(--secondary-500);
        box-shadow: var(--shadow-md);
        position: relative;
        overflow: hidden;
    }
    
    .premium-arabic::before {
        content: '﷽';
        position: absolute;
        top: 10px;
        right: 10px;
        font-size: 1.5rem;
        color: rgba(16, 185, 129, 0.1);
        font-weight: bold;
    }
    
    /* Premium Buttons */
    .premium-button {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        padding: 12px 24px;
        border-radius: var(--radius-xl);
        border: 2px solid transparent;
        font-family: 'Inter', 'Noto Kufi Arabic', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        cursor: pointer;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        text-decoration: none;
        position: relative;
        overflow: hidden;
    }
    
    .premium-button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.5s;
    }
    
    .premium-button:hover::before {
        left: 100%;
    }
    
    .btn-primary {
        background: linear-gradient(135deg, var(--primary-600), var(--primary-700));
        color: white;
        box-shadow: var(--shadow-md);
    }
    
    .btn-primary:hover {
        background: linear-gradient(135deg, var(--primary-700), var(--primary-800));
        transform: translateY(-2px);
        box-shadow: var(--shadow-xl);
        border-color: var(--primary-400);
    }
    
    .btn-secondary {
        background: linear-gradient(135deg, var(--secondary-600), var(--secondary-700));
        color: white;
        box-shadow: var(--shadow-md);
    }
    
    .btn-secondary:hover {
        background: linear-gradient(135deg, var(--secondary-700), var(--secondary-800));
        transform: translateY(-2px);
        box-shadow: var(--shadow-xl);
    }
    
    .btn-ghost {
        background: rgba(255, 255, 255, 0.8);
        color: var(--gray-700);
        border-color: var(--gray-300);
        backdrop-filter: blur(10px);
    }
    
    .btn-ghost:hover {
        background: rgba(255, 255, 255, 0.95);
        color: var(--gray-900);
        border-color: var(--gray-400);
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
    }
    
    .btn-favorite {
        background: linear-gradient(135deg, var(--error-500), var(--accent-500));
        color: white;
    }
    
    .btn-favorite:hover {
        background: linear-gradient(135deg, var(--error-600), var(--accent-600));
        transform: translateY(-2px) scale(1.05);
    }
    
    /* Enhanced Stats Grid */
    .premium-stats {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 2rem;
        margin-bottom: 3rem;
    }
    
    .stat-card {
        background: linear-gradient(135deg, 
            rgba(255, 255, 255, 0.9),
            rgba(255, 255, 255, 0.7));
        backdrop-filter: blur(20px);
        border-radius: var(--radius-2xl);
        padding: 2rem;
        text-align: center;
        box-shadow: var(--shadow-lg);
        border: 1px solid rgba(255, 255, 255, 0.5);
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .stat-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--primary-500), var(--secondary-500), var(--accent-500));
    }
    
    .stat-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-2xl);
    }
    
    .stat-card.primary {
        background: linear-gradient(135deg, var(--primary-500), var(--primary-600));
        color: white;
    }
    
    .stat-card.secondary {
        background: linear-gradient(135deg, var(--secondary-500), var(--secondary-600));
        color: white;
    }
    
    .stat-card.accent {
        background: linear-gradient(135deg, var(--accent-500), var(--accent-600));
        color: white;
    }
    
    .stat-number {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        background: linear-gradient(45deg, currentColor, rgba(255,255,255,0.8));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .stat-label {
        font-size: 1.1rem;
        font-weight: 500;
        opacity: 0.9;
    }
    
    /* Enhanced Greeting Box */
    .premium-greeting {
        background: linear-gradient(135deg, 
            rgba(16, 185, 129, 0.1),
            rgba(59, 130, 246, 0.1),
            rgba(245, 158, 11, 0.1));
        backdrop-filter: blur(20px);
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: var(--radius-2xl);
        padding: 2rem;
        margin: 2rem 0;
        text-align: center;
        box-shadow: var(--shadow-lg);
        position: relative;
        overflow: hidden;
    }
    
    .premium-greeting::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(16, 185, 129, 0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }
    
    .greeting-content {
        position: relative;
        z-index: 1;
    }
    
    .greeting-title {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-700);
        margin-bottom: 0.5rem;
        font-family: 'Amiri', serif;
    }
    
    /* Enhanced Empty State */
    .premium-empty {
        text-align: center;
        padding: 4rem 2rem;
        color: var(--gray-500);
        background: rgba(255, 255, 255, 0.5);
        border-radius: var(--radius-2xl);
        border: 2px dashed var(--gray-300);
        margin: 2rem 0;
    }
    
    .empty-icon {
        font-size: 5rem;
        margin-bottom: 1rem;
        opacity: 0.4;
        background: linear-gradient(45deg, var(--primary-500), var(--secondary-500));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes rotate {
        from {
            transform: rotate(0deg);
        }
        to {
            transform: rotate(360deg);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.7;
        }
    }
    
    /* Responsive Design */
    @media (max-width: 1024px) {
        .header-title {
            font-size: 2.5rem;
        }
        
        .premium-tabs {
            grid-template-columns: repeat(2, 1fr);
        }
        
        .premium-stats {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    
    @media (max-width: 768px) {
        .header-title {
            font-size: 2rem;
        }
        
        .header-subtitle {
            font-size: 1.1rem;
        }
        
        .premium-arabic {
            font-size: 1.6rem;
            padding: 1.5rem;
        }
        
        .premium-tabs {
            grid-template-columns: 1fr;
            gap: 8px;
        }
        
        .premium-stats {
            grid-template-columns: 1fr;
        }
        
        .premium-tab {
            padding: 12px 16px;
            font-size: 0.9rem;
        }
        
        .stat-number {
            font-size: 2.5rem;
        }
    }
    
    /* Streamlit Component Overrides */
    .stTabs [data-baseweb="tab-list"] {
        display: none;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        padding: 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-600), var(--primary-700)) !important;
        color: white !important;
        border: 2px solid transparent !important;
        border-radius: var(--radius-xl) !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        font-family: 'Inter', 'Noto Kufi Arabic', sans-serif !important;
        font-size: 0.95rem !important;
        width: 100% !important;
        box-shadow: var(--shadow-md) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .stButton > button::before {
        content: '' !important;
        position: absolute !important;
        top: 0 !important;
        left: -100% !important;
        width: 100% !important;
        height: 100% !important;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent) !important;
        transition: left 0.5s !important;
    }
    
    .stButton > button:hover::before {
        left: 100% !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, var(--primary-700), var(--primary-800)) !important;
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-xl) !important;
        border-color: var(--primary-400) !important;
    }
    
    .stButton > button:focus {
        background: linear-gradient(135deg, var(--primary-700), var(--primary-800)) !important;
        box-shadow: 0 0 0 4px rgba(16, 185, 129, 0.2), var(--shadow-xl) !important;
    }
    
    .stTextInput > div > div > input {
        border-radius: var(--radius-xl) !important;
        border: 2px solid rgba(16, 185, 129, 0.2) !important;
        padding: 16px 24px !important;
        font-size: 1.1rem !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        background: rgba(255, 255, 255, 0.9) !important;
        color: var(--gray-800) !important;
        box-shadow: var(--shadow-md) !important;
        backdrop-filter: blur(10px) !important;
        font-family: 'Inter', 'Noto Kufi Arabic', sans-serif !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--primary-500) !important;
        box-shadow: 0 0 0 4px rgba(16, 185, 129, 0.1), var(--shadow-lg) !important;
        background: rgba(255, 255, 255, 0.95) !important;
        transform: translateY(-2px) !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: var(--gray-500) !important;
        font-style: normal !important;
    }
    
    .stSelectbox > div > div > div {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 2px solid rgba(16, 185, 129, 0.2) !important;
        border-radius: var(--radius-xl) !important;
        box-shadow: var(--shadow-md) !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stSelectbox > div > div > div:focus-within {
        border-color: var(--primary-500) !important;
        box-shadow: 0 0 0 4px rgba(16, 185, 129, 0.1), var(--shadow-lg) !important;
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, var(--primary-500), var(--secondary-500)) !important;
    }
    
    .stSlider > div > div > div > div > div {
        background: white !important;
        border: 3px solid var(--primary-500) !important;
        box-shadow: var(--shadow-md) !important;
    }
    
    .stMetric {
        background: rgba(255, 255, 255, 0.8) !important;
        padding: 1rem !important;
        border-radius: var(--radius-xl) !important;
        box-shadow: var(--shadow-md) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.5) !important;
    }
    
    .stAlert {
        border-radius: var(--radius-xl) !important;
        border: none !important;
        box-shadow: var(--shadow-lg) !important;
        backdrop-filter: blur(20px) !important;
    }
    
    .stAlert[data-baseweb="notification"] {
        background: rgba(16, 185, 129, 0.1) !important;
        border-left: 4px solid var(--primary-500) !important;
    }
    
    .stSuccess {
        background: rgba(16, 185, 129, 0.1) !important;
        border-left: 4px solid var(--success-500) !important;
        color: var(--success-800) !important;
    }
    
    .stWarning {
        background: rgba(245, 158, 11, 0.1) !important;
        border-left: 4px solid var(--warning-500) !important;
        color: var(--warning-800) !important;
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.1) !important;
        border-left: 4px solid var(--error-500) !important;
        color: var(--error-800) !important;
    }
    
    .stInfo {
        background: rgba(59, 130, 246, 0.1) !important;
        border-left: 4px solid var(--secondary-500) !important;
        color: var(--secondary-800) !important;
    }
    
    .stCode {
        background: rgba(0, 0, 0, 0.05) !important;
        border-radius: var(--radius-lg) !important;
        border: 1px solid var(--gray-200) !important;
        font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    }
    
    /* Enhanced loading spinner */
    .stSpinner > div {
        border-top-color: var(--primary-500) !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--gray-100);
        border-radius: var(--radius-md);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--primary-500), var(--secondary-500));
        border-radius: var(--radius-md);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, var(--primary-600), var(--secondary-600));
    }
</style>
""", unsafe_allow_html=True)

# Enhanced utility functions
def remove_tashkeel(text):
    """Remove Arabic diacritics for better text processing"""
    tashkeel_pattern = re.compile(r'[\u064B-\u065F\u0670]')
    return tashkeel_pattern.sub('', text)

def manual_cosine_similarity(a, b):
    """Manual cosine similarity calculation"""
    a_dense = a.toarray().flatten()
    b_dense = b.toarray()
    dot_products = np.dot(b_dense, a_dense)
    a_norm = np.linalg.norm(a_dense)
    b_norms = np.linalg.norm(b_dense, axis=1)
    return dot_products / (a_norm * b_norms + 1e-10)

@st.cache_data
def load_data():
    """Load and cache the adhkar data with enhanced error handling"""
    try:
        df = pd.read_csv('adhkar_df.csv')
        if df.empty:
            st.error("⚠️ ملف البيانات فارغ")
            return pd.DataFrame()
        return df.dropna()
    except FileNotFoundError:
        st.error("❌ لم يتم العثور على ملف البيانات 'adhkar_df.csv'")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ خطأ في تحميل البيانات: {e}")
        return pd.DataFrame()

def load_model_and_vectorizer():
    """Load model and data with enhanced error handling"""
    try:
        if JOBLIB_AVAILABLE:
            vectorizer = joblib.load("tfidf_vectorizer.pkl")
            df = pd.read_csv("adhkar_df.csv")
            return vectorizer, df
        else:
            with open("tfidf_vectorizer.pkl", 'rb') as f:
                vectorizer = pickle.load(f)
            df = pd.read_csv("adhkar_df.csv")
            return vectorizer, df
    except FileNotFoundError:
        st.warning("⚠️ لم يتم العثور على ملف النموذج - الميزات الذكية معطلة")
        return None, pd.DataFrame()
    except Exception as e:
        st.error(f"❌ خطأ في تحميل النموذج: {e}")
        return None, pd.DataFrame()

def find_similar_dua(user_dua, vectorizer, adhkar_df):
    """Enhanced dua similarity finder"""
    if not user_dua or not user_dua.strip():
        return "❗ الرجاء إدخال دعاء صحيح", ""
    
    clean_dua = remove_tashkeel(user_dua.strip())
    if len(clean_dua) < 3:
        return "❗ الرجاء إدخال نص أطول من حرفين", ""
    
    try:
        user_vector = vectorizer.transform([clean_dua])
        tfidf_matrix = vectorizer.transform(adhkar_df['clean_text'])
        similarities = manual_cosine_similarity(user_vector, tfidf_matrix)
        best_idx = similarities.argmax()
        best_score = similarities[best_idx]
        
        if best_score < 0.08:
            return "❌ لم يتم العثور على دعاء مشابه بدقة كافية", ""
        
        return adhkar_df.iloc[best_idx]['category'], adhkar_df.iloc[best_idx]['text']
    except Exception as e:
        return f"❌ خطأ في البحث: {str(e)}", ""

def semantic_search(query, vectorizer, df, top_k=5):
    """Enhanced semantic search with better filtering"""
    try:
        if vectorizer is None or df.empty:
            return pd.DataFrame(), []
        
        clean_query = remove_tashkeel(query.strip())
        if len(clean_query) < 2:
            return pd.DataFrame(), []
        
        query_vector = vectorizer.transform([clean_query])
        tfidf_matrix = vectorizer.transform(df['clean_text'])
        similarities = manual_cosine_similarity(query_vector, tfidf_matrix)
        
        # Enhanced filtering with adaptive threshold
        min_threshold = max(0.05, similarities.max() * 0.3)
        valid_indices = np.where(similarities >= min_threshold)[0]
        
        if len(valid_indices) == 0:
            return pd.DataFrame(), []
        
        # Sort and get top k
        sorted_indices = valid_indices[similarities[valid_indices].argsort()[::-1]]
        top_indices = sorted_indices[:top_k]
        top_similarities = similarities[top_indices]
        
        result_df = df.iloc[top_indices].copy()
        return result_df, top_similarities.tolist()
        
    except Exception as e:
        st.error(f"❌ خطأ في البحث الدلالي: {e}")
        return pd.DataFrame(), []

def find_similar_adhkar(adhkar_text, vectorizer, df, top_k=3):
    """Enhanced similar adhkar finder"""
    try:
        if vectorizer is None or df.empty:
            return pd.DataFrame(), []
        
        clean_text = remove_tashkeel(adhkar_text)
        current_indices = df[df['clean_text'].str.contains(clean_text[:50], na=False)].index
        
        if len(current_indices) == 0:
            return pd.DataFrame(), []
        
        current_vector = vectorizer.transform([clean_text])
        tfidf_matrix = vectorizer.transform(df['clean_text'])
        similarities = manual_cosine_similarity(current_vector, tfidf_matrix)
        
        # Remove current adhkar
        for idx in current_indices:
            if idx < len(similarities):
                similarities[idx] = -1
        
        # Get top similar with better threshold
        valid_mask = similarities > 0.1
        if not np.any(valid_mask):
            return pd.DataFrame(), []
        
        valid_indices = np.where(valid_mask)[0]
        valid_similarities = similarities[valid_indices]
        
        # Sort and select top k
        sorted_order = valid_similarities.argsort()[::-1]
        top_indices = valid_indices[sorted_order[:top_k]]
        top_similarities = valid_similarities[sorted_order[:top_k]]
        
        result_df = df.iloc[top_indices].copy()
        return result_df, top_similarities.tolist()
        
    except Exception as e:
        st.error(f"❌ خطأ في العثور على أذكار مشابهة: {e}")
        return pd.DataFrame(), []

def get_time_based_greeting():
    """Enhanced time-based greeting with more periods"""
    current_hour = datetime.now().hour
    
    if 5 <= current_hour < 12:
        return "🌅 صباح الخير والبركة - أذكار الصباح", "morning", "var(--accent-500)"
    elif 12 <= current_hour < 15:
        return "☀️ ظهر مبارك - أذكار الظهيرة", "afternoon", "var(--primary-500)"
    elif 15 <= current_hour < 18:
        return "🌤️ عصر طيب - أذكار العصر", "late_afternoon", "var(--secondary-500)"
    elif 18 <= current_hour < 21:
        return "🌆 مساء الخير والنور - أذكار المساء", "evening", "var(--accent-600)"
    elif 21 <= current_hour < 23:
        return "🌙 ليلة مباركة - أذكار الليل", "night", "var(--secondary-700)"
    else:
        return "⭐ تصبح على خير - أذكار النوم", "sleep", "var(--primary-700)"

def initialize_session_state():
    """Enhanced session state initialization"""
    defaults = {
        'counter': 0,
        'daily_adhkar_count': 0,
        'weekly_adhkar_count': 0,
        'favorite_adhkar': [],
        'reading_streak': 0,
        'last_date': datetime.now().date(),
        'last_week': datetime.now().isocalendar()[1],
        'active_tab': 'search',
        'search_history': [],
        'user_preferences': {
            'font_size': 'medium',
            'theme': 'light',
            'notifications': True
        }
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Reset counters for new day/week
    current_date = datetime.now().date()
    current_week = datetime.now().isocalendar()[1]
    
    if st.session_state.last_date != current_date:
        if st.session_state.last_date == current_date - pd.Timedelta(days=1):
            st.session_state.reading_streak += 1
        else:
            st.session_state.reading_streak = 0
        
        st.session_state.daily_adhkar_count = 0
        st.session_state.last_date = current_date
    
    if st.session_state.last_week != current_week:
        st.session_state.weekly_adhkar_count = 0
        st.session_state.last_week = current_week

def get_category_info(category):
    """Enhanced category information with icons and colors"""
    category_info = {
        'morning': {
            'name': 'أذكار الصباح',
            'icon': '🌅',
            'color': 'accent',
            'description': 'أذكار تُقال عند بداية النهار'
        },
        'evening': {
            'name': 'أذكار المساء', 
            'icon': '🌆',
            'color': 'secondary',
            'description': 'أذكار تُقال عند نهاية النهار'
        },
        'general': {
            'name': 'أذكار عامة',
            'icon': '📿',
            'color': 'primary',
            'description': 'أذكار يمكن قولها في أي وقت'
        },
        'istighfar': {
            'name': 'استغفار وتوبة',
            'icon': '🤲',
            'color': 'success',
            'description': 'أذكار الاستغفار والتوبة'
        },
        'protection': {
            'name': 'حماية وأمان',
            'icon': '🛡️',
            'color': 'error',
            'description': 'أذكار الحماية من الشر'
        }
    }
    
    return category_info.get(category, {
        'name': category,
        'icon': '📖',
        'color': 'primary',
        'description': 'فئة من الأذكار'
    })

def display_enhanced_card(adhkar_row, similarity_score=None, is_similar=False, card_id=None):
    """Enhanced adhkar card with premium styling"""
    category_info = get_category_info(adhkar_row['category'])
    is_favorite = adhkar_row['text'] in st.session_state.favorite_adhkar
    
    # Generate unique key if not provided
    if card_id is None:
        card_id = f"card_{hash(adhkar_row['text'][:50])}"
    
    # Create premium card container
    with st.container():
        # Card styling with glassmorphism
        st.markdown(f"""
        <div class="premium-card {'card-glow' if is_similar else ''}">
            <div style="padding: 2rem;">
        """, unsafe_allow_html=True)
        
        # Header section
        col_badge, col_score, col_fav = st.columns([2, 1, 1])
        
        with col_badge:
            st.markdown(f"""
            <div class="premium-badge badge-{category_info['color']}">
                <span style="font-size: 1.1rem;">{category_info['icon']}</span>
                <span>{category_info['name']}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col_score:
            if similarity_score is not None:
                percentage = int(similarity_score * 100)
                color = "var(--success-500)" if percentage > 70 else "var(--warning-500)" if percentage > 40 else "var(--error-500)"
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, {color}20, {color}10);
                    color: {color};
                    padding: 6px 12px;
                    border-radius: var(--radius-3xl);
                    text-align: center;
                    font-weight: 600;
                    border: 1px solid {color}30;
                ">
                    🎯 {percentage}%
                </div>
                """, unsafe_allow_html=True)
        
        with col_fav:
            fav_icon = "❤️" if is_favorite else "🤍"
            st.markdown(f"""
            <div style="text-align: center; font-size: 1.5rem; cursor: pointer;">
                {fav_icon}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Arabic text with enhanced styling
        st.markdown(f"""
        <div class="premium-arabic">
            {adhkar_row['text']}
        </div>
        """, unsafe_allow_html=True)
        
        # Action buttons with enhanced styling
        st.markdown("---")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📖 قرأت", key=f"read_{card_id}", use_container_width=True):
                st.session_state.counter += 1
                st.session_state.daily_adhkar_count += 1
                st.session_state.weekly_adhkar_count += 1
                
                # Celebration for milestones
                if st.session_state.daily_adhkar_count % 10 == 0:
                    st.balloons()
                    st.success(f"🎉 مبارك! قرأت {st.session_state.daily_adhkar_count} ذكر اليوم!")
                else:
                    st.success("✅ تم احتساب القراءة!")
        
        with col2:
            fav_text = "💔 إزالة" if is_favorite else "❤️ أضف للمفضلة"
            button_class = "btn-favorite" if not is_favorite else "btn-ghost"
            
            if st.button(fav_text, key=f"fav_{card_id}", use_container_width=True):
                if is_favorite:
                    st.session_state.favorite_adhkar.remove(adhkar_row['text'])
                    st.success("💔 تم إزالة الذكر من المفضلة")
                else:
                    st.session_state.favorite_adhkar.append(adhkar_row['text'])
                    st.success("❤️ تم إضافة الذكر للمفضلة!")
                st.rerun()
        
        with col3:
            if SKLEARN_AVAILABLE and st.button("🔍 مشابه", key=f"similar_{card_id}", use_container_width=True):
                st.session_state.current_adhkar_for_similarity = adhkar_row['text']
                st.session_state.show_similar = True
                st.rerun()
        
        with col4:
            if st.button("📋 نسخ", key=f"copy_{card_id}", use_container_width=True):
                st.code(adhkar_row['text'], language="text")
                st.success("📋 تم نسخ النص!")
        
        # Close card div
        st.markdown("</div></div>", unsafe_allow_html=True)

def show_installation_guide():
    """Enhanced installation guide with better styling"""
    st.markdown("""
    <div class="premium-greeting">
        <div class="greeting-content">
            <h3 style="color: var(--primary-700); margin-bottom: 1rem;">🛠️ دليل التثبيت للميزات الذكية</h3>
            <p style="color: var(--gray-700); margin-bottom: 1.5rem;">
                لتفعيل جميع الميزات الذكية، يرجى تثبيت المكتبات المطلوبة:
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Installation commands in tabs
    tab1, tab2, tab3 = st.tabs(["🚀 تثبيت سريع", "📋 متطلبات كاملة", "☁️ Streamlit Cloud"])
    
    with tab1:
        st.code("""
# تثبيت المكتبات الأساسية
pip install scikit-learn joblib

# تشغيل التطبيق
streamlit run app.py
        """, language="bash")
    
    with tab2:
        st.code("""
# تثبيت جميع المتطلبات
pip install streamlit>=1.28.0 pandas>=1.5.0 numpy>=1.24.0 scikit-learn>=1.3.0 joblib>=1.3.0

# أو استخدام requirements.txt
pip install -r requirements.txt
        """, language="bash")
    
    with tab3:
        st.code("""
# محتوى ملف requirements.txt للـ Streamlit Cloud
streamlit>=1.28.0
pandas>=1.5.0  
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
        """, language="text")
    
    # Feature comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🟢 الميزات المتاحة حالياً:**
        - ✅ عرض الأذكار الأساسية
        - ✅ البحث التقليدي
        - ✅ حفظ المفضلة
        - ✅ إحصائيات أساسية
        """)
    
    with col2:
        st.markdown("""
        **🔵 الميزات بعد التثبيت:**
        - 🤖 البحث الذكي بالمعنى
        - 🎯 اقتراحات مخصصة  
        - 🔍 العثور على أذكار مشابهة
        - 📊 تحليلات متقدمة
        """)

def main():
    """Enhanced main application with premium features"""
    # Show dependency warnings
    if not SKLEARN_AVAILABLE:
        st.warning("⚠️ scikit-learn غير مثبت. الميزات الذكية معطلة. للتثبيت: pip install scikit-learn")
    
    if not JOBLIB_AVAILABLE:
        st.info("ℹ️ joblib غير مثبت. سيتم استخدام pickle كبديل. للتثبيت: pip install joblib")
    
    # Initialize enhanced session state
    initialize_session_state()
    
    # Load data and model with enhanced error handling
    vectorizer, df = load_model_and_vectorizer()
    
    if df.empty:
        df = load_data()
        if df.empty:
            st.error("❌ لا يمكن تحميل البيانات. يرجى التأكد من وجود ملف البيانات.")
            st.stop()
    
    # Enhanced header with dynamic content
    greeting, time_period, time_color = get_time_based_greeting()
    ai_status = "🤖 مفعل" if (SKLEARN_AVAILABLE and vectorizer is not None) else "❌ غير متاح"
    
    st.markdown(f"""
    <div class="premium-header">
        <div class="header-content">
            <div class="header-logo">
                <img src="https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Mosque/3D/mosque_3d.png" 
                     alt="Islamic Adhkar" 
                     style="height: 100px; width: auto;">
            </div>
            <h1 class="header-title">أذكار المسلم الذكي</h1>
            <p class="header-subtitle">Islamic Adhkar with AI-Powered Search & Recommendations</p>
            <div class="ai-status">
                <span>🤖</span>
                <span>الذكاء الاصطناعي: {ai_status}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced time-based greeting
    st.markdown(f"""
    <div class="premium-greeting">
        <div class="greeting-content">
            <h2 class="greeting-title">{greeting}</h2>
            <p style="color: var(--gray-600); font-size: 1.1rem;">
                بارك الله في وقتك، واجعل أذكارك نوراً في قلبك
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced navigation tabs
    st.markdown('<div class="premium-tabs">', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("🔍 البحث الذكي", key="tab_search", use_container_width=True):
            st.session_state.active_tab = 'search'
    
    with col2:
        if st.button("🤖 الذكاء الاصطناعي", key="tab_ai", use_container_width=True):
            st.session_state.active_tab = 'ai'
    
    with col3:
        if st.button("❤️ المفضلة", key="tab_favorites", use_container_width=True):
            st.session_state.active_tab = 'favorites'
    
    with col4:
        if st.button("📊 الإحصائيات", key="tab_stats", use_container_width=True):
            st.session_state.active_tab = 'stats'
    
    with col5:
        if st.button("ℹ️ حول التطبيق", key="tab_about", use_container_width=True):
            st.session_state.active_tab = 'about'
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main content area based on active tab
    if st.session_state.active_tab == 'search':
        show_search_tab(df, vectorizer)
    elif st.session_state.active_tab == 'ai':
        show_ai_tab(df, vectorizer)
    elif st.session_state.active_tab == 'favorites':
        show_favorites_tab(df, vectorizer)
    elif st.session_state.active_tab == 'stats':
        show_stats_tab(df)
    elif st.session_state.active_tab == 'about':
        show_about_tab(df)

def show_search_tab(df, vectorizer):
    """Enhanced search tab with traditional and smart search"""
    st.markdown("### 🔍 البحث في الأذكار")
    
    # Search input with enhanced styling
    search_query = st.text_input(
        "",
        placeholder="ابحث في الأذكار بالكلمات المفتاحية...",
        help="ابحث عن الأذكار باستخدام أي كلمة أو عبارة",
        key="traditional_search",
        label_visibility="collapsed"
    )
    
    # Search filters
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        selected_categories = st.multiselect(
            "فلترة حسب الفئة",
            options=df['category'].unique(),
            default=[],
            key="category_filter"
        )
    with col2:
        sort_by = st.selectbox(
            "ترتيب النتائج",
            ["الافتراضي", "الأقصر أولاً", "الأطول أولاً"],
            key="sort_results"
        )
    with col3:
        max_results = st.selectbox(
            "عدد النتائج",
            [10, 20, 50, 100],
            index=1,
            key="max_results"
        )
    
    # Perform search
    if search_query:
        with st.spinner("🔍 جاري البحث..."):
            # Filter dataframe
            filtered_df = df.copy()
            
            if selected_categories:
                filtered_df = filtered_df[filtered_df['category'].isin(selected_categories)]
            
            # Search in text
            search_mask = filtered_df['text'].str.contains(search_query, case=False, na=False)
            results_df = filtered_df[search_mask]
            
            # Sort results
            if sort_by == "الأقصر أولاً":
                results_df = results_df.sort_values('text', key=lambda x: x.str.len())
            elif sort_by == "الأطول أولاً":
                results_df = results_df.sort_values('text', key=lambda x: x.str.len(), ascending=False)
            
            # Limit results
            results_df = results_df.head(max_results)
            
            if not results_df.empty:
                st.success(f"🎯 تم العثور على {len(results_df)} نتيجة")
                
                for idx, (_, row) in enumerate(results_df.iterrows()):
                    display_enhanced_card(row, card_id=f"search_{idx}")
            else:
                st.markdown("""
                <div class="premium-empty">
                    <div class="empty-icon">🔍</div>
                    <h3>لا توجد نتائج</h3>
                    <p>جرب كلمات مختلفة أو قم بتوسيع معايير البحث</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Quick category browsing
    if not search_query:
        st.markdown("### 📚 تصفح حسب الفئات")
        
        categories = df['category'].unique()
        cols = st.columns(min(len(categories), 3))
        
        for i, category in enumerate(categories):
            with cols[i % 3]:
                category_info = get_category_info(category)
                count = len(df[df['category'] == category])
                
                if st.button(
                    f"{category_info['icon']} {category_info['name']}\n({count} ذكر)",
                    key=f"cat_{category}",
                    use_container_width=True
                ):
                    category_adhkar = df[df['category'] == category].head(10)
                    st.markdown(f"### {category_info['icon']} {category_info['name']}")
                    st.caption(category_info['description'])
                    
                    for idx, (_, row) in enumerate(category_adhkar.iterrows()):
                        display_enhanced_card(row, card_id=f"cat_{category}_{idx}")

def show_ai_tab(df, vectorizer):
    """Enhanced AI tab with advanced features"""
    if not SKLEARN_AVAILABLE or vectorizer is None:
        st.markdown("""
        <div class="premium-empty">
            <div class="empty-icon">🤖</div>
            <h3>الميزات الذكية غير متاحة</h3>
            <p>يتطلب تثبيت scikit-learn لتفعيل الميزات الذكية</p>
        </div>
        """, unsafe_allow_html=True)
        show_installation_guide()
        return
    
    st.markdown("### 🤖 البحث الذكي بالذكاء الاصطناعي")
    
    # Enhanced unified search
    st.markdown("#### 🧠 البحث الذكي الموحد")
    
    col_search, col_button = st.columns([4, 1])
    
    with col_search:
        search_query = st.text_input(
            "",
            placeholder="ابحث بذكاء: اكتب حالتك، مشاعرك، أو ما تحتاجه...",
            help="مثال: 'أشعر بالقلق'، 'أريد الحماية'، 'أدعية للوالدين'",
            key="ai_search",
            label_visibility="collapsed"
        )
    
    with col_button:
        search_pressed = st.button("🤖 بحث ذكي", key="ai_search_btn", use_container_width=True)
    
    # Advanced search options
    with st.expander("⚙️ خيارات البحث المتقدمة", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            search_mode = st.selectbox(
                "نوع البحث",
                ["ذكي شامل", "بحث دلالي", "البحث عن دعاء مناسب", "تحليل المشاعر"],
                key="ai_search_mode"
            )
        
        with col2:
            search_depth = st.slider("عدد النتائج", 3, 15, 5, key="ai_depth")
        
        with col3:
            min_similarity = st.slider("دقة التشابه", 0.05, 0.8, 0.15, 0.05, key="ai_similarity")
        
        with col4:
            include_similar = st.checkbox("إظهار الأذكار المشابهة", value=True, key="include_similar")
    
    # Perform AI search
    if search_query and (search_pressed or search_query):
        with st.spinner("🤖 جاري التحليل الذكي..."):
            
            if search_mode == "البحث عن دعاء مناسب":
                category, similar_text = find_similar_dua(search_query, vectorizer, df)
                if similar_text:
                    st.success(f"✨ تم العثور على دعاء مناسب في فئة: **{category}**")
                    result_row = pd.Series({
                        'text': similar_text,
                        'category': category
                    }, name='dua_result')
                    display_enhanced_card(result_row, card_id="dua_result", is_similar=True)
                else:
                    st.info(category)
            
            elif search_mode == "بحث دلالي":
                semantic_results, similarities = semantic_search(
                    search_query, vectorizer, df, top_k=search_depth
                )
                
                if not semantic_results.empty:
                    valid_results = []
                    for idx, sim in enumerate(similarities):
                        if sim >= min_similarity:
                            valid_results.append((semantic_results.iloc[idx], sim))
                    
                    if valid_results:
                        st.success(f"🎯 تم العثور على {len(valid_results)} نتيجة ذكية")
                        
                        for idx, (row, sim) in enumerate(valid_results):
                            display_enhanced_card(row, similarity_score=sim, 
                                                card_id=f"semantic_{idx}", is_similar=True)
                    else:
                        st.warning("لم يتم العثور على نتائج تتجاوز حد التشابه المحدد")
                else:
                    st.info("لم يتم العثور على نتائج. جرب كلمات مختلفة.")
            
            elif search_mode == "تحليل المشاعر":
                # Emotion-based search
                emotion_keywords = {
                    'قلق': ['حماية', 'أمان', 'protection'],
                    'حزن': ['صبر', 'تسلية', 'عزاء'],
                    'خوف': ['حماية', 'أمان', 'استعاذة'],
                    'شكر': ['حمد', 'شكر', 'general'],
                    'توبة': ['استغفار', 'توبة', 'istighfar']
                }
                
                detected_emotion = None
                for emotion, keywords in emotion_keywords.items():
                    if any(keyword in search_query for keyword in keywords):
                        detected_emotion = emotion
                        break
                
                if detected_emotion:
                    st.info(f"🎭 تم اكتشاف المشاعر: **{detected_emotion}**")
                
                # Continue with semantic search
                semantic_results, similarities = semantic_search(
                    search_query, vectorizer, df, top_k=search_depth
                )
                
                if not semantic_results.empty:
                    for idx, (_, row) in enumerate(semantic_results.iterrows()):
                        if similarities[idx] >= min_similarity:
                            display_enhanced_card(row, similarity_score=similarities[idx],
                                                card_id=f"emotion_{idx}", is_similar=True)
            
            else:  # "ذكي شامل"
                all_results = []
                
                # Semantic search
                semantic_results, semantic_similarities = semantic_search(
                    search_query, vectorizer, df, top_k=search_depth//2
                )
                
                for idx, (_, row) in enumerate(semantic_results.iterrows()):
                    if semantic_similarities[idx] >= min_similarity:
                        all_results.append((row, semantic_similarities[idx], "بحث دلالي"))
                
                # Dua finder
                category, similar_text = find_similar_dua(search_query, vectorizer, df)
                if similar_text:
                    is_duplicate = any(result[0]['text'] == similar_text for result in all_results)
                    if not is_duplicate:
                        dua_row = pd.Series({
                            'text': similar_text,
                            'category': category
                        }, name='dua_comprehensive')
                        all_results.append((dua_row, 0.95, "دعاء مناسب"))
                
                if all_results:
                    st.success(f"🎯 تم العثور على {len(all_results)} نتيجة شاملة")
                    
                    # Sort by similarity
                    all_results.sort(key=lambda x: x[1], reverse=True)
                    
                    for idx, (row, similarity, search_type) in enumerate(all_results):
                        st.markdown(f"**🔍 نوع البحث:** {search_type}")
                        display_enhanced_card(row, 
                                            similarity_score=similarity if similarity < 1 else None,
                                            card_id=f"comprehensive_{idx}", is_similar=True)
                else:
                    st.info("لم يتم العثور على نتائج مناسبة. جرب عبارات مختلفة.")
    
    # Quick AI search suggestions
    st.markdown("### 🚀 اقتراحات البحث الذكي")
    
    suggestions = [
        "أشعر بالقلق والتوتر", "أريد الحماية من الحسد", "أدعية للوالدين والأهل",
        "الاستغفار والتوبة", "الحمد والشكر لله", "دعاء للمريض والشفاء",
        "طلب الهداية والتوفيق", "أذكار قبل النوم", "البركة في الرزق"
    ]
    
    cols = st.columns(3)
    for i, suggestion in enumerate(suggestions):
        with cols[i % 3]:
            if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                with st.spinner(f"🔍 جاري البحث عن: {suggestion}"):
                    semantic_results, similarities = semantic_search(
                        suggestion, vectorizer, df, top_k=3
                    )
                    if not semantic_results.empty:
                        st.markdown(f"### 🎯 نتائج: {suggestion}")
                        for idx, (_, row) in enumerate(semantic_results.iterrows()):
                            display_enhanced_card(row, similarity_score=similarities[idx],
                                                card_id=f"quick_{i}_{idx}", is_similar=True)

def show_favorites_tab(df, vectorizer):
    """Enhanced favorites tab with smart features"""
    st.markdown("### ❤️ الأذكار المفضلة")
    
    if not st.session_state.favorite_adhkar:
        st.markdown("""
        <div class="premium-empty">
            <div class="empty-icon">❤️</div>
            <h3>لا توجد أذكار مفضلة</h3>
            <p>أضف أذكارك المفضلة من أقسام البحث لتظهر هنا</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Favorites statistics
    fav_count = len(st.session_state.favorite_adhkar)
    fav_categories = []
    
    for fav_text in st.session_state.favorite_adhkar:
        matching_rows = df[df['text'] == fav_text]
        if not matching_rows.empty:
            fav_categories.append(matching_rows.iloc[0]['category'])
    
    most_common_category = Counter(fav_categories).most_common(1)
    most_common_cat = most_common_category[0][0] if most_common_category else "غير محدد"
    
    # Display statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card primary">
            <div class="stat-number">{fav_count}</div>
            <div class="stat-label">إجمالي المفضلة</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        category_info = get_category_info(most_common_cat)
        st.markdown(f"""
        <div class="stat-card secondary">
            <div class="stat-number">{category_info['icon']}</div>
            <div class="stat-label">الفئة المفضلة<br>{category_info['name']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        unique_categories = len(set(fav_categories))
        st.markdown(f"""
        <div class="stat-card accent">
            <div class="stat-number">{unique_categories}</div>
            <div class="stat-label">فئات متنوعة</div>
        </div>
        """, unsafe_allow_html=True)
    
    # AI-powered suggestions
    if SKLEARN_AVAILABLE and vectorizer is not None and fav_count > 0:
        if st.button("🤖 اقتراحات ذكية بناءً على مفضلتك", key="smart_suggestions", use_container_width=True):
            with st.spinner("🤖 جاري تحليل تفضيلاتك..."):
                all_suggestions = []
                
                # Analyze favorite adhkar to find similar ones
                for fav_adhkar in st.session_state.favorite_adhkar[:5]:  # Limit to prevent slowness
                    similar_results, similarities = find_similar_adhkar(
                        fav_adhkar, vectorizer, df, top_k=3
                    )
                    
                    for idx, (_, row) in enumerate(similar_results.iterrows()):
                        if (row['text'] not in st.session_state.favorite_adhkar and 
                            similarities[idx] > 0.2):
                            all_suggestions.append((row, similarities[idx]))
                
                # Remove duplicates and sort by similarity
                unique_suggestions = {}
                for row, sim in all_suggestions:
                    text_key = row['text'][:100]  # Use first 100 chars as key
                    if text_key not in unique_suggestions or unique_suggestions[text_key][1] < sim:
                        unique_suggestions[text_key] = (row, sim)
                
                final_suggestions = list(unique_suggestions.values())
                final_suggestions.sort(key=lambda x: x[1], reverse=True)
                
                if final_suggestions:
                    st.markdown("### 🤖 اقتراحات ذكية مخصصة لك:")
                    st.caption("بناءً على تحليل أذكارك المفضلة")
                    
                    for idx, (row, sim) in enumerate(final_suggestions[:8]):
                        display_enhanced_card(row, similarity_score=sim, 
                                            card_id=f"suggestion_{idx}", is_similar=True)
                else:
                    st.info("لا توجد اقتراحات جديدة متاحة حالياً. جرب إضافة المزيد من الأذكار المتنوعة للمفضلة.")
    
    st.markdown("---")
    
    # Display favorite adhkar
    st.markdown("### 📚 مجموعتك المفضلة:")
    
    # Sort options for favorites
    col1, col2 = st.columns([2, 1])
    with col1:
        sort_favorites = st.selectbox(
            "ترتيب المفضلة",
            ["الأحدث أولاً", "الأقدم أولاً", "حسب الفئة", "حسب الطول"],
            key="sort_favorites"
        )
    
    with col2:
        if st.button("🗑️ مسح جميع المفضلة", key="clear_all_favorites"):
            st.session_state.favorite_adhkar = []
            st.success("تم مسح جميع الأذكار المفضلة")
            st.rerun()
    
    # Sort and display favorites
    favorites_with_data = []
    for i, adhkar_text in enumerate(st.session_state.favorite_adhkar):
        matching_rows = df[df['text'] == adhkar_text]
        if not matching_rows.empty:
            row = matching_rows.iloc[0]
            favorites_with_data.append((row, i))
        else:
            # Create temporary row for orphaned favorites
            temp_row = pd.Series({
                'text': adhkar_text,
                'category': 'مفضلة محفوظة'
            }, name=f'orphan_{i}')
            favorites_with_data.append((temp_row, i))
    
    # Apply sorting
    if sort_favorites == "الأقدم أولاً":
        favorites_with_data.reverse()
    elif sort_favorites == "حسب الفئة":
        favorites_with_data.sort(key=lambda x: x[0]['category'])
    elif sort_favorites == "حسب الطول":
        favorites_with_data.sort(key=lambda x: len(x[0]['text']))
    
    # Display sorted favorites
    for row, original_index in favorites_with_data:
        display_enhanced_card(row, card_id=f"fav_{original_index}")

def show_stats_tab(df):
    """Enhanced statistics tab with advanced analytics"""
    st.markdown("### 📊 الإحصائيات والتحليلات المتقدمة")
    
    # Overall statistics with enhanced styling
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card primary">
            <div class="stat-number">{len(df):,}</div>
            <div class="stat-label">إجمالي الأذكار</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card secondary">
            <div class="stat-number">{len(df['category'].unique())}</div>
            <div class="stat-label">عدد الفئات</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card accent">
            <div class="stat-number">{st.session_state.daily_adhkar_count}</div>
            <div class="stat-label">أذكار اليوم</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stat-card primary">
            <div class="stat-number">{len(st.session_state.favorite_adhkar)}</div>
            <div class="stat-label">المفضلة</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Personal achievement statistics
    st.markdown("### 🏆 إنجازاتك الشخصية")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📖 إجمالي القراءات",
            f"{st.session_state.counter:,}",
            delta=st.session_state.daily_adhkar_count
        )
    
    with col2:
        st.metric(
            "🔥 أيام متتالية",
            f"{st.session_state.reading_streak}",
            delta=1 if st.session_state.daily_adhkar_count > 0 else 0
        )
    
    with col3:
        weekly_target = 50
        weekly_progress = (st.session_state.weekly_adhkar_count / weekly_target) * 100
        st.metric(
            "📅 هدف الأسبوع",
            f"{weekly_progress:.1f}%",
            delta=f"{st.session_state.weekly_adhkar_count}/{weekly_target}"
        )
    
    with col4:
        if st.session_state.favorite_adhkar:
            avg_fav_length = np.mean([len(text) for text in st.session_state.favorite_adhkar])
            st.metric(
                "❤️ متوسط طول المفضلة",
                f"{avg_fav_length:.0f} حرف"
            )
        else:
            st.metric("❤️ المفضلة", "0")
    
    # Category analysis
    st.markdown("### 📈 تحليل الفئات")
    
    category_counts = df['category'].value_counts()
    
    # Create better visualization
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.bar_chart(category_counts, height=400)
    
    with col2:
        st.markdown("**🏆 أكثر الفئات شيوعاً:**")
        for i, (category, count) in enumerate(category_counts.head(5).items(), 1):
            category_info = get_category_info(category)
            percentage = (count / len(df)) * 100
            st.markdown(f"""
            <div style="
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 8px 16px;
                margin: 4px 0;
                background: linear-gradient(135deg, var(--primary-50), var(--secondary-50));
                border-radius: var(--radius-lg);
                border-right: 4px solid var(--primary-500);
            ">
                <span>{i}. {category_info['icon']} {category_info['name']}</span>
                <span><strong>{count}</strong> ({percentage:.1f}%)</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Text analysis
    st.markdown("### 📏 تحليل النصوص المتقدم")
    
    text_lengths = df['text'].str.len()
    word_counts = df['text'].str.split().str.len()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("متوسط الطول", f"{text_lengths.mean():.0f} حرف")
    
    with col2:
        st.metric("متوسط الكلمات", f"{word_counts.mean():.1f} كلمة")
    
    with col3:
        st.metric("أقصر نص", f"{text_lengths.min()} حرف")
    
    with col4:
        st.metric("أطول نص", f"{text_lengths.max()} حرف")
    
    # Length distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 توزيع أطوال النصوص (بالأحرف):**")
        length_bins = pd.cut(text_lengths, bins=10)
        length_dist = length_bins.value_counts().sort_index()
        st.bar_chart(length_dist)
    
    with col2:
        st.markdown("**📝 توزيع عدد الكلمات:**")
        word_bins = pd.cut(word_counts, bins=8)
        word_dist = word_bins.value_counts().sort_index()
        st.bar_chart(word_dist)
    
    # Reading patterns and achievements
    st.markdown("### 🎯 أنماط القراءة والإنجازات")
    
    # Calculate achievements
    achievements = []
    
    if st.session_state.counter >= 100:
        achievements.append("🥉 قارئ مبتدئ - 100 قراءة")
    if st.session_state.counter >= 500:
        achievements.append("🥈 قارئ متوسط - 500 قراءة")  
    if st.session_state.counter >= 1000:
        achievements.append("🥇 قارئ متقدم - 1000 قراءة")
    if st.session_state.reading_streak >= 7:
        achievements.append("🔥 مداوم أسبوعي - 7 أيام متتالية")
    if st.session_state.reading_streak >= 30:
        achievements.append("⭐ مداوم شهري - 30 يوم متتالية")
    if len(st.session_state.favorite_adhkar) >= 10:
        achievements.append("❤️ جامع أذكار - 10 مفضلة")
    if len(st.session_state.favorite_adhkar) >= 50:
        achievements.append("💎 خبير أذكار - 50 مفضلة")
    
    if achievements:
        st.markdown("**🏆 إنجازاتك:**")
        for achievement in achievements:
            st.success(achievement)
    else:
        st.info("📈 ابدأ رحلتك في قراءة الأذكار لتحصل على إنجازات!")
    
    # Progress towards next achievement
    next_targets = [
        (100, "قارئ مبتدئ"),
        (500, "قارئ متوسط"),
        (1000, "قارئ متقدم"),
        (2000, "قارئ خبير"),
        (5000, "قارئ محترف")
    ]
    
    for target, title in next_targets:
        if st.session_state.counter < target:
            progress = (st.session_state.counter / target) * 100
            remaining = target - st.session_state.counter
            st.progress(progress / 100)
            st.caption(f"🎯 التالي: {title} - باقي {remaining} قراءة ({progress:.1f}%)")
            break

def show_about_tab(df):
    """Enhanced about tab with comprehensive information"""
    st.markdown("## ℹ️ حول تطبيق أذكار المسلم الذكي")
    
    # Application overview with premium styling
    st.markdown(f"""
    <div class="premium-greeting">
        <div class="greeting-content">
            <h3 style="color: var(--primary-700); margin-bottom: 1rem;">🕌 تطبيق شامل للأذكار والأدعية</h3>
            <p style="color: var(--gray-700); font-size: 1.1rem; line-height: 1.6;">
                تطبيق متطور يجمع بين التراث الإسلامي العريق والتكنولوجيا الحديثة، 
                يحتوي على مجموعة واسعة من الأذكار والأدعية المأخوذة من القرآن الكريم والسنة النبوية الشريفة.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🌟 المميزات", "🤖 التقنيات", "📊 الإحصائيات", "🛠️ التطوير"])
    
    with tab1:
        st.markdown("### 🌟 مميزات التطبيق")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 📖 المميزات الأساسية:
            - ✅ **مكتبة شاملة**: أكثر من {count} ذكر ودعاء أصيل
            - 🔍 **بحث متقدم**: بحث تقليدي بالكلمات المفتاحية
            - 📚 **تصنيف ذكي**: تنظيم الأذكار حسب الأوقات والمناسبات
            - ⭐ **المفضلة**: حفظ أذكارك المفضلة للوصول السريع
            - 📊 **تتبع التقدم**: إحصائيات مفصلة عن قراءاتك
            - 🎯 **أهداف شخصية**: تحديد وتتبع أهدافك اليومية
            - 🏆 **نظام الإنجازات**: حوافز للمداومة على الأذكار
            - 📱 **تصميم حديث**: واجهة أنيقة ومتجاوبة
            """.format(count=len(df)))
        
        with col2:
            if SKLEARN_AVAILABLE:
                vocab_size = len(st.session_state.get('vectorizer_vocab', []))
                st.markdown(f"""
                #### 🤖 المميزات الذكية (مفعلة):
                - 🧠 **بحث ذكي بالمعنى**: فهم المقصد وراء كلماتك
                - 🎯 **العثور على الدعاء المناسب**: اكتب حالتك واحصل على الدعاء الأنسب
                - 🔍 **اكتشاف الأذكار المشابهة**: توسيع آفاق القراءة
                - 📊 **تحليل المشاعر**: تحليل ذكي للحالة النفسية
                - 🎁 **توصيات مخصصة**: اقتراحات بناءً على تفضيلاتك
                - 📈 **تحليل النصوص**: معالجة {vocab_size:,} كلمة مختلفة
                - 🔧 **خوارزميات متقدمة**: TF-IDF والتشابه الدلالي
                - 🚀 **أداء محسن**: استجابة سريعة ودقيقة
                """)
            else:
                st.markdown("""
                #### ❌ المميزات الذكية (غير متاحة):
                - يتطلب تثبيت المكتبات المطلوبة
                - راجع تبويب "🛠️ التطوير" للتفاصيل
                """)
    
    with tab2:
        st.markdown("### 🤖 التقنيات المستخدمة")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 🖥️ تقنيات الواجهة:
            - **Streamlit**: إطار عمل Python للتطبيقات التفاعلية
            - **HTML5 & CSS3**: تصميم حديث ومتجاوب
            - **JavaScript**: تفاعلات ديناميكية
            - **Google Fonts**: خطوط عربية أنيقة (Amiri, Noto Kufi)
            - **Responsive Design**: متوافق مع جميع الأجهزة
            - **Modern UI/UX**: تصميم مستوحى من أحدث التطبيقات
            """)
        
        with col2:
            st.markdown("""
            #### 🧠 تقنيات الذكاء الاصطناعي:
            - **scikit-learn**: مكتبة التعلم الآلي
            - **TF-IDF Vectorization**: تحويل النصوص لمتجهات رقمية
            - **Cosine Similarity**: قياس التشابه بين النصوص
            - **Natural Language Processing**: معالجة اللغة الطبيعية
            - **Text Preprocessing**: تنظيف وتحضير النصوص العربية
            - **Semantic Search**: البحث بالمعنى والدلالة
            """)
        
        st.markdown("#### 📚 مكتبات البيانات:")
        
        libraries_info = [
            ("pandas", "تحليل ومعالجة البيانات", "✅"),
            ("numpy", "العمليات الرياضية والمصفوفات", "✅"),
            ("scikit-learn", "الذكاء الاصطناعي والتعلم الآلي", "✅" if SKLEARN_AVAILABLE else "❌"),
            ("joblib", "حفظ وتحميل النماذج", "✅" if JOBLIB_AVAILABLE else "⚠️"),
            ("re", "معالجة التعبيرات النمطية", "✅"),
            ("datetime", "التعامل مع التواريخ والأوقات", "✅")
        ]
        
        for lib, desc, status in libraries_info:
            col1, col2, col3 = st.columns([1, 3, 1])
            with col1:
                st.code(lib)
            with col2:
                st.write(desc)
            with col3:
                st.write(status)
    
    with tab3:
        st.markdown("### 📊 إحصائيات التطبيق")
        
        # Calculate detailed statistics
        total_chars = df['text'].str.len().sum()
        total_words = df['text'].str.split().str.len().sum()
        avg_words_per_adhkar = total_words / len(df)
        
        unique_words = set()
        for text in df['text']:
            unique_words.update(text.split())
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 📈 إحصائيات المحتوى:
            """)
            
            stats_data = [
                ("📖 إجمالي الأذكار", f"{len(df):,}"),
                ("📚 عدد الفئات", f"{len(df['category'].unique())}"),
                ("📝 إجمالي الكلمات", f"{total_words:,}"),
                ("🔤 إجمالي الأحرف", f"{total_chars:,}"),
                ("💬 متوسط الكلمات لكل ذكر", f"{avg_words_per_adhkar:.1f}"),
                ("🗂️ الكلمات الفريدة", f"{len(unique_words):,}")
            ]
            
            for label, value in stats_data:
                st.metric(label, value)
        
        with col2:
            st.markdown("#### 📊 توزيع الفئات:")
            
            category_stats = df['category'].value_counts()
            for category, count in category_stats.items():
                category_info = get_category_info(category)
                percentage = (count / len(df)) * 100
                
                st.markdown(f"""
                <div style="
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    padding: 12px 16px;
                    margin: 8px 0;
                    background: linear-gradient(135deg, var(--primary-50), var(--secondary-50));
                    border-radius: var(--radius-lg);
                    border-right: 4px solid var(--primary-500);
                ">
                    <span>
                        <span style="font-size: 1.2rem; margin-right: 8px;">{category_info['icon']}</span>
                        <strong>{category_info['name']}</strong>
                    </span>
                    <span>
                        <span style="font-size: 1.1rem; font-weight: 600;">{count}</span>
                        <span style="color: var(--gray-500); margin-left: 8px;">({percentage:.1f}%)</span>
                    </span>
                </div>
                """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown("### 🛠️ دليل التطوير والتثبيت")
        
        if not SKLEARN_AVAILABLE:
            st.warning("⚠️ بعض المميزات معطلة - يرجى اتباع دليل التثبيت")
            show_installation_guide()
        else:
            st.success("✅ جميع المكتبات مثبتة ومفعلة!")
        
        st.markdown("#### 🚀 خطوات التشغيل السريع:")
        
        setup_steps = """
        ```bash
        # 1. استنساخ المشروع
        git clone [repository-url]
        cd adhkar-app
        
        # 2. إنشاء بيئة افتراضية
        python -m venv venv
        source venv/bin/activate  # Linux/Mac
        # أو
        venv\\Scripts\\activate  # Windows
        
        # 3. تثبيت المتطلبات
        pip install -r requirements.txt
        
        # 4. تشغيل التطبيق
        streamlit run app.py
        ```
        """
        
        st.code(setup_steps, language="bash")
        
        st.markdown("#### 📁 هيكل المشروع:")
        
        project_structure = """
        adhkar-app/
        ├── app.py                 # الملف الرئيسي للتطبيق
        ├── adhkar_df.csv          # قاعدة بيانات الأذكار
        ├── tfidf_vectorizer.pkl   # نموذج الذكاء الاصطناعي
        ├── requirements.txt       # متطلبات المشروع
        ├── README.md             # دليل الاستخدام
        └── assets/               # الملفات المساعدة
            ├── styles.css        # ملفات التنسيق
            └── scripts.js        # السكريبتات
        """
        
        st.code(project_structure, language="text")
        
        st.markdown("#### 🤝 المساهمة في التطوير:")
        
        st.markdown("""
        - 📧 **تواصل معنا**: لاقتراح تحسينات أو إضافات جديدة
        - 🐛 **الإبلاغ عن مشاكل**: ساعدنا في تحسين التطبيق
        - 💡 **اقتراح ميزات**: شاركنا أفكارك لتطوير التطبيق
        - 📖 **إضافة أذكار**: ساهم في إثراء المحتوى
        - 🌍 **الترجمة**: ساعد في دعم لغات أخرى
        """)
    
    # Footer with dua
    st.markdown("---")
    st.markdown(f"""
    <div class="premium-greeting">
        <div class="greeting-content">
            <h4 style="color: var(--primary-700); margin-bottom: 1rem;">🤲 دعاء</h4>
            <p style="
                font-family: 'Amiri', serif;
                font-size: 1.3rem;
                color: var(--gray-800);
                line-height: 1.8;
                text-align: center;
                margin-bottom: 1rem;
            ">
                "اللهم اجعل هذا العمل خالصاً لوجهك الكريم، وانفع به المسلمين في كل مكان، 
                واجعله في ميزان حسناتنا يوم القيامة"
            </p>
            <p style="color: var(--gray-600); text-align: center; font-style: italic;">
                <strong>تذكر:</strong> المداومة على القليل خير من الانقطاع عن الكثير
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Version and credits
    st.markdown("""
    <div style="
        text-align: center;
        padding: 2rem;
        color: var(--gray-500);
        font-size: 0.9rem;
        border-top: 1px solid var(--gray-200);
        margin-top: 2rem;
    ">
        <p><strong>إصدار التطبيق:</strong> 2.0 Premium Edition</p>
        <p><strong>آخر تحديث:</strong> {datetime.now().strftime('%Y-%m-%d')}</p>
        <p>تم التطوير بـ ❤️ للمسلمين في كل مكان</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np
from datasets import load_dataset
import os
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def remove_tashkeel(text):
    """Remove Arabic diacritics for better text processing"""
    if not isinstance(text, str):
        return ""
    tashkeel_pattern = re.compile(r'[\u064B-\u065F\u0670]')
    return tashkeel_pattern.sub('', text)

def preprocess_arabic_text(text):
    """Enhanced Arabic text preprocessing"""
    if not isinstance(text, str):
        return ""
    
    # Remove diacritics
    text = remove_tashkeel(text)
    
    # Normalize Arabic letters
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ي', 'ى', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

@st.cache_data
def load_quran_dataset_enhanced():
    """Enhanced Quran dataset loading with better error handling and caching"""
    
    try:
        # Show loading progress
        with st.spinner("📖 جاري تحميل بيانات القرآن والتفسير من Hugging Face..."):
            
            # Set cache directory to avoid permission issues
            cache_dir = "./hf_cache"
            os.makedirs(cache_dir, exist_ok=True)
            
            # Load dataset with custom cache directory
            dataset = load_dataset(
                "MohamedRashad/Quran-Tafseer",
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            
            # Convert to pandas DataFrame
            if 'train' in dataset:
                df = pd.DataFrame(dataset['train'])
            else:
                # If there's no 'train' split, use the first available split
                split_name = list(dataset.keys())[0]
                df = pd.DataFrame(dataset[split_name])
                st.info(f"Using dataset split: {split_name}")
            
            # Display initial dataset info
            st.success(f"✅ تم تحميل {len(df)} سجل من Hugging Face")
            
            # Show dataset columns for debugging
            st.info(f"أعمدة البيانات المتاحة: {list(df.columns)}")
            
            # Handle different possible column names
            column_mapping = {
                'ayah': 'text',
                'verse': 'text', 
                'arabic': 'text',
                'quran_text': 'text',
                'verse_text': 'text',
                'surah_name': 'surah',
                'chapter': 'surah',
                'chapter_name': 'surah',
                'verse_number': 'ayah_number',
                'ayah_num': 'ayah_number',
                'verse_id': 'ayah_number',
                'tafsir': 'tafseer',
                'interpretation': 'tafseer',
                'explanation': 'tafseer'
            }
            
            # Apply column mapping
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns and new_col not in df.columns:
                    df[new_col] = df[old_col]
                    st.info(f"تم تعيين العمود '{old_col}' إلى '{new_col}'")
            
            # Ensure required columns exist
            if 'text' not in df.columns:
                # Try to find any column that might contain the Quranic text
                possible_text_cols = ['ayah', 'verse', 'arabic', 'quran_text', 'verse_text']
                text_col = None
                for col in possible_text_cols:
                    if col in df.columns:
                        text_col = col
                        break
                
                if text_col:
                    df['text'] = df[text_col]
                    st.success(f"تم استخدام العمود '{text_col}' كنص الآيات")
                else:
                    st.error("❌ لا يمكن العثور على عمود يحتوي على نص الآيات")
                    return pd.DataFrame()
            
            # Handle missing columns with defaults
            if 'surah' not in df.columns:
                df['surah'] = 'غير محدد'
                st.warning("⚠️ عمود السورة غير موجود - تم استخدام قيمة افتراضية")
            
            if 'ayah_number' not in df.columns:
                df['ayah_number'] = range(1, len(df) + 1)
                st.warning("⚠️ عمود رقم الآية غير موجود - تم إنشاء ترقيم تلقائي")
            
            if 'tafseer' not in df.columns:
                df['tafseer'] = ''
                st.warning("⚠️ عمود التفسير غير موجود - سيتم البحث في النص فقط")
            
            # Clean and preprocess text
            df['text'] = df['text'].fillna('').astype(str)
            df['clean_text'] = df['text'].apply(preprocess_arabic_text)
            
            # Preprocess tafseer if available
            if 'tafseer' in df.columns:
                df['tafseer'] = df['tafseer'].fillna('').astype(str)
                df['clean_tafseer'] = df['tafseer'].apply(preprocess_arabic_text)
            
            # Remove empty rows
            initial_count = len(df)
            df = df[df['clean_text'].str.strip() != ''].copy()
            df = df.dropna(subset=['text']).copy()
            final_count = len(df)
            
            if initial_count != final_count:
                st.info(f"تم إزالة {initial_count - final_count} سجل فارغ")
            
            # Reset index
            df = df.reset_index(drop=True)
            
            # Show sample data for verification
            st.markdown("### 🔍 عينة من البيانات المحملة:")
            sample_size = min(3, len(df))
            for i in range(sample_size):
                row = df.iloc[i]
                with st.expander(f"الآية {i+1}: {row.get('surah', 'غير محدد')} - آية {row.get('ayah_number', i+1)}"):
                    st.write(f"**النص:** {row['text'][:100]}...")
                    if row.get('tafseer') and row['tafseer'].strip():
                        st.write(f"**التفسير:** {row['tafseer'][:100]}...")
            
            # Final statistics
            st.success(f"""
            ✅ **تم تحميل البيانات بنجاح:**
            - إجمالي الآيات: {len(df):,}
            - السور المختلفة: {len(df['surah'].unique()) if 'surah' in df.columns else 'غير محدد'}
            - آيات بتفسير: {len(df[df['tafseer'].str.strip() != '']) if 'tafseer' in df.columns else 0}
            - متوسط طول الآية: {df['text'].str.len().mean():.0f} حرف
            """)
            
            return df
            
    except Exception as e:
        st.error(f"❌ خطأ في تحميل البيانات من Hugging Face: {str(e)}")
        
        # Provide detailed error information
        st.markdown(f"""
        **تفاصيل الخطأ:**
        - نوع الخطأ: {type(e).__name__}
        - الرسالة: {str(e)}
        
        **الحلول المقترحة:**
        1. تأكد من الاتصال بالإنترنت
        2. تحقق من صحة اسم المجموعة: `MohamedRashad/Quran-Tafseer`
        3. جرب تثبيت/تحديث مكتبة datasets: `pip install --upgrade datasets`
        4. امنح صلاحيات الكتابة للمجلد الحالي
        """)
        
        # Try to load a fallback local dataset if available
        return load_local_fallback()

def load_local_fallback():
    """Fallback to create sample data if Hugging Face is not available"""
    st.warning("🔄 جاري استخدام بيانات تجريبية...")
    
    # Create comprehensive sample data for demonstration
    sample_data = {
        'text': [
            'بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ',
            'الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ',
            'الرَّحْمَٰنِ الرَّحِيمِ',
            'مَالِكِ يَوْمِ الدِّينِ',
            'إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ',
            'اهْدِنَا الصِّرَاطَ الْمُسْتَقِيمَ',
            'صِرَاطَ الَّذِينَ أَنْعَمْتَ عَلَيْهِمْ غَيْرِ الْمَغْضُوبِ عَلَيْهِمْ وَلَا الضَّالِّينَ',
            'الم',
            'ذَٰلِكَ الْكِتَابُ لَا رَيْبَ ۛ فِيهِ ۛ هُدًى لِّلْمُتَّقِينَ',
            'الَّذِينَ يُؤْمِنُونَ بِالْغَيْبِ وَيُقِيمُونَ الصَّلَاةَ وَمِمَّا رَزَقْنَاهُمْ يُنفِقُونَ'
        ],
        'surah': ['الفاتحة'] * 7 + ['البقرة'] * 3,
        'ayah_number': [1, 2, 3, 4, 5, 6, 7, 1, 2, 3],
        'tafseer': [
            'بسم الله الرحمن الرحيم: افتتاح كل سورة بحمد الله والثناء عليه، وهي آية من القرآن في أول كل سورة',
            'الحمد لله: الثناء على الله بصفاته الجميلة وأفعاله الحميدة، وهو سبحانه المستحق للحمد والثناء',
            'الرحمن الرحيم: من أسماء الله الحسنى، الرحمن ذو الرحمة الواسعة، والرحيم ذو الرحمة الخاصة بالمؤمنين',
            'مالك يوم الدين: الله هو المالك المتصرف يوم القيامة، يوم الجزاء والحساب',
            'إياك نعبد وإياك نستعين: التوحيد في العبادة والاستعانة، فنعبد الله وحده ونستعين به وحده',
            'اهدنا الصراط المستقيم: دعاء بالهداية إلى الطريق المستقيم وهو الإسلام',
            'صراط الذين أنعمت عليهم: طريق الأنبياء والصديقين والشهداء والصالحين، غير المغضوب عليهم وهم اليهود، ولا الضالين وهم النصارى',
            'الم: من الحروف المقطعة في أوائل السور، والله أعلم بمرادها',
            'ذلك الكتاب لا ريب فيه: هذا القرآن الكريم لا شك فيه ولا ريب، وهو هداية للمتقين',
            'الذين يؤمنون بالغيب: صفة المتقين أنهم يؤمنون بما غاب عنهم مما أخبر الله به، ويقيمون الصلاة وينفقون مما رزقهم الله'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    df['clean_text'] = df['text'].apply(preprocess_arabic_text)
    df['clean_tafseer'] = df['tafseer'].apply(preprocess_arabic_text)
    
    st.warning("⚠️ يتم استخدام بيانات تجريبية محدودة. لتحميل البيانات الكاملة، تأكد من تثبيت مكتبة datasets والاتصال بالإنترنت")
    
    return df

def test_dataset_loading():
    """Test function to verify dataset loading"""
    st.markdown("## 🧪 اختبار تحميل البيانات")
    
    if st.button("🔄 اختبار تحميل المجموعة من Hugging Face"):
        df = load_quran_dataset_enhanced()
        
        if not df.empty:
            st.success("✅ تم تحميل البيانات بنجاح!")
            
            # Show detailed information
            st.markdown("### 📊 معلومات مفصلة عن البيانات:")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("عدد الآيات", len(df))
                st.metric("عدد الأعمدة", len(df.columns))
            
            with col2:
                unique_surahs = len(df['surah'].unique()) if 'surah' in df.columns else 0
                st.metric("عدد السور", unique_surahs)
                has_tafseer = len(df[df['tafseer'].str.strip() != '']) if 'tafseer' in df.columns else 0
                st.metric("آيات بتفسير", has_tafseer)
            
            with col3:
                avg_length = df['text'].str.len().mean() if 'text' in df.columns else 0
                st.metric("متوسط طول الآية", f"{avg_length:.0f}")
                
                if 'tafseer' in df.columns:
                    avg_tafseer_length = df['tafseer'].str.len().mean()
                    st.metric("متوسط طول التفسير", f"{avg_tafseer_length:.0f}")
            
            # Show column information
            st.markdown("### 📋 الأعمدة المتاحة:")
            for col in df.columns:
                non_null_count = df[col].notna().sum()
                st.write(f"- **{col}**: {non_null_count}/{len(df)} قيمة غير فارغة")
            
            # Show data types
            st.markdown("### 🏷️ أنواع البيانات:")
            st.write(df.dtypes)
            
            # Show sample rows
            st.markdown("### 👀 عينة من البيانات:")
            st.dataframe(df.head())
            
        else:
            st.error("❌ فشل في تحميل البيانات")

# Usage example
if __name__ == "__main__":
    st.set_page_config(
        page_title="اختبار تحميل بيانات القرآن",
        page_icon="📖",
        layout="wide"
    )
    
    st.title("📖 اختبار تحميل بيانات القرآن والتفسير")
    
    test_dataset_loading()

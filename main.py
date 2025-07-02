import sounddevice as sd
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import speech_recognition as sr
from datetime import datetime
import difflib
import re
import os
import tempfile

# Verify SpeechRecognition installation
try:
    recognizer_test = sr.Recognizer()
    if not hasattr(recognizer_test, 'recognize_google'):
        print("⚠️ Warning: Google Speech Recognition tidak tersedia")
    else:
        print("✅ SpeechRecognition library OK")
except Exception as e:
    print(f"❌ SpeechRecognition error: {e}")
    print("Install dengan: pip install SpeechRecognition")

class VoiceReadinessChecker:
    def __init__(self, sample_rate=16000):
        """
        Inisialisasi Voice Readiness Checker dengan Speech Recognition
        """
        self.sample_rate = sample_rate
        self.recognizer = sr.Recognizer()
        
        print("🔧 Initializing with proven settings...")
        
        # Kalimat contoh yang pendek dan mudah
        self.sample_sentences = [
            "Saya siap kerja",
            "Keselamatan utama", 
            "Kondisi sehat",
            "Peralatan aman",
            "Tim komunikasi baik"
        ]
    
    def record_audio(self, duration=8):
        """Record audio untuk durasi tertentu (diperpanjang untuk kalimat lengkap)"""
        print(f"🎤 Recording selama {duration} detik...")
        print("✨ TIPS: Bicara dengan jelas, tidak terlalu cepat, dan jeda antar kata")
        print("📢 Mulai berbicara sekarang!")
        
        # Countdown untuk persiapan
        for i in range(3, 0, -1):
            print(f"⏰ {i}...")
            import time
            time.sleep(0.8)
        
        print("🔴 RECORDING DIMULAI!")
        
        audio_data = sd.rec(int(duration * self.sample_rate), 
                          samplerate=self.sample_rate, 
                          channels=1, 
                          dtype='float32')
        sd.wait()
        print("✅ Recording selesai!")
        return audio_data.flatten()
    
    def save_temp_audio(self, audio_data):
        """Simpan audio ke temporary file untuk speech recognition"""
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(temp_file.name, audio_data, self.sample_rate)
        return temp_file.name
    
    def speech_to_text(self, audio_file_path):
        """Speech recognition dengan ambient noise adjustment"""
        try:
            print("🤖 Memproses speech recognition...")
            
            with sr.AudioFile(audio_file_path) as source:
                print("🔧 Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
                print("👂 Listening to full audio...")
                audio = self.recognizer.listen(source)  # Simple approach
                print("✅ Audio loaded successfully")
            
            print("📡 Calling Google Speech API...")
            
            languages = [
                ('id-ID', 'Indonesian'),
                ('en-US', 'English')
            ]
            
            for lang_code, lang_name in languages:
                try:
                    print(f"🌍 Trying {lang_name}...")
                    text = self.recognizer.recognize_google(audio, language=lang_code)  # type: ignore
                    print(f"✅ SUCCESS with {lang_name}: '{text}'")
                    return text.lower().strip()
                except sr.UnknownValueError:
                    print(f"❌ {lang_name}: Could not understand audio")
                except sr.RequestError as e:
                    print(f"❌ {lang_name}: API Error - {e}")
            
            return "TIDAK_TERDETEKSI"
            
        except Exception as e:
            print(f"❌ Error speech recognition: {e}")
            return "ERROR"
    
    def calculate_similarity(self, text1, text2):
        """
        Hitung similarity antara dua text menggunakan multiple metrics
        """
        text1 = self.normalize_text(text1)
        text2 = self.normalize_text(text2)
        
        # 1. Sequence Matcher (overall similarity)
        sequence_similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
        
        # 2. Word-level comparison
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if len(words1) == 0 and len(words2) == 0:
            word_similarity = 1.0
        elif len(words1) == 0 or len(words2) == 0:
            word_similarity = 0.0
        else:
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            word_similarity = intersection / union if union > 0 else 0.0
        
        # 3. Character-level similarity (untuk typos)
        char_similarity = difflib.SequenceMatcher(None, 
                                                 ''.join(text1.split()), 
                                                 ''.join(text2.split())).ratio()
        
        # Weighted combination
        final_similarity = (
            sequence_similarity * 0.5 +
            word_similarity * 0.3 +
            char_similarity * 0.2
        )
        
        return {
            'overall': final_similarity,
            'sequence': sequence_similarity,
            'word': word_similarity,
            'character': char_similarity
        }
    
    def normalize_text(self, text):
        """Normalize text untuk comparison yang lebih akurat"""
        # Convert ke lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def get_pronunciation_score(self, target_sentence, recognized_text):
        """
        Berikan scoring untuk pronunciation accuracy
        """
        if recognized_text in ["TIDAK_TERDETEKSI", "ERROR_API", "ERROR"]:
            return {
                'score': 0,
                'status': 'FAILED',
                'reason': recognized_text,
                'similarity': {}
            }
        
        similarity = self.calculate_similarity(target_sentence, recognized_text)
        overall_score = similarity['overall'] * 100
        
        # Determine status
        if overall_score >= 80:
            status = "EXCELLENT"
        elif overall_score >= 65:
            status = "GOOD"
        elif overall_score >= 50:
            status = "MODERATE"
        else:
            status = "POOR"
        
        return {
            'score': round(overall_score, 1),
            'status': status,
            'similarity': similarity,
            'reason': None
        }
    
    def get_random_sentence(self):
        """Ambil kalimat random dari daftar sample sentences"""
        import random
        return random.choice(self.sample_sentences)
    
    def cleanup_temp_file(self, file_path):
        """Hapus temporary file"""
        try:
            os.unlink(file_path)
        except:
            pass

# Test function untuk speech recognition dan comparison
def test_speech_recognition():
    """Test complete speech recognition dan text comparison"""
    print("=== VOICE READINESS CHECKER - TAHAP 2 ===")
    print("Speech Recognition & Text Comparison\n")
    
    checker = VoiceReadinessChecker()
    
    # Pilih kalimat target
    target_sentence = checker.get_random_sentence()
    
    print("📝 INSTRUKSI:")
    print("Silakan ucapkan kalimat berikut dengan jelas:")
    print(f"📢 \"{target_sentence}\"")
    print()
    
    input("Tekan Enter untuk mulai recording...")
    
    try:
        audio_data = checker.record_audio(duration=5)
        
        # Simpan ke temporary file
        temp_audio_file = checker.save_temp_audio(audio_data)
        
        print("🤖 Memproses speech recognition...")
        
        # Speech to text
        recognized_text = checker.speech_to_text(temp_audio_file)
        
        print(f"\n📊 HASIL ANALISIS:")
        print(f"Target    : \"{target_sentence}\"")
        print(f"Recognized: \"{recognized_text}\"")
        print()
        
        # Get pronunciation score
        result = checker.get_pronunciation_score(target_sentence, recognized_text)
        
        if result['status'] == 'FAILED':
            print(f"❌ Speech Recognition GAGAL: {result['reason']}")
            if result['reason'] == "TIDAK_TERDETEKSI":
                print("💡 Saran: Bicara lebih keras dan jelas")
            elif result['reason'] == "ERROR_API":
                print("💡 Saran: Periksa koneksi internet")
        else:
            print(f"🎯 Pronunciation Score: {result['score']}/100")
            print(f"📈 Status: {result['status']}")
            
            similarity = result['similarity']
            print(f"\n📊 Detail Similarity:")
            print(f"  Overall   : {similarity['overall']:.1%}")
            print(f"  Sequence  : {similarity['sequence']:.1%}")
            print(f"  Word Match: {similarity['word']:.1%}")
            print(f"  Character : {similarity['character']:.1%}")
            
            if result['score'] >= 80:
                print("✅ SANGAT BAIK! Pronunciation accuracy tinggi")
            elif result['score'] >= 65:
                print("✅ BAIK! Pronunciation cukup akurat")
            elif result['score'] >= 50:
                print("⚠️ CUKUP! Perlu sedikit perbaikan pronunciation")
            else:
                print("❌ KURANG! Pronunciation perlu diperbaiki")
                print("💡 Saran: Ulangi dengan lebih pelan dan jelas")
        
        checker.cleanup_temp_file(temp_audio_file)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Troubleshooting:")
        print("1. Pastikan koneksi internet stabil (untuk Google Speech API)")
        print("2. Bicara dengan jelas dan tidak terlalu cepat")
        print("3. Install: pip install SpeechRecognition")

if __name__ == "__main__":
    test_speech_recognition()
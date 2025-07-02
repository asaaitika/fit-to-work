import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
import tempfile
import os

import sounddevice as sd
import soundfile as sf
import librosa
import librosa.display
import speech_recognition as sr
import difflib
import random

# Configure Streamlit
st.set_page_config(
    page_title="Fit-to-Work Voice Checker - Real System",
    page_icon="🏭",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: bold;
}
.success-box {
    background-color: #d4edda;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #28a745;
    margin: 1rem 0;
}
.warning-box {
    background-color: #fff3cd;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #ffc107;
    margin: 1rem 0;
}
.danger-box {
    background-color: #f8d7da;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #dc3545;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

class RealVoiceChecker:
    """
    Real Voice Checker
    """
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.recognizer = sr.Recognizer()
        
        # Emotion labels dari kode asli
        self.emotion_labels = {
            'ready': 'Siap & Fokus',
            'tired': 'Lelah/Mengantuk', 
            'stressed': 'Stress/Tegang',
            'calm': 'Tenang & Stabil',
            'uncertain': 'Ragu/Tidak Yakin'
        }
        
        # Sample sentences dari kode asli
        self.sample_sentences = [
            "Saya siap kerja",
            "Keselamatan utama", 
            "Kondisi sehat",
            "Peralatan aman",
            "Tim komunikasi baik"
        ]
        
        print("🔧 Real Voice Checker initialized")
    
    def record_audio_streamlit(self, duration=6):
        """Real audio recording untuk Streamlit"""
        try:
            st.info(f"🎤 Recording selama {duration} detik...")
            st.info("📢 Ucapkan kalimat target dengan jelas!")
            
            # Real audio recording menggunakan sounddevice
            audio_data = sd.rec(int(duration * self.sample_rate), 
                              samplerate=self.sample_rate, 
                              channels=1, 
                              dtype='float32')
            
            # Show recording progress
            progress_bar = st.progress(0)
            for i in range(duration):
                time.sleep(1)
                progress_bar.progress((i + 1) / duration)
            
            sd.wait()  # Wait for recording to finish
            st.success("✅ Recording selesai!")
            
            return audio_data.flatten()
            
        except Exception as e:
            st.error(f"❌ Error during recording: {e}")
            st.error("Pastikan microphone terhubung dan permission diberikan")
            return None
    
    def save_temp_audio(self, audio_data):
        """Save audio to temporary file"""
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(temp_file.name, audio_data, self.sample_rate)
        return temp_file.name
    
    def speech_to_text(self, audio_file_path):
        """Speech recognition"""
        try:
            st.info("🤖 Memproses speech recognition...")
            
            with sr.AudioFile(audio_file_path) as source:
                st.info("🔧 Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
                st.info("🔧 Reading audio...")
                audio = self.recognizer.listen(source)
                st.success("✅ Audio loaded successfully")
            
            st.info("📡 Calling Google Speech API...")
            
            # Test dengan bahasa Indonesia dan English
            languages = [('id-ID', 'Indonesian'), ('en-US', 'English')]
            
            for lang_code, lang_name in languages:
                try:
                    st.info(f"🌍 Trying {lang_name}...")
                    text = self.recognizer.recognize_google(audio, language=lang_code)  # type: ignore
                    st.success(f"✅ SUCCESS with {lang_name}: '{text}'")
                    return text.lower().strip()
                except sr.UnknownValueError:
                    st.warning(f"❌ {lang_name}: Could not understand audio")
                except sr.RequestError as e:
                    st.error(f"❌ {lang_name}: API Error - {e}")
            
            return "TIDAK_TERDETEKSI"
            
        except Exception as e:
            st.error(f"❌ Error speech recognition: {e}")
            return "ERROR"
    
    def calculate_pronunciation_similarity(self, target_sentence, recognized_text):
        """Calculate similarity"""
        if recognized_text in ["TIDAK_TERDETEKSI", "ERROR"]:
            return 0
        
        # Normalize texts
        target = target_sentence.lower().strip()
        recognized = recognized_text.lower().strip()
        
        # Calculate similarity
        similarity = difflib.SequenceMatcher(None, target, recognized).ratio()
        return round(similarity * 100, 1)
    
    def extract_voice_features(self, audio_data):
        """Extract voice features"""
        st.info("🔍 Extracting voice features...")
        
        features = {}
        
        # 1. Pitch Analysis
        try:
            pitches, magnitudes = librosa.piptrack(y=audio_data, sr=self.sample_rate, 
                                                  threshold=0.1, fmin=50, fmax=400)
            
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if len(pitch_values) > 0:
                features['pitch_mean'] = np.mean(pitch_values)
                features['pitch_std'] = np.std(pitch_values)
                features['pitch_range'] = np.max(pitch_values) - np.min(pitch_values)
            else:
                features['pitch_mean'] = 0
                features['pitch_std'] = 0
                features['pitch_range'] = 0
                
        except Exception as e:
            st.warning(f"⚠️ Pitch analysis warning: {e}")
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
            features['pitch_range'] = 0
        
        # 2. Energy Analysis - EXACT same code
        features['rms_energy'] = np.sqrt(np.mean(audio_data**2))
        features['max_amplitude'] = np.max(np.abs(audio_data))
        features['energy_variance'] = np.var(audio_data**2)
        
        # 3. Speaking Rate Analysis - EXACT same code
        frame_length = int(0.025 * self.sample_rate)
        hop_length = int(0.01 * self.sample_rate)
        
        energy_frames = []
        for i in range(0, len(audio_data) - frame_length, hop_length):
            frame = audio_data[i:i + frame_length]
            energy = np.sum(frame**2)
            energy_frames.append(energy)
        
        energy_frames = np.array(energy_frames)
        energy_threshold = np.mean(energy_frames) * 0.1
        
        speech_frames = np.sum(energy_frames > energy_threshold)
        total_frames = len(energy_frames)
        
        features['speech_rate'] = speech_frames / total_frames if total_frames > 0 else 0
        
        # 4. Spectral Features - EXACT same code
        try:
            mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
            features['mfcc_mean'] = np.mean(mfccs, axis=1)
            features['mfcc_std'] = np.std(mfccs, axis=1)
            
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            
            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
        except Exception as e:
            st.warning(f"⚠️ Spectral features warning: {e}")
            features['spectral_centroid_mean'] = 1500
            features['zcr_mean'] = 0.05
        
        # 5. Temporal Features - EXACT same code
        silence_threshold = features['rms_energy'] * 0.1
        silent_frames = np.sum(energy_frames < silence_threshold)
        features['silence_ratio'] = silent_frames / total_frames if total_frames > 0 else 0
        
        st.success(f"✅ Extracted {len(features)} voice features")
        return features
    
    def analyze_emotion_patterns(self, features):
        """Analyze emotion patterns"""
        st.info("🧠 Analyzing emotion patterns...")
        
        emotion_scores = {}
        
        # Analyze READY/FOCUSED patterns
        ready_score = 0
        if 100 < features['pitch_mean'] < 300:
            ready_score += 20
        if features['pitch_std'] < 50:
            ready_score += 15
        if 0.01 < features['rms_energy'] < 0.1:
            ready_score += 20
        if features['speech_rate'] > 0.3:
            ready_score += 15
        if features['spectral_centroid_mean'] > 1000:
            ready_score += 15
        if features['silence_ratio'] < 0.7:
            ready_score += 15
        emotion_scores['ready'] = min(ready_score, 100)
        
        # Analyze TIRED patterns
        tired_score = 0
        if features['pitch_mean'] < 150:
            tired_score += 25
        if features['rms_energy'] < 0.02:
            tired_score += 30
        if features['speech_rate'] < 0.4:
            tired_score += 25
        if features['silence_ratio'] > 0.5:
            tired_score += 20
        emotion_scores['tired'] = min(tired_score, 100)
        
        # Analyze STRESSED patterns
        stressed_score = 0
        if features['pitch_mean'] > 250:
            stressed_score += 25
        if features['pitch_std'] > 60:
            stressed_score += 25
        if features['energy_variance'] > 0.01:
            stressed_score += 25
        if features['zcr_mean'] > 0.1:
            stressed_score += 25
        emotion_scores['stressed'] = min(stressed_score, 100)
        
        # Analyze CALM patterns
        calm_score = 0
        if 120 < features['pitch_mean'] < 220:
            calm_score += 25
        if features['pitch_std'] < 30:
            calm_score += 25
        if features['energy_variance'] < 0.005:
            calm_score += 25
        if 0.4 < features['speech_rate'] < 0.7:
            calm_score += 25
        emotion_scores['calm'] = min(calm_score, 100)
        
        # Analyze UNCERTAIN patterns
        uncertain_score = 0
        if features['pitch_std'] > 40:
            uncertain_score += 30
        if features['silence_ratio'] > 0.6:
            uncertain_score += 30
        if features['speech_rate'] < 0.5:
            uncertain_score += 20
        if features['spectral_centroid_mean'] < 1500:
            uncertain_score += 20
        emotion_scores['uncertain'] = min(uncertain_score, 100)
        
        return emotion_scores
    
    def determine_work_readiness(self, emotion_scores):
        """Determine work readiness"""
        st.info("⚖️ Determining work readiness...")
        
        positive_emotions = emotion_scores['ready'] + emotion_scores['calm']
        negative_emotions = emotion_scores['tired'] + emotion_scores['stressed'] + emotion_scores['uncertain']
        
        readiness_score = (positive_emotions * 0.7) - (negative_emotions * 0.3)
        readiness_score = max(0, min(100, readiness_score))
        
        # EXACT same status determination
        if readiness_score >= 70:
            status = "SIAP KERJA"
            recommendation = "Kondisi mental dan fisik baik untuk bekerja"
            color = "🟢"
        elif readiness_score >= 50:
            status = "CUKUP SIAP"
            recommendation = "Bisa bekerja, tapi perhatikan kondisi diri"
            color = "🟡"
        else:
            status = "BELUM SIAP"
            recommendation = "Sebaiknya istirahat dulu atau konsultasi supervisor"
            color = "🔴"
        
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        
        return {
            'readiness_score': round(readiness_score, 1),
            'status': status,
            'recommendation': recommendation,
            'color': color,
            'dominant_emotion': dominant_emotion[0],
            'dominant_score': dominant_emotion[1],
            'emotion_breakdown': emotion_scores
        }
    
    def get_random_sentence(self):
        """Get random sentence"""
        return random.choice(self.sample_sentences)
    
    def cleanup_temp_file(self, file_path):
        """Cleanup temporary file"""
        try:
            os.unlink(file_path)
        except:
            pass

# Cognitive Assessment Classes
class RealCognitiveAssessment:
    """Real cognitive assessment dengan interactive Streamlit UI"""
    
    def __init__(self):
        self.cognitive_tests = {
            'attention': self.attention_test_streamlit,
            'memory': self.memory_test_streamlit, 
            'reaction': self.reaction_time_test_streamlit,
            'math': self.simple_math_test_streamlit,
            'sequence': self.sequence_test_streamlit
        }
        print("🧠 Real Cognitive Assessment Module initialized")
    
    def attention_test_streamlit(self):
        """REAL attention test dengan Streamlit UI"""
        st.subheader("🎯 ATTENTION TEST")
        st.write("Hitung berapa huruf 'A' dalam teks berikut:")
        
        texts = [
            "BANANA ADALAH BUAH YANG MANIS DAN BERGIZI TINGGI",
            "APLIKASI ANDROID SANGAT MEMBANTU AKTIVITAS HARIAN",
            "AREA PARKIR YANG AMAN MEMBERIKAN RASA TENANG",
            "ALARM KEBAKARAN AKTIF SETIAP SAAT DI AREA KERJA"
        ]
        
        if 'attention_text' not in st.session_state:
            st.session_state.attention_text = random.choice(texts)
            st.session_state.attention_start_time = time.time()
        
        selected_text = st.session_state.attention_text
        correct_count = selected_text.count('A')
        
        st.code(selected_text, language=None)
        
        user_answer = st.number_input("Berapa huruf 'A'?", min_value=0, max_value=50, value=0)
        
        if st.button("Submit Answer", type="primary"):
            response_time = time.time() - st.session_state.attention_start_time
            
            if user_answer == correct_count:
                score = max(0, 100 - int(response_time * 5))
                result = "✅ BENAR"
                st.success(f"{result} - Score: {score}/100")
            else:
                score = max(0, 50 - int(response_time * 3))
                result = f"❌ SALAH (jawaban: {correct_count})"
                st.error(f"{result} - Score: {score}/100")
            
            # Reset untuk test berikutnya
            del st.session_state.attention_text
            del st.session_state.attention_start_time
            
            return {
                'test_type': 'attention',
                'score': score,
                'response_time': round(response_time, 2),
                'correct': user_answer == correct_count,
                'result': result
            }
        
        return None
    
    def memory_test_streamlit(self):
        """REAL memory test dengan Streamlit UI"""
        st.subheader("🧠 MEMORY TEST")
        
        if 'memory_sequence' not in st.session_state:
            sequence_length = random.randint(4, 6)
            st.session_state.memory_sequence = [random.randint(1, 9) for _ in range(sequence_length)]
            st.session_state.memory_shown = False
            st.session_state.memory_start_time = time.time()
        
        if not st.session_state.memory_shown:
            st.write("Hafalkan sequence angka berikut (akan hilang dalam 5 detik):")
            sequence_str = " - ".join(map(str, st.session_state.memory_sequence))
            st.code(sequence_str, language=None)
            
            if st.button("Saya sudah hafal, lanjutkan test"):
                st.session_state.memory_shown = True
                st.rerun()
        else:
            st.write("Sekarang ketik sequence yang tadi:")
            user_input = st.text_input("Sequence (pisahkan dengan spasi):", placeholder="1 2 3 4")
            
            if st.button("Submit Sequence", type="primary"):
                response_time = time.time() - st.session_state.memory_start_time
                
                try:
                    user_sequence = [int(x.strip()) for x in user_input.split()]
                    
                    if user_sequence == st.session_state.memory_sequence:
                        score = max(0, 100 - int(response_time * 3))
                        result = "✅ BENAR"
                        st.success(f"{result} - Score: {score}/100")
                    else:
                        correct_positions = sum(1 for i, (a, b) in enumerate(zip(user_sequence, st.session_state.memory_sequence)) if a == b)
                        score = max(0, int(correct_positions / len(st.session_state.memory_sequence) * 70) - int(response_time * 2))
                        result = f"❌ SALAH (benar: {st.session_state.memory_sequence})"
                        st.error(f"{result} - Score: {score}/100")
                    
                    # Reset
                    sequence_copy = st.session_state.memory_sequence.copy()
                    del st.session_state.memory_sequence
                    del st.session_state.memory_shown
                    del st.session_state.memory_start_time
                    
                    return {
                        'test_type': 'memory',
                        'score': score,
                        'response_time': round(response_time, 2),
                        'correct': user_sequence == sequence_copy,
                        'result': result
                    }
                    
                except (ValueError, IndexError):
                    st.error("Format input tidak valid. Gunakan angka dipisah spasi.")
        
        return None
    
    def reaction_time_test_streamlit(self):
        """REAL reaction time test"""
        st.subheader("⚡ REACTION TIME TEST")
        st.write("Klik tombol 'REACT!' secepat mungkin saat melihat '🚨 GO!'")
        
        if 'reaction_waiting' not in st.session_state:
            if st.button("Mulai Test"):
                st.session_state.reaction_waiting = True
                delay = random.uniform(2, 5)
                st.session_state.reaction_delay = delay
                st.session_state.reaction_start_time = time.time()
                st.rerun()
        
        elif st.session_state.reaction_waiting:
            current_time = time.time()
            elapsed = current_time - st.session_state.reaction_start_time
            
            if elapsed >= st.session_state.reaction_delay:
                st.markdown("### 🚨 GO!")
                if st.button("REACT!", type="primary"):
                    reaction_time = time.time() - (st.session_state.reaction_start_time + st.session_state.reaction_delay)
                    
                    if reaction_time < 0.5:
                        score = 100
                        result = "🏆 EXCELLENT"
                    elif reaction_time < 1.0:
                        score = 80
                        result = "✅ GOOD"
                    elif reaction_time < 2.0:
                        score = 60
                        result = "⚠️ AVERAGE"
                    else:
                        score = 30
                        result = "❌ SLOW"
                    
                    st.success(f"{result} - Reaction time: {reaction_time:.3f}s - Score: {score}/100")
                    
                    # Reset
                    del st.session_state.reaction_waiting
                    del st.session_state.reaction_delay
                    del st.session_state.reaction_start_time
                    
                    return {
                        'test_type': 'reaction',
                        'score': score,
                        'response_time': round(reaction_time, 3),
                        'correct': True,
                        'result': f"{result} ({reaction_time:.3f}s)"
                    }
            else:
                st.write("Tunggu instruksi...")
                time.sleep(0.1)
                st.rerun()
        
        return None
    
    def simple_math_test_streamlit(self):
        """REAL math test"""
        st.subheader("🔢 MATH TEST")
        
        if 'math_problem' not in st.session_state:
            problems = [
                (lambda: (random.randint(10, 50), random.randint(5, 20)), lambda a, b: a + b, "+"),
                (lambda: (random.randint(30, 80), random.randint(5, 25)), lambda a, b: a - b, "-"),
                (lambda: (random.randint(2, 12), random.randint(2, 9)), lambda a, b: a * b, "×")
            ]
            
            generator, operation, symbol = random.choice(problems)
            a, b = generator()
            correct_answer = operation(a, b)
            
            st.session_state.math_problem = {
                'a': a, 'b': b, 'operation': operation, 'symbol': symbol, 'answer': correct_answer
            }
            st.session_state.math_start_time = time.time()
        
        problem = st.session_state.math_problem
        st.write("Hitung dengan cepat:")
        st.markdown(f"### {problem['a']} {problem['symbol']} {problem['b']} = ?")
        
        user_answer = st.number_input("Jawaban:", value=0)
        
        if st.button("Submit Math Answer", type="primary"):
            response_time = time.time() - st.session_state.math_start_time
            
            if user_answer == problem['answer']:
                score = max(0, 100 - int(response_time * 10))
                result = "✅ BENAR"
                st.success(f"{result} - Score: {score}/100")
            else:
                score = max(0, 30 - int(response_time * 5))
                result = f"❌ SALAH (jawaban: {problem['answer']})"
                st.error(f"{result} - Score: {score}/100")
            
            # Reset
            del st.session_state.math_problem
            del st.session_state.math_start_time
            
            return {
                'test_type': 'math',
                'score': score,
                'response_time': round(response_time, 2),
                'correct': user_answer == problem['answer'],
                'result': result
            }
        
        return None
    
    def sequence_test_streamlit(self):
        """REAL sequence test"""
        st.subheader("🔄 SEQUENCE TEST")
        
        if 'sequence_pattern' not in st.session_state:
            patterns = [
                ([2, 4, 6, 8], 10, "Bilangan genap"),
                ([1, 3, 5, 7], 9, "Bilangan ganjil"),
                ([5, 10, 15, 20], 25, "Kelipatan 5"),
                ([1, 4, 9, 16], 25, "Kuadrat"),
                ([2, 6, 18, 54], 162, "×3"),
            ]
            st.session_state.sequence_pattern = random.choice(patterns)
            st.session_state.sequence_start_time = time.time()
        
        sequence, answer, description = st.session_state.sequence_pattern
        
        st.write("Lanjutkan pola berikut:")
        sequence_str = " - ".join(map(str, sequence)) + " - ?"
        st.code(sequence_str, language=None)
        
        user_answer = st.number_input("Angka selanjutnya:", value=0)
        
        if st.button("Submit Sequence Answer", type="primary"):
            response_time = time.time() - st.session_state.sequence_start_time
            
            if user_answer == answer:
                score = max(0, 100 - int(response_time * 8))
                result = f"✅ BENAR ({description})"
                st.success(f"{result} - Score: {score}/100")
            else:
                score = max(0, 40 - int(response_time * 4))
                result = f"❌ SALAH (jawaban: {answer} - {description})"
                st.error(f"{result} - Score: {score}/100")
            
            # Reset
            del st.session_state.sequence_pattern
            del st.session_state.sequence_start_time
            
            return {
                'test_type': 'sequence',
                'score': score,
                'response_time': round(response_time, 2),
                'correct': user_answer == answer,
                'result': result
            }
        
        return None
    
    def run_cognitive_battery_streamlit(self, num_tests=3):
        """Run cognitive battery dengan Streamlit UI"""
        st.subheader("🧠 COGNITIVE ASSESSMENT BATTERY")
        
        if 'cognitive_tests_completed' not in st.session_state:
            st.session_state.cognitive_tests_completed = []
            available_tests = list(self.cognitive_tests.keys())
            st.session_state.selected_tests = random.sample(available_tests, min(num_tests, len(available_tests)))
            st.session_state.current_test_index = 0
        
        total_tests = len(st.session_state.selected_tests)
        current_index = st.session_state.current_test_index
        
        if current_index < total_tests:
            st.write(f"Test {current_index + 1}/{total_tests}")
            
            # Progress bar
            progress = current_index / total_tests
            st.progress(progress)
            
            # Run current test
            test_name = st.session_state.selected_tests[current_index]
            result = self.cognitive_tests[test_name]()
            
            if result:
                st.session_state.cognitive_tests_completed.append(result)
                st.session_state.current_test_index += 1
                
                if st.session_state.current_test_index < total_tests:
                    if st.button("Lanjut ke test berikutnya"):
                        st.rerun()
                else:
                    # All tests completed
                    st.success("🎉 Semua cognitive tests selesai!")
                    
                    # Calculate overall cognitive score
                    results = st.session_state.cognitive_tests_completed
                    total_score = sum(r['score'] for r in results)
                    avg_score = total_score / len(results)
                    
                    # Determine cognitive status
                    if avg_score >= 80:
                        status = "🟢 COGNITIVE EXCELLENT"
                        recommendation = "Kesiapan mental sangat baik untuk bekerja"
                    elif avg_score >= 65:
                        status = "🟡 COGNITIVE GOOD"
                        recommendation = "Kesiapan mental baik, bisa bekerja normal"
                    elif avg_score >= 50:
                        status = "🟠 COGNITIVE MODERATE"
                        recommendation = "Perlu perhatian extra, hindari tugas complex"
                    else:
                        status = "🔴 COGNITIVE POOR"
                        recommendation = "Sebaiknya istirahat, tidak disarankan bekerja"
                    
                    cognitive_summary = {
                        'total_tests': len(results),
                        'individual_results': results,
                        'average_score': round(avg_score, 1),
                        'total_score': total_score,
                        'status': status,
                        'recommendation': recommendation,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    return cognitive_summary
        else:
            # All tests completed, return summary
            results = st.session_state.cognitive_tests_completed
            total_score = sum(r['score'] for r in results)
            avg_score = total_score / len(results)
            
            if avg_score >= 80:
                status = "🟢 COGNITIVE EXCELLENT"
                recommendation = "Kesiapan mental sangat baik untuk bekerja"
            elif avg_score >= 65:
                status = "🟡 COGNITIVE GOOD"
                recommendation = "Kesiapan mental baik, bisa bekerja normal"
            elif avg_score >= 50:
                status = "🟠 COGNITIVE MODERATE"
                recommendation = "Perlu perhatian extra, hindari tugas complex"
            else:
                status = "🔴 COGNITIVE POOR"
                recommendation = "Sebaiknya istirahat, tidak disarankan bekerja"
            
            return {
                'total_tests': len(results),
                'individual_results': results,
                'average_score': round(avg_score, 1),
                'total_score': total_score,
                'status': status,
                'recommendation': recommendation,
                'timestamp': datetime.now().isoformat()
            }
        
        return None

# Main Streamlit Application
def main():
    """Main Streamlit application dengan REAL functionality"""
    
    # Header
    st.markdown('<h1 class="main-header">🏭 Fit-to-Work Voice Readiness Checker</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'assessment_history' not in st.session_state:
        st.session_state.assessment_history = []
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'home'
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox("Choose Mode:", [
        "🏠 Home", 
        "🎤 Voice Analysis", 
        "🧠 Cognitive Tests", 
        "📊 Complete Assessment",
        "📈 Results Dashboard"
    ])
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "🎤 Voice Analysis":
        show_real_voice_analysis()
    elif page == "🧠 Cognitive Tests":
        show_real_cognitive_tests()
    elif page == "📊 Complete Assessment":
        show_real_complete_assessment()
    else:
        show_results_dashboard()

def show_home_page():
    """Home page dengan overview"""
    st.markdown("### 🎯 Overview")
    
    st.warning("""
    **Requirements:**
    - Microphone yang berfungsi
    - Internet connection (untuk Google Speech API)
    - Permission untuk akses microphone
    """)
    
    # Quick start buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎤 Test Voice Analysis", type="primary", use_container_width=True):
            st.session_state.current_page = 'voice'
            st.rerun()
    
    with col2:
        if st.button("🧠 Test Cognitive", type="secondary", use_container_width=True):
            st.session_state.current_page = 'cognitive'
            st.rerun()
    
    with col3:
        if st.button("📊 Complete Assessment", use_container_width=True):
            st.session_state.current_page = 'complete'
            st.rerun()
    
    # System status
    st.markdown("### 🔧 System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Test microphone availability
        try:
            devices = sd.query_devices()
            input_devices = [d for d in devices if d.get('max_input_channels', 0) > 0]
            if input_devices:
                st.success("🎤 Microphone: OK")
            else:
                st.error("🎤 Microphone: Not Found")
        except:
            st.error("🎤 Microphone: Error")
    
    with col2:
        # Test speech recognition
        try:
            recognizer = sr.Recognizer()
            st.success("🗣️ Speech API: Ready")
        except:
            st.error("🗣️ Speech API: Error")
    
    with col3:
        # Test librosa
        try:
            import librosa
            st.success("📊 Audio Processing: OK")
        except:
            st.error("📊 Audio Processing: Error")
    
    with col4:
        st.success("🌐 Streamlit: Online")

def show_real_voice_analysis():
    """Real voice analysis page"""
    st.markdown("### 🎤 Real Voice Analysis")
    
    checker = RealVoiceChecker()
    
    # Step 1: Select target sentence
    st.markdown("#### Step 1: Select Target Sentence")
    target_sentence = st.selectbox(
        "Choose sentence to say:",
        checker.sample_sentences
    )
    
    # Step 2: Real audio recording
    st.markdown("#### Step 2: Record Your Voice")
    st.info(f"📝 **Say this sentence:** \"{target_sentence}\"")
    
    if st.button("🎤 Start Real Recording", type="primary"):
        # Real audio recording
        audio_data = checker.record_audio_streamlit(duration=6)
        
        if audio_data is not None:
            # Save to temporary file
            temp_audio_file = checker.save_temp_audio(audio_data)
            
            try:
                # Step 3: Real speech recognition
                st.markdown("#### Step 3: Speech Recognition")
                recognized_text = checker.speech_to_text(temp_audio_file)
                
                # Step 4: Calculate pronunciation score
                pronunciation_score = checker.calculate_pronunciation_similarity(target_sentence, recognized_text)
                
                # Step 5: Real voice feature extraction
                st.markdown("#### Step 4: Voice Feature Extraction")
                features = checker.extract_voice_features(audio_data)
                
                # Step 6: Real emotion analysis
                st.markdown("#### Step 5: Emotion Analysis")
                emotion_scores = checker.analyze_emotion_patterns(features)
                voice_result = checker.determine_work_readiness(emotion_scores)
                
                # Display results
                st.markdown("#### 📊 Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🗣️ Speech Recognition:**")
                    st.write(f"Target: {target_sentence}")
                    st.write(f"Recognized: {recognized_text}")
                    st.metric("Pronunciation Score", f"{pronunciation_score}%")
                
                with col2:
                    st.markdown("**🎭 Voice Analysis:**")
                    st.metric("Readiness Score", f"{voice_result['readiness_score']}/100")
                    st.write(f"Status: {voice_result['color']} {voice_result['status']}")
                    st.write(f"Dominant Emotion: {checker.emotion_labels[voice_result['dominant_emotion']]}")
                
                # Emotion breakdown chart
                st.markdown("**📈 Emotion Breakdown:**")
                emotion_df = pd.DataFrame([
                    {'Emotion': checker.emotion_labels[k], 'Score': v} 
                    for k, v in emotion_scores.items()
                ])
                
                fig = px.bar(emotion_df, x='Emotion', y='Score',
                           title="Real-time Emotion Analysis Results",
                           color='Score', color_continuous_scale='viridis')
                st.plotly_chart(fig, use_container_width=True)
                
                # Voice characteristics
                st.markdown("**🔍 Voice Characteristics:**")
                char_col1, char_col2, char_col3 = st.columns(3)
                
                with char_col1:
                    st.metric("Pitch Mean", f"{features['pitch_mean']:.1f} Hz")
                
                with char_col2:
                    st.metric("RMS Energy", f"{features['rms_energy']:.4f}")
                
                with char_col3:
                    st.metric("Speech Rate", f"{features['speech_rate']:.2f}")
                
            finally:
                # Cleanup
                checker.cleanup_temp_file(temp_audio_file)

def show_real_cognitive_tests():
    """Real cognitive tests page"""
    st.markdown("### 🧠 Real Cognitive Assessment")
    
    cognitive_assessment = RealCognitiveAssessment()
    
    st.info("Choose individual cognitive tests to run:")
    
    test_options = {
        '🎯 Attention Test': 'attention',
        '🧠 Memory Test': 'memory',
        '⚡ Reaction Time Test': 'reaction',
        '🔢 Math Test': 'math',
        '🔄 Sequence Test': 'sequence'
    }
    
    selected_test_name = st.selectbox("Select Test:", list(test_options.keys()))
    selected_test = test_options[selected_test_name]
    
    st.markdown(f"#### {selected_test_name}")
    
    # Run the selected test
    if selected_test == 'attention':
        result = cognitive_assessment.attention_test_streamlit()
    elif selected_test == 'memory':
        result = cognitive_assessment.memory_test_streamlit()
    elif selected_test == 'reaction':
        result = cognitive_assessment.reaction_time_test_streamlit()
    elif selected_test == 'math':
        result = cognitive_assessment.simple_math_test_streamlit()
    else:
        result = cognitive_assessment.sequence_test_streamlit()
    
    if result:
        st.success(f"Test completed! Score: {result['score']}/100")
        
        # Save individual test result
        if 'individual_test_results' not in st.session_state:
            st.session_state.individual_test_results = []
        
        st.session_state.individual_test_results.append(result)

def show_real_complete_assessment():
    """Real complete assessment workflow"""
    st.markdown("### 📊 Complete Real Assessment")
    
    # Initialize assessment state
    if 'complete_assessment_step' not in st.session_state:
        st.session_state.complete_assessment_step = 'start'
    
    if st.session_state.complete_assessment_step == 'start':
        st.markdown("#### 🎯 Complete Fit-to-Work Assessment")
        st.info("""
        This assessment includes:
        1. **Real Voice Analysis** (6-second recording)
        2. **Real Cognitive Battery** (3 tests)
        3. **Final Integration** (weighted scoring)
        
        Total time: ~5-7 minutes
        """)
        
        if st.button("🚀 Start Complete Assessment", type="primary"):
            st.session_state.complete_assessment_step = 'voice'
            st.rerun()
    
    elif st.session_state.complete_assessment_step == 'voice':
        st.markdown("#### 🎤 Step 1: Voice Analysis")
        
        checker = RealVoiceChecker()
        target_sentence = checker.get_random_sentence()
        
        st.info(f"📝 **Say this sentence:** \"{target_sentence}\"")
        
        if st.button("🎤 Record Voice", type="primary"):
            audio_data = checker.record_audio_streamlit(duration=6)
            
            if audio_data is not None:
                temp_audio_file = checker.save_temp_audio(audio_data)
                
                try:
                    # Process voice analysis
                    recognized_text = checker.speech_to_text(temp_audio_file)
                    pronunciation_score = checker.calculate_pronunciation_similarity(target_sentence, recognized_text)
                    features = checker.extract_voice_features(audio_data)
                    emotion_scores = checker.analyze_emotion_patterns(features)
                    voice_result = checker.determine_work_readiness(emotion_scores)
                    
                    # Store results
                    st.session_state.complete_voice_result = {
                        'target_sentence': target_sentence,
                        'recognized_text': recognized_text,
                        'pronunciation_score': pronunciation_score,
                        'readiness_score': voice_result['readiness_score'],
                        'dominant_emotion': voice_result['dominant_emotion'],
                        'emotion_breakdown': emotion_scores,
                        'voice_features': features
                    }
                    
                    st.success(f"✅ Voice analysis complete! Score: {voice_result['readiness_score']}/100")
                    st.session_state.complete_assessment_step = 'cognitive'
                    
                    if st.button("Continue to Cognitive Tests"):
                        st.rerun()
                        
                finally:
                    checker.cleanup_temp_file(temp_audio_file)
    
    elif st.session_state.complete_assessment_step == 'cognitive':
        st.markdown("#### 🧠 Step 2: Cognitive Assessment")
        
        cognitive_assessment = RealCognitiveAssessment()
        cognitive_result = cognitive_assessment.run_cognitive_battery_streamlit(num_tests=3)
        
        if cognitive_result:
            st.session_state.complete_cognitive_result = cognitive_result
            st.session_state.complete_assessment_step = 'results'
            
            if st.button("View Final Results"):
                st.rerun()
    
    elif st.session_state.complete_assessment_step == 'results':
        st.markdown("#### 🎯 Final Assessment Results")
        
        voice_result = st.session_state.complete_voice_result
        cognitive_result = st.session_state.complete_cognitive_result
        
        # Calculate final score
        voice_score = voice_result['readiness_score']
        cognitive_score = cognitive_result['average_score']
        final_score = voice_score * 0.6 + cognitive_score * 0.4
        
        # Determine final status
        if final_score >= 75:
            status = "🟢 FIT TO WORK"
            card_class = "success-box"
            recommendation = "Pekerja siap dan aman untuk bekerja"
        elif final_score >= 60:
            status = "🟡 CONDITIONAL FIT"
            card_class = "warning-box"
            recommendation = "Bisa bekerja dengan pengawasan atau tugas ringan"
        else:
            status = "🔴 NOT FIT TO WORK"
            card_class = "danger-box"
            recommendation = "Tidak disarankan bekerja, perlu istirahat atau konsultasi"
        
        # Display final results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Voice Score", f"{voice_score:.1f}/100")
        with col2:
            st.metric("Cognitive Score", f"{cognitive_score:.1f}/100")
        with col3:
            st.metric("Final Score", f"{final_score:.1f}/100")
        
        st.markdown(f"""
        <div class="{card_class}">
            <h3>{status}</h3>
            <p><strong>Final Score:</strong> {final_score:.1f}/100</p>
            <p><strong>Recommendation:</strong> {recommendation}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed breakdown
        st.markdown("#### 📋 Detailed Results")
        
        detail_col1, detail_col2 = st.columns(2)
        
        with detail_col1:
            st.markdown("**🗣️ Voice Analysis:**")
            st.write(f"Target: {voice_result['target_sentence']}")
            st.write(f"Recognized: {voice_result['recognized_text']}")
            st.write(f"Pronunciation: {voice_result['pronunciation_score']:.1f}%")
            st.write(f"Dominant Emotion: {voice_result['dominant_emotion']}")
        
        with detail_col2:
            st.markdown("**🧠 Cognitive Tests:**")
            for i, test in enumerate(cognitive_result['individual_results'], 1):
                st.write(f"Test {i} ({test['test_type']}): {test['score']}/100")
        
        # Save complete assessment
        if st.button("💾 Save Complete Assessment", type="primary"):
            assessment_record = {
                'timestamp': datetime.now().isoformat(),
                'final_score': round(final_score, 1),
                'status': status,
                'voice_score': round(voice_score, 1),
                'cognitive_score': round(cognitive_score, 1),
                'voice_details': voice_result,
                'cognitive_details': cognitive_result,
                'recommendation': recommendation
            }
            
            st.session_state.assessment_history.append(assessment_record)
            st.success("✅ Assessment saved successfully!")
        
        # Reset for new assessment
        if st.button("🔄 Start New Assessment"):
            # Reset all assessment states
            for key in list(st.session_state.keys()):
                key_str = str(key)
                if key_str.startswith('complete_'):
                    del st.session_state[key]
                elif key_str.startswith('cognitive_'):
                    del st.session_state[key]
            st.session_state.complete_assessment_step = 'start'
            st.rerun()

def show_results_dashboard():
    """Results dashboard dengan saved assessments"""
    st.markdown("### 📈 Results Dashboard")
    
    if not st.session_state.assessment_history:
        st.warning("No saved assessments yet. Complete some assessments first!")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(st.session_state.assessment_history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Assessments", len(df))
    
    with col2:
        avg_score = df['final_score'].mean()
        st.metric("Average Final Score", f"{avg_score:.1f}/100")
    
    with col3:
        fit_count = len(df[df['final_score'] >= 70])
        st.metric("Fit to Work", f"{fit_count}/{len(df)}")
    
    with col4:
        if len(df) > 1:
            latest_score = df['final_score'].iloc[-1]
            previous_score = df['final_score'].iloc[-2]
            delta = latest_score - previous_score
            st.metric("Latest vs Previous", f"{latest_score:.1f}", delta=f"{delta:+.1f}")
        else:
            st.metric("Latest Score", f"{df['final_score'].iloc[-1]:.1f}")
    
    # Score trends
    st.markdown("#### 📈 Score Trends")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index + 1,
        y=df['final_score'],
        mode='lines+markers',
        name='Final Score',
        line=dict(color='blue', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index + 1,
        y=df['voice_score'],
        mode='lines+markers',
        name='Voice Score',
        line=dict(color='green', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index + 1,
        y=df['cognitive_score'],
        mode='lines+markers',
        name='Cognitive Score',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title="Assessment Scores Over Time",
        xaxis_title="Assessment Number",
        yaxis_title="Score",
        yaxis=dict(range=[0, 100]),
        hovermode='x'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent assessments table
    st.markdown("#### 📋 Recent Assessments")
    
    display_df = df[['timestamp', 'final_score', 'status', 'voice_score', 'cognitive_score']].copy()
    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    
    st.dataframe(display_df, use_container_width=True)
    
    # Export functionality
    if st.button("📥 Export Data as CSV"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"voice_readiness_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
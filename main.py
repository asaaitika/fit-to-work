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

# Force cloud mode detection - streamlit.app domain check
def check_audio_mode():
    # Check if running on streamlit.app domain
    try:
        # Get the current URL or check environment
        import streamlit as st
        
        # Check session state for URL (if available)
        if hasattr(st, 'get_option'):
            server_address = st.get_option('server.address')
            if server_address and 'streamlit.app' in str(server_address):
                return "streamlit_native"
        
        # Check for streamlit cloud environment variables
        if (os.getenv('STREAMLIT_SHARING') == 'true' or 
            os.getenv('STREAMLIT_CLOUD') == 'true' or
            os.getenv('HOME') == '/home/adminuser'):
            return "streamlit_native"
            
        # Force cloud mode if path contains mount/src (Streamlit Cloud signature)
        if '/mount/src' in os.getcwd():
            return "streamlit_native"
    except Exception:
        return "streamlit_native"

import streamlit as st

try:
    AUDIO_MODE = check_audio_mode()
except:
    AUDIO_MODE = check_audio_mode()

# Override for streamlit.app domain - simple hardcode approach
try:
    import urllib.parse
    if 'streamlit.app' in str(st.get_option('server.headless')):
        AUDIO_MODE = "streamlit_native"
except:
    pass

if AUDIO_MODE == "streamlit_native":
    st.info("🌐 Running in cloud mode - using web-based audio input")
else:
    st.success("🎤 Running in local mode - using direct microphone access")

import soundfile as sf
import librosa
import librosa.display
import speech_recognition as sr
import difflib
import random

# Configure Streamlit
st.set_page_config(
    page_title="Fit-to-Work Voice Checker - System",
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
            
.timer-warning {
    background-color: #fff3cd;
    color: #856404;
    padding: 0.75rem;
    border-radius: 0.25rem;
    border-left: 5px solid #ffeeba;
    margin-bottom: 1rem;
    font-weight: bold;
    animation: pulse 1.5s infinite;
}

.timer-info {
    background-color: #d1ecf1;
    color: #0c5460;
    padding: 0.5rem;
    border-radius: 0.25rem;
    margin-bottom: 1rem;
    border-left: 5px solid #bee5eb;
}

.time-result {
    font-size: 1.1rem;
    margin-top: 0.5rem;
    color: #6c757d;
    padding: 0.5rem;
    background-color: #f8f9fa;
    border-radius: 0.25rem;
    border-left: 3px solid #6c757d;
}

.time-fast {
    color: #28a745;
    border-left-color: #28a745;
}

.time-medium {
    color: #fd7e14;
    border-left-color: #fd7e14;
}

.time-slow {
    color: #dc3545;
    border-left-color: #dc3545;
}
            
/* CSS untuk memperbaiki tampilan tabel reaction time test */

/* Styling untuk tabel reaction time */
.reaction-time-table {
    width: 100%;
    border-collapse: collapse;
    margin: 15px 0;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.reaction-time-table th {
    background-color: #343a40;
    color: white;
    padding: 12px 15px;
    text-align: center;
    font-weight: bold;
    border: 1px solid #454d55;
}

.reaction-time-table td {
    padding: 10px 15px;
    border: 1px solid #454d55;
    text-align: center;
}

/* Warna baris bergantian */
.reaction-time-table tr:nth-child(odd) {
    background-color: #2c3034;
    color: white;
}

.reaction-time-table tr:nth-child(even) {
    background-color: #212529;
    color: white;
}

/* Warna untuk kategori waktu */
.reaction-time-excellent {
    color: #28a745;
    font-weight: bold;
}

.reaction-time-good {
    color: #17a2b8;
    font-weight: bold;
}

.reaction-time-average {
    color: #ffc107;
    font-weight: bold;
}

.reaction-time-slow {
    color: #dc3545;
    font-weight: bold;
}

/* Warna untuk skor */
.reaction-score {
    font-weight: bold;
    font-size: 1.1em;
}

.score-excellent {
    color: #28a745;
}

.score-good {
    color: #17a2b8;
}

.score-average {
    color: #ffc107;
}

.score-slow {
    color: #dc3545;
}

/* Ikon kategori */
.icon-excellent, .icon-good, .icon-average, .icon-slow {
    font-size: 1.2em;
    margin-right: 5px;
}

.icon-excellent {
    color: gold;
}

.icon-good {
    color: #17a2b8;
}

.icon-average {
    color: #ffc107;
}

.icon-slow {
    color: #dc3545;
}

/* Pulse animation for timer */
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.8; }
    100% { opacity: 1; }
}

/* Timer counter with progress bar */
.timer-counter {
    font-size: 1.25rem;
    font-weight: bold;
    text-align: center;
    margin: 10px 0;
    color: #dc3545;
    position: relative;
    padding: 8px;
    background-color: #f8f9fa;
    border-radius: 5px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* Advanced visual timer with progress */
.visual-timer {
    position: relative;
    height: 40px;
    margin: 15px 0;
    background-color: #f8f9fa;
    border-radius: 5px;
    overflow: hidden;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.timer-progress {
    position: absolute;
    top: 0;
    left: 0;
    height: 100%;
    background: linear-gradient(90deg, #28a745, #ffc107, #dc3545);
    transition: width 1s linear;
    z-index: 1;
}

.timer-text {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #343a40;
    font-weight: bold;
    z-index: 2;
}

/* Score impact visualization */
.score-impact {
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 15px 0;
    padding: 10px;
    background-color: #f8f9fa;
    border-radius: 5px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.max-score {
    font-size: 1.2rem;
    font-weight: bold;
    color: #28a745;
    padding: 5px 10px;
    background-color: #d4edda;
    border-radius: 3px;
    margin-right: 10px;
}

.time-penalty {
    font-size: 1.2rem;
    font-weight: bold;
    color: #dc3545;
    padding: 5px 10px;
    background-color: #f8d7da;
    border-radius: 3px;
    margin: 0 10px;
}

.final-score {
    font-size: 1.2rem;
    font-weight: bold;
    color: #007bff;
    padding: 5px 10px;
    background-color: #cce5ff;
    border-radius: 3px;
    margin-left: 10px;
}

/* Table for time impact */
.time-table {
    width: 100%;
    border-collapse: collapse;
    margin: 15px 0;
}

.time-table th {
    background-color: #f8f9fa;
    padding: 8px 12px;
    text-align: left;
    border-bottom: 2px solid #dee2e6;
}

.time-table td {
    padding: 8px 12px;
    border-bottom: 1px solid #dee2e6;
}

.time-table tr:nth-child(even) {
    background-color: #f8f9fa;
}

.time-table .time-col {
    color: #dc3545;
    font-weight: bold;
}

.time-table .score-col {
    color: #007bff;
    font-weight: bold;
}

/* Enhanced box designs */
.success-box, .warning-box, .danger-box {
    color: #ffffff;
    padding: 20px;
    border-radius: 10px;
    margin: 15px 0;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.success-box {
    background-color: #0ead69; /* Vibrant green */
    border-left: 7px solid #09874f;
}

.warning-box {
    background-color: #ff9f1c; /* Bright orange */
    border-left: 7px solid #e67e00;
}

.danger-box {
    background-color: #e63946; /* Bold red */
    border-left: 7px solid #c1121f;
}

.success-box:hover, .warning-box:hover, .danger-box:hover {
    transform: translateY(-5px);
    box-shadow: 0 6px 15px rgba(0,0,0,0.15);
}

/* Status icon animation */
.status-icon {
    display: inline-block;
    margin-right: 8px;
    animation: bounce 1s infinite alternate;
}

@keyframes bounce {
    from { transform: translateY(0); }
    to { transform: translateY(-5px); }
}            
</style>
""", unsafe_allow_html=True)

class RealVoiceChecker:
    """
    Voice Checker with cloud compatibility
    """
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.recognizer = sr.Recognizer()
        
        self.emotion_labels = {
            'ready': 'Siap & Fokus',
            'tired': 'Lelah/Mengantuk', 
            'stressed': 'Stress/Tegang',
            'calm': 'Tenang & Stabil',
            'uncertain': 'Ragu/Tidak Yakin'
        }
        
        self.sample_sentences = [
            "Saya siap kerja",
            "Keselamatan utama", 
            "Kondisi sehat",
            "Peralatan aman",
            "Tim komunikasi baik"
        ]
        
        print(f"🔧 Voice Checker initialized - Audio mode: {AUDIO_MODE}")
    
    def record_audio_streamlit_native(self):
        """Audio recording menggunakan st.audio_input untuk cloud - simplified version"""
        st.info("🎤 Record your voice using the microphone button below:")
        
        # Use Streamlit native audio input
        audio_bytes = st.audio_input("🎙️ Click to record your voice")
        
        if audio_bytes is not None:
            st.success("✅ Audio recorded successfully!")
            
            try:
                import tempfile
                import uuid
                
                temp_dir = "/tmp" if os.path.exists("/tmp") else tempfile.gettempdir()
                
                temp_filename = f"voice_record_{uuid.uuid4().hex}.wav"
                temp_path = os.path.join(temp_dir, temp_filename)
                
                with open(temp_path, 'wb') as f:
                    f.write(audio_bytes.getvalue())
                
                if not os.path.exists(temp_path):
                    st.error("Failed to save audio file")
                    return None, None
                
                st.info(f"📁 Audio saved to: {temp_path}")
                
                audio_data, sample_rate = librosa.load(temp_path, sr=self.sample_rate)
                
                return audio_data, temp_path
                
            except Exception as e:
                st.error(f"Error processing audio: {e}")
                import traceback
                st.error(f"Traceback: {traceback.format_exc()}")
                return None, None
        
        return None, None
    
    def record_audio_sounddevice(self, duration=6):
        """Original sounddevice recording for local use"""
        try:
            # Import sounddevice inside method to avoid unbound variable error
            import sounddevice as sd
            
            st.info(f"🎤 Recording selama {duration} detik...")
            st.info("📢 Ucapkan kalimat target dengan jelas!")
            
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
            
            return audio_data.flatten(), None
            
        except Exception as e:
            st.error(f"❌ Error during recording: {e}")
            st.error("Pastikan microphone terhubung dan permission diberikan")
            return None, None
    
    def record_audio_streamlit(self, duration=6):
        """Unified audio recording method"""
        if AUDIO_MODE == "sounddevice":
            return self.record_audio_sounddevice(duration)
        else:
            return self.record_audio_streamlit_native()
    
    def save_temp_audio(self, audio_data):
        """Save audio to temporary file"""
        if audio_data is None:
            return None
            
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False, dir=tempfile.gettempdir())
        sf.write(temp_file.name, audio_data, self.sample_rate)
        temp_file.close()
        return temp_file.name
    
    def speech_to_text(self, audio_file_path):
        """Speech recognition with better error handling"""
        if audio_file_path is None:
            return "NO_AUDIO"
        
        # Verify file exists
        if not os.path.exists(audio_file_path):
            st.error(f"❌ Audio file not found: {audio_file_path}")
            return "FILE_NOT_FOUND"
            
        try:
            st.info("🤖 Memproses speech recognition...")
            st.info(f"📁 Using file: {audio_file_path}")
            
            # Check file size
            file_size = os.path.getsize(audio_file_path)
            st.info(f"📊 File size: {file_size} bytes")
            
            if file_size == 0:
                st.error("❌ Audio file is empty")
                return "EMPTY_FILE"
            
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
                    text = self.recognizer.recognize_google(audio, language=lang_code) #type: ignore
                    st.success(f"✅ SUCCESS with {lang_name}: '{text}'")
                    return text.lower().strip()
                except sr.UnknownValueError:
                    st.warning(f"❌ {lang_name}: Could not understand audio")
                except sr.RequestError as e:
                    st.error(f"❌ {lang_name}: API Error - {e}")
            
            return "TIDAK_TERDETEKSI"
            
        except Exception as e:
            st.error(f"❌ Error speech recognition: {e}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return "ERROR"
    
    def calculate_pronunciation_similarity(self, target_sentence, recognized_text):
        """Calculate similarity"""
        if recognized_text in ["TIDAK_TERDETEKSI", "ERROR", "NO_AUDIO"]:
            return 0
        
        # Normalize texts
        target = target_sentence.lower().strip()
        recognized = recognized_text.lower().strip()
        
        # Calculate similarity
        similarity = difflib.SequenceMatcher(None, target, recognized).ratio()
        return round(similarity * 100, 1)
    
    def extract_voice_features(self, audio_data):
        """Extract voice features with multi-criteria null audio detection"""
        if audio_data is None:
            # Return features that indicate NO AUDIO
            return {
                'pitch_mean': 0,
                'pitch_std': 0,
                'pitch_range': 0,
                'rms_energy': 0,
                'max_amplitude': 0,
                'energy_variance': 0,
                'speech_rate': 0,
                'mfcc_mean': np.zeros(13),
                'mfcc_std': np.zeros(13),
                'spectral_centroid_mean': 0,
                'spectral_centroid_std': 0,
                'zcr_mean': 0,
                'zcr_std': 0,
                'silence_ratio': 1.0,  # 100% silence
                'audio_detected': False
            }
        
        st.info("🔍 Extracting voice features...")
        
        features = {}
        
        # Step 1: Basic amplitude and energy check
        max_amplitude = np.max(np.abs(audio_data))
        rms_energy = np.sqrt(np.mean(audio_data**2))
        
        # Step 2: Speech content analysis
        frame_length = int(0.025 * self.sample_rate)
        hop_length = int(0.01 * self.sample_rate)
        
        energy_frames = []
        for i in range(0, len(audio_data) - frame_length, hop_length):
            frame = audio_data[i:i + frame_length]
            energy = np.sum(frame**2)
            energy_frames.append(energy)
        
        energy_frames = np.array(energy_frames)
        if len(energy_frames) > 0:
            energy_threshold = max(float(np.mean(energy_frames) * 0.5), 0.001)
            speech_frames = np.sum(energy_frames > energy_threshold)
            speech_ratio = speech_frames / len(energy_frames)
        else:
            speech_ratio = 0
        
        # Step 3: Pitch analysis for validation
        pitch_mean = 0
        pitch_count = 0
        pitch_values = []
        try:
            pitches, magnitudes = librosa.piptrack(y=audio_data, sr=self.sample_rate, 
                                                  threshold=0.1, fmin=50, fmax=400)
            
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if len(pitch_values) > 0:
                pitch_mean = np.mean(pitch_values)
                pitch_count = len(pitch_values)
        except:
            pass
        
        # Step 4: Multi-criteria silence detection
        silence_criteria = []
        
        # Criteria 1: Very low energy
        if rms_energy < 0.01:
            silence_criteria.append("Low energy")
        
        # Criteria 2: Very low amplitude  
        if max_amplitude < 0.02:
            silence_criteria.append("Low amplitude")
        
        # Criteria 3: Minimal speech content
        if speech_ratio < 0.15:
            silence_criteria.append("Minimal speech")
        
        # Criteria 4: No valid pitch detected OR pitch too low for human speech
        if pitch_count == 0 or (pitch_mean > 0 and pitch_mean < 85):
            silence_criteria.append("Invalid pitch")
        
        # Criteria 5: Total frames with very low pitch detection
        if pitch_count < (len(energy_frames) * 0.1):  # Less than 10% of frames have detectable pitch
            silence_criteria.append("Insufficient pitch frames")
        
        # If 3 or more criteria indicate silence, classify as no audio
        if len(silence_criteria) >= 3:
            st.warning(f"⚠️ Audio classified as silent/noise. Criteria met: {', '.join(silence_criteria)}")
            st.info(f"📊 Debug: Energy={rms_energy:.6f}, Amplitude={max_amplitude:.6f}, Speech={speech_ratio:.1%}, Pitch={pitch_mean:.1f}Hz ({pitch_count} frames)")
            return {
                'pitch_mean': 0,
                'pitch_std': 0,
                'pitch_range': 0,
                'rms_energy': rms_energy,
                'max_amplitude': max_amplitude,
                'energy_variance': np.var(audio_data**2),
                'speech_rate': speech_ratio,
                'mfcc_mean': np.zeros(13),
                'mfcc_std': np.zeros(13),
                'spectral_centroid_mean': 0,
                'spectral_centroid_std': 0,
                'zcr_mean': 0,
                'zcr_std': 0,
                'silence_ratio': 1.0 - speech_ratio,
                'audio_detected': False
            }
        
        features['audio_detected'] = True
        st.success("✅ Valid speech audio detected for analysis")
        st.info(f"📊 Audio quality: Energy={rms_energy:.4f}, Pitch={pitch_mean:.1f}Hz, Speech={speech_ratio:.1%}")
        
        # Continue with full feature extraction for valid audio
        if len(pitch_values) > 0:
            features['pitch_mean'] = np.mean(pitch_values)
            features['pitch_std'] = np.std(pitch_values)
            features['pitch_range'] = np.max(pitch_values) - np.min(pitch_values)
        else:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
            features['pitch_range'] = 0
        
        # Energy Analysis
        features['rms_energy'] = rms_energy
        features['max_amplitude'] = max_amplitude
        features['energy_variance'] = np.var(audio_data**2)
        features['speech_rate'] = speech_ratio
        
        # Spectral Features
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
            features['spectral_centroid_mean'] = 0
            features['zcr_mean'] = 0
        
        # Temporal Features
        silence_threshold = features['rms_energy'] * 0.1
        silent_frames = np.sum(energy_frames < silence_threshold)
        features['silence_ratio'] = silent_frames / len(energy_frames) if len(energy_frames) > 0 else 1.0
        
        st.success(f"✅ Extracted {len(features)} voice features")
        return features
    
    def analyze_emotion_patterns(self, features):
        """Analyze emotion patterns with null audio detection"""
        st.info("🧠 Analyzing emotion patterns...")
        
        # Check if audio was actually detected
        if not features.get('audio_detected', True):
            st.warning("⚠️ No audio detected - returning baseline scores")
            return {
                'ready': 0,
                'tired': 0,
                'stressed': 0,
                'calm': 0,
                'uncertain': 100  # High uncertainty for no audio
            }
        
        emotion_scores = {}
        
        # Analyze READY/FOCUSED patterns (more generous scoring)
        ready_score = 0
        if 100 < features['pitch_mean'] < 350:  # Wider range
            ready_score += 25
        if features['pitch_std'] < 60:  # More generous
            ready_score += 20
        if 0.005 < features['rms_energy'] < 0.15:  # Wider energy range
            ready_score += 25
        if features['speech_rate'] > 0.2:  # Lower threshold
            ready_score += 15
        if features['spectral_centroid_mean'] > 800:  # Lower threshold
            ready_score += 10
        if features['silence_ratio'] < 0.8:  # More generous
            ready_score += 5
        emotion_scores['ready'] = min(ready_score, 100)
        
        # Analyze TIRED patterns (more restrictive)
        tired_score = 0
        if features['pitch_mean'] < 120:  # Very low pitch only
            tired_score += 30
        if features['rms_energy'] < 0.015:  # Very low energy only
            tired_score += 35
        if features['speech_rate'] < 0.25:  # Very slow speech only
            tired_score += 25
        if features['silence_ratio'] > 0.7:  # Lots of silence
            tired_score += 10
        emotion_scores['tired'] = min(tired_score, 100)
        
        # Analyze STRESSED patterns (more restrictive)
        stressed_score = 0
        if features['pitch_mean'] > 300:  # Very high pitch only
            stressed_score += 30
        if features['pitch_std'] > 80:  # Very high variation only
            stressed_score += 30
        if features['energy_variance'] > 0.015:  # High energy variation
            stressed_score += 25
        if features['zcr_mean'] > 0.12:  # High zero crossing rate
            stressed_score += 15
        emotion_scores['stressed'] = min(stressed_score, 100)
        
        # Analyze CALM patterns (more generous)
        calm_score = 0
        if 120 < features['pitch_mean'] < 250:  # Normal pitch range
            calm_score += 30
        if features['pitch_std'] < 40:  # Stable pitch
            calm_score += 25
        if features['energy_variance'] < 0.008:  # Stable energy
            calm_score += 25
        if 0.3 < features['speech_rate'] < 0.8:  # Normal speech rate
            calm_score += 20
        emotion_scores['calm'] = min(calm_score, 100)
        
        # Analyze UNCERTAIN patterns (more restrictive)
        uncertain_score = 0
        if features['pitch_std'] > 60:  # High pitch variation
            uncertain_score += 35
        if features['silence_ratio'] > 0.75:  # Lots of pauses
            uncertain_score += 35
        if features['speech_rate'] < 0.3:  # Very hesitant speech
            uncertain_score += 20
        if features['spectral_centroid_mean'] < 1200:  # Unclear speech
            uncertain_score += 10
        emotion_scores['uncertain'] = min(uncertain_score, 100)
        
        return emotion_scores
    
    def determine_work_readiness(self, emotion_scores, pronunciation_score=None):
        """Determine work readiness with improved formula and null audio handling"""
        st.info("⚖️ Determining work readiness...")
        
        # Check for no audio detected case
        if emotion_scores.get('uncertain', 0) == 100 and all(score == 0 for key, score in emotion_scores.items() if key != 'uncertain'):
            st.warning("⚠️ No audio detected - assessment cannot be completed")
            return {
                'readiness_score': 0,
                'base_score': 0,
                'pronunciation_penalty': 0,
                'dominant_emotion': 'uncertain',
                'dominant_score': 100,
                'status': "BELUM SIAP",
                'recommendation': "Tidak ada audio terdeteksi - silakan ulangi recording",
                'color': "🔴",
                'emotion_breakdown': emotion_scores
            }
        
        # 1. Get dominant emotion first
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        dominant_name, dominant_score = dominant_emotion
        
        # 2. Base score from dominant emotion with weighted approach
        if dominant_name == 'ready':
            base_readiness_score = 70 + (dominant_score * 0.3)  # 70-100 range
        elif dominant_name == 'calm':
            base_readiness_score = 60 + (dominant_score * 0.4)  # 60-100 range
        elif dominant_name == 'tired':
            base_readiness_score = 30 - (dominant_score * 0.3)  # 0-30 range
        elif dominant_name == 'stressed':
            base_readiness_score = 40 - (dominant_score * 0.4)  # 0-40 range
        else:  # uncertain
            base_readiness_score = 45 - (dominant_score * 0.45) # 0-45 range
        
        # 3. Secondary emotions modifier (smaller impact)
        positive_boost = (emotion_scores['ready'] + emotion_scores['calm']) * 0.1
        negative_penalty = (emotion_scores['tired'] + emotion_scores['stressed'] + emotion_scores['uncertain']) * 0.05
        
        base_readiness_score = base_readiness_score + positive_boost - negative_penalty
        base_readiness_score = max(0, min(100, base_readiness_score))
        
        # 4. Apply pronunciation penalty if provided
        if pronunciation_score is not None:
            # Reduced pronunciation penalty (less harsh)
            if pronunciation_score < 20:
                pronunciation_penalty = 25  # Heavy penalty
            elif pronunciation_score < 40:
                pronunciation_penalty = 15  # Moderate penalty  
            elif pronunciation_score < 60:
                pronunciation_penalty = 8   # Light penalty
            else:
                pronunciation_penalty = 0   # No penalty
            
            final_readiness_score = base_readiness_score - pronunciation_penalty
            st.info(f"📢 Pronunciation penalty applied: -{pronunciation_penalty} points")
        else:
            final_readiness_score = base_readiness_score
            pronunciation_penalty = 0
        
        final_readiness_score = max(0, min(100, final_readiness_score))
        
        # 5. Enhanced status determination
        if final_readiness_score >= 75:
            if pronunciation_score is not None and pronunciation_score < 40:
                status = "CUKUP SIAP"  # Downgrade due to poor communication
                recommendation = "Kondisi mental baik, tapi perlu perhatian komunikasi"
                color = "🟡"
            else:
                status = "SIAP KERJA"
                recommendation = "Kondisi mental dan fisik baik untuk bekerja"
                color = "🟢"
        elif final_readiness_score >= 55:
            status = "CUKUP SIAP"
            recommendation = "Bisa bekerja, tapi perhatikan kondisi diri"
            color = "🟡"
        else:
            status = "BELUM SIAP"
            recommendation = "Sebaiknya istirahat dulu atau konsultasi supervisor"
            color = "🔴"
        
        return {
            'readiness_score': round(final_readiness_score, 1),
            'base_score': round(base_readiness_score, 1),
            'pronunciation_penalty': pronunciation_penalty,
            'dominant_emotion': dominant_name,
            'dominant_score': dominant_score,
            'status': status,
            'recommendation': recommendation,
            'color': color,
            'emotion_breakdown': emotion_scores
        }
    
    def get_random_sentence(self):
        """Get random sentence"""
        return random.choice(self.sample_sentences)
    
    def cleanup_temp_file(self, file_path):
        """Cleanup temporary file with better error handling"""
        if file_path and os.path.exists(file_path):
            try:
                os.unlink(file_path)
                st.info(f"🗑️ Cleaned up: {file_path}")
            except Exception as e:
                st.warning(f"⚠️ Could not delete temp file: {e}")

class RealCognitiveAssessment:
    """Cognitive assessment dengan interactive Streamlit UI"""
    
    def __init__(self):
        self.cognitive_tests = {
            'attention': self.attention_test_streamlit,
            'memory': self.memory_test_streamlit, 
            'reaction': self.reaction_time_test_streamlit,
            'math': self.simple_math_test_streamlit,
            'sequence': self.sequence_test_streamlit
        }
        print("🧠 Cognitive Assessment Module initialized")

    def attention_test_streamlit(self):
        """Attention test dengan timer helper"""
        
        timer_helper = TimerHelper()
        
        st.markdown(
            timer_helper.create_timer_warning(
                "Hitung semua huruf dengan cepat dan teliti.", 
                5
            ), 
            unsafe_allow_html=True
        )
        
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
        
        current_time = time.time()
        elapsed_time = current_time - st.session_state.attention_start_time
        max_time = 15  # Waktu maksimal untuk tes atensi
        
        # Render visual timer
        st.markdown(
            timer_helper.create_visual_timer(elapsed_time, max_time),
            unsafe_allow_html=True
        )
        
        penalty = int(elapsed_time * 5)
        current_score = max(0, 100 - penalty)
        
        st.markdown(
            timer_helper.create_score_impact(100, penalty, current_score),
            unsafe_allow_html=True
        )
        
        st.markdown(
            timer_helper.create_auto_refresh_timer(
                st.session_state.attention_start_time,
                max_time=max_time
            ),
            unsafe_allow_html=True
        )
        
        user_answer = st.number_input("Berapa huruf 'A'?", min_value=0, max_value=50, value=0)
        
        if st.button("Submit Answer", type="primary"):
            response_time = time.time() - st.session_state.attention_start_time
            
            if user_answer == correct_count:
                penalty = int(response_time * 5)
                score = max(0, 100 - penalty)
                result = "✅ BENAR"
                st.success(f"{result} - Score: {score}/100")
                
                st.markdown(
                    timer_helper.create_time_result(
                        response_time,
                        penalty,
                        100,
                        score,
                        'attention'
                    ),
                    unsafe_allow_html=True
                )
                
                if response_time < 8:
                    st.markdown("""
                    <div style="background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px; margin-top: 10px;">
                        <strong>🏆 Excellent!</strong> Kecepatan Anda sangat baik. Pertahankan!
                    </div>
                    """, unsafe_allow_html=True)
                elif response_time < 12:
                    st.markdown("""
                    <div style="background-color: #fff3cd; color: #856404; padding: 10px; border-radius: 5px; margin-top: 10px;">
                        <strong>👍 Good!</strong> Coba tingkatkan sedikit lagi kecepatan untuk skor lebih tinggi.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; margin-top: 10px;">
                        <strong>💪 Perlu ditingkatkan!</strong> Latih kemampuan observasi cepat untuk meningkatkan skor.
                    </div>
                    """, unsafe_allow_html=True)
                
            else:
                penalty = int(response_time * 3)
                score = max(0, 50 - penalty)
                result = f"❌ SALAH (jawaban: {correct_count})"
                st.error(f"{result} - Score: {score}/100")
                
                st.markdown(
                    f"""
                    <div class="time-result time-slow">
                        ⏱️ <strong>Analisis Waktu:</strong>
                        <br>Waktu pengerjaan: {response_time:.2f} detik
                        <br>Penalti waktu: -{penalty} poin
                        <br>Skor dasar untuk jawaban salah: 50/100
                        <br>Skor akhir dengan penalti waktu: {score}/100
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                st.markdown("""
                <div style="background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; margin-top: 10px;">
                    <strong>🔄 Coba lagi!</strong> Fokus pada keseimbangan antara kecepatan dan akurasi.
                </div>
                """, unsafe_allow_html=True)
            
            del st.session_state.attention_text
            del st.session_state.attention_start_time
            
            return {
                'test_type': 'attention',
                'score': score,
                'response_time': round(response_time, 2),
                'correct': user_answer == correct_count,
                'result': result
            }
        
        time.sleep(0.1)
        st.rerun()
        
        return None
    
    def memory_test_streamlit(self):
        """Memory test dengan Streamlit UI"""
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
        """Reaction time test dengan tabel yang lebih mudah dilihat"""
        
        st.markdown("""
        <div class="timer-warning">
            ⏱️ PERHATIAN: Test ini mengukur KECEPATAN REAKSI Anda!
            <br>Skor berdasarkan seberapa cepat Anda bereaksi terhadap stimulus.
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("ℹ️ Cara Penilaian Waktu Reaksi"):
            st.markdown("""
            <table class="reaction-time-table">
                <tr>
                    <th>Waktu Reaksi</th>
                    <th>Kategori</th>
                    <th>Skor</th>
                </tr>
                <tr>
                    <td class="reaction-time-excellent">< 0.5 detik</td>
                    <td><span class="icon-excellent">🏆</span> EXCELLENT</td>
                    <td class="reaction-score score-excellent">100/100</td>
                </tr>
                <tr>
                    <td class="reaction-time-good">0.5 - 1.0 detik</td>
                    <td><span class="icon-good">✅</span> GOOD</td>
                    <td class="reaction-score score-good">80/100</td>
                </tr>
                <tr>
                    <td class="reaction-time-average">1.0 - 2.0 detik</td>
                    <td><span class="icon-average">⚠️</span> AVERAGE</td>
                    <td class="reaction-score score-average">60/100</td>
                </tr>
                <tr>
                    <td class="reaction-time-slow">> 2.0 detik</td>
                    <td><span class="icon-slow">❌</span> SLOW</td>
                    <td class="reaction-score score-slow">30/100</td>
                </tr>
            </table>
            
            <p style="margin-top: 15px; color: #e9ecef;">
            <strong>Tips:</strong> Fokuskan pandangan Anda ke layar dan persiapkan jari Anda di atas tombol untuk reaksi tercepat!
            </p>
            """, unsafe_allow_html=True)
        
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
                st.markdown("""
                <div style="text-align: center; margin: 20px 0; animation: pulse 0.5s infinite alternate;">
                    <h1 style="font-size: 5rem; color: #dc3545;">🚨 GO!</h1>
                </div>
                """, unsafe_allow_html=True)
                
                # Timer mulai berjalan dari muncul GO!
                reaction_timer_start = time.time() - (st.session_state.reaction_start_time + st.session_state.reaction_delay)
                
                # Visual timer untuk reaction
                st.markdown(f"""
                <div class="visual-timer" style="background-color: #f8d7da;">
                    <div class="timer-progress" style="background: linear-gradient(90deg, #28a745, #dc3545); width: {min(100, reaction_timer_start * 100)}%;"></div>
                    <div class="timer-text">⏱️ {reaction_timer_start:.3f} detik</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <script>
                    function updateReactionTimer() {
                        const startTime = %s;
                        const timerElement = document.querySelector('.timer-text');
                        const progressBar = document.querySelector('.timer-progress');
                        
                        if (timerElement && progressBar) {
                            setInterval(() => {
                                const currentTime = new Date().getTime() / 1000;
                                const elapsedTime = currentTime - startTime;
                                timerElement.textContent = `⏱️ ${elapsedTime.toFixed(3)} detik`;
                                
                                // Update progress width - max at 100%%
                                const width = Math.min(100, elapsedTime * 100);
                                progressBar.style.width = width + '%%';
                                
                                // Change color based on time
                                if(elapsedTime < 0.5) {
                                    timerElement.style.color = '#28a745';
                                } else if(elapsedTime < 1.0) {
                                    timerElement.style.color = '#ffc107';
                                } else {
                                    timerElement.style.color = '#dc3545';
                                }
                            }, 10); // Update every 10ms for smoother timer
                        }
                    }
                    
                    document.addEventListener('DOMContentLoaded', updateReactionTimer);
                </script>
                """ % (time.time()), unsafe_allow_html=True)
                
                if st.button("REACT!", type="primary", use_container_width=True):
                    reaction_time = time.time() - (st.session_state.reaction_start_time + st.session_state.reaction_delay)
                    
                    # Determine score and class based on reaction time
                    if reaction_time < 0.5:
                        score = 100
                        result = "🏆 EXCELLENT"
                        time_class = "time-fast"
                        icon_class = "icon-excellent"
                    elif reaction_time < 1.0:
                        score = 80
                        result = "✅ GOOD"
                        time_class = "time-medium"
                        icon_class = "icon-good"
                    elif reaction_time < 2.0:
                        score = 60
                        result = "⚠️ AVERAGE"
                        time_class = "time-slow"
                        icon_class = "icon-average"
                    else:
                        score = 30
                        result = "❌ SLOW"
                        time_class = "time-slow"
                        icon_class = "icon-slow"
                    
                    st.success(f"{result} - Reaction time: {reaction_time:.3f}s - Score: {score}/100")
                    
                    st.markdown(f"""
                    <div class="time-result {time_class}">
                        <strong>⏱️ Analisis Waktu Reaksi:</strong> {reaction_time:.3f} detik
                    </div>
                    
                    <div class="score-impact">
                        <div class="final-score">Skor: {score}/100</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div style="margin-top: 15px; padding: 15px; border-radius: 8px; background-color: #212529; color: white;">
                        <h3>Hasil Reaction Time Test</h3>
                        <div style="display: flex; align-items: center; margin-top: 10px;">
                            <div style="flex: 1;">
                                <span class="{icon_class}" style="font-size: 2rem;">{result.split()[0]}</span>
                            </div>
                            <div style="flex: 2;">
                                <div style="font-size: 1.2rem; font-weight: bold;">{result.split()[1]}</div>
                                <div style="font-size: 0.9rem; opacity: 0.8;">Kategori</div>
                            </div>
                            <div style="flex: 1; text-align: right;">
                                <div style="font-size: 1.5rem; font-weight: bold;" class="reaction-score score-{result.split()[1].lower()}">{score}/100</div>
                                <div style="font-size: 0.9rem; opacity: 0.8;">Skor</div>
                            </div>
                        </div>
                        <div style="margin-top: 15px; background-color: #2c3034; padding: 10px; border-radius: 5px;">
                            <div style="font-weight: bold; margin-bottom: 5px;">Waktu Reaksi Anda:</div>
                            <div style="font-size: 1.8rem; font-weight: bold; color: #{time_class.replace('time-', '') == 'fast' and '28a745' or time_class.replace('time-', '') == 'medium' and 'ffc107' or 'dc3545'};">
                                {reaction_time:.3f} detik
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Reaction time comparison
                    st.markdown("""
                    <div style="margin-top: 15px; padding: 15px; border-radius: 8px; background-color: #212529; color: white;">
                        <h3>Perbandingan Waktu Reaksi</h3>
                        <table class="reaction-time-table" style="margin-top: 10px;">
                            <tr>
                                <th>Kelompok</th>
                                <th>Waktu Reaksi Tipikal</th>
                            </tr>
                            <tr>
                                <td>Atlet profesional</td>
                                <td class="reaction-time-excellent">0.1 - 0.2 detik</td>
                            </tr>
                            <tr>
                                <td>Rata-rata orang</td>
                                <td class="reaction-time-good">0.2 - 0.5 detik</td>
                            </tr>
                            <tr>
                                <td>Saat lelah/mengantuk</td>
                                <td class="reaction-time-average">0.5 - 1.0 detik</td>
                            </tr>
                            <tr>
                                <td>Setelah konsumsi alkohol</td>
                                <td class="reaction-time-slow">1.0+ detik</td>
                            </tr>
                        </table>
                    </div>
                    """, unsafe_allow_html=True)
                    
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
                
                time.sleep(0.01)
                st.rerun()
            else:
                # Waiting message with countdown
                remaining = st.session_state.reaction_delay - elapsed
                
                if remaining > 0.5:
                    st.write("Tunggu instruksi...")
                    
                    st.markdown("""
                    <div style="text-align: center; margin: 20px 0;">
                        <div style="font-size: 1.5rem; color: #e9ecef;">
                            Bersiap...
                        </div>
                        <div style="display: flex; justify-content: center; margin-top: 10px;">
                            <div style="width: 12px; height: 12px; background-color: #007bff; border-radius: 50%; margin: 0 5px; animation: bounce 0.6s infinite alternate;"></div>
                            <div style="width: 12px; height: 12px; background-color: #007bff; border-radius: 50%; margin: 0 5px; animation: bounce 0.6s 0.2s infinite alternate;"></div>
                            <div style="width: 12px; height: 12px; background-color: #007bff; border-radius: 50%; margin: 0 5px; animation: bounce 0.6s 0.4s infinite alternate;"></div>
                        </div>
                    </div>
                    
                    <style>
                    @keyframes bounce {
                        from { transform: translateY(0); }
                        to { transform: translateY(-10px); }
                    }
                    </style>
                    """, unsafe_allow_html=True)
                
                time.sleep(0.1)
                st.rerun()
        
        return None
    

    def simple_math_test_streamlit(self):
        """Math test with advanced visual timer and score impact visualization"""
        
        st.markdown("""
        <div class="timer-warning">
            ⏱️ PERHATIAN: Test ini menggunakan WAKTU! Semakin cepat menjawab, semakin tinggi skor Anda.
            <br>Setiap detik akan mengurangi 10 poin untuk jawaban benar.
        </div>
        """, unsafe_allow_html=True)
        
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
        
        # Advanced visual timer
        current_time = time.time()
        elapsed_time = current_time - st.session_state.math_start_time
        max_time = 10  # seconds
        progress_percent = min(100, (elapsed_time / max_time) * 100)
        
        # Color changes based on time
        if elapsed_time < 3:
            timer_color = "#28a745"
            time_class = "time-fast"
        elif elapsed_time < 6:
            timer_color = "#ffc107"
            time_class = "time-medium"
        else:
            timer_color = "#dc3545"
            time_class = "time-slow"
        
        st.markdown(f"""
        <div class="visual-timer">
            <div class="timer-progress" style="width: {progress_percent}%;"></div>
            <div class="timer-text">⏱️ {elapsed_time:.1f} detik</div>
        </div>
        
        <div class="score-impact">
            <div class="max-score">Start: 100</div>
            <div class="time-penalty">-{int(elapsed_time * 10)} poin</div>
            <div class="final-score">Current: {max(0, 100 - int(elapsed_time * 10))}</div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Lihat pengaruh waktu pada skor"):
            st.markdown("""
            <table class="time-table">
                <tr>
                    <th>Waktu (detik)</th>
                    <th>Penalti</th>
                    <th>Skor Maksimal</th>
                </tr>
                <tr>
                    <td class="time-col">1</td>
                    <td>-10 poin</td>
                    <td class="score-col">90/100</td>
                </tr>
                <tr>
                    <td class="time-col">3</td>
                    <td>-30 poin</td>
                    <td class="score-col">70/100</td>
                </tr>
                <tr>
                    <td class="time-col">5</td>
                    <td>-50 poin</td>
                    <td class="score-col">50/100</td>
                </tr>
                <tr>
                    <td class="time-col">7</td>
                    <td>-70 poin</td>
                    <td class="score-col">30/100</td>
                </tr>
                <tr>
                    <td class="time-col">10+</td>
                    <td>-100 poin</td>
                    <td class="score-col">0/100</td>
                </tr>
            </table>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <script>
            // Auto-refresh function
            function refreshTimer() {
                const currentTime = new Date().getTime() / 1000;
                const startTime = %s;
                const elapsedTime = currentTime - startTime;
                const maxTime = 10;
                
                // Update progress bar
                const progressPercent = Math.min(100, (elapsedTime / maxTime) * 100);
                const progressBar = document.querySelector('.timer-progress');
                const timerText = document.querySelector('.timer-text');
                const timePenalty = document.querySelector('.time-penalty');
                const finalScore = document.querySelector('.final-score');
                
                if(progressBar && timerText) {
                    progressBar.style.width = progressPercent + '%%';
                    timerText.textContent = '⏱️ ' + elapsedTime.toFixed(1) + ' detik';
                    
                    // Update penalty and score
                    if(timePenalty && finalScore) {
                        const penalty = Math.floor(elapsedTime * 10);
                        const currentScore = Math.max(0, 100 - penalty);
                        timePenalty.textContent = '-' + penalty + ' poin';
                        finalScore.textContent = 'Current: ' + currentScore;
                    }
                    
                    // Change color based on time
                    if(elapsedTime < 3) {
                        timerText.style.color = '#28a745';
                    } else if(elapsedTime < 6) {
                        timerText.style.color = '#ffc107';
                    } else {
                        timerText.style.color = '#dc3545';
                    }
                }
                
                // Refresh every 100ms for smooth animation
                setTimeout(refreshTimer, 100);
            }
            
            // Start the refresh cycle
            document.addEventListener('DOMContentLoaded', refreshTimer);
        </script>
        """ % st.session_state.math_start_time, unsafe_allow_html=True)
        
        user_answer = st.number_input("Jawaban:", value=0)
        
        submit_col1, submit_col2 = st.columns([1, 3])
        
        with submit_col1:
            submit_button = st.button("Submit", type="primary")
        
        with submit_col2:
            st.markdown(f"""
            <div style="padding-top: 3px;">
                Jawaban cepat = skor tinggi!
            </div>
            """, unsafe_allow_html=True)
        
        if submit_button:
            response_time = time.time() - st.session_state.math_start_time
            
            # Determine time classification
            time_class = "time-fast" if response_time < 5 else "time-medium" if response_time < 10 else "time-slow"
            
            if user_answer == problem['answer']:
                score = max(0, 100 - int(response_time * 10))
                result = "✅ BENAR"
                st.success(f"{result} - Score: {score}/100")
                
                st.markdown(f"""
                <div class="time-result {time_class}">
                    <strong>⏱️ Analisis Waktu Pengerjaan:</strong>
                </div>
                
                <div class="score-impact">
                    <div class="max-score">Base: 100</div>
                    <div class="time-penalty">-{int(response_time * 10)} poin</div>
                    <div class="final-score">Final: {score}</div>
                </div>
                
                <p>Anda menyelesaikan soal dalam {response_time:.2f} detik. 
                Setiap detik mengurangi 10 poin dari skor maksimal.</p>
                """, unsafe_allow_html=True)
                
                if response_time < 3:
                    st.markdown("""
                    <div style="background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px; margin-top: 10px;">
                        <strong>🏆 Excellent!</strong> Kecepatan Anda sangat baik. Pertahankan!
                    </div>
                    """, unsafe_allow_html=True)
                elif response_time < 6:
                    st.markdown("""
                    <div style="background-color: #fff3cd; color: #856404; padding: 10px; border-radius: 5px; margin-top: 10px;">
                        <strong>👍 Good!</strong> Coba tingkatkan sedikit lagi kecepatan untuk skor lebih tinggi.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; margin-top: 10px;">
                        <strong>💪 Perlu ditingkatkan!</strong> Latih kemampuan berhitung cepat untuk meningkatkan skor.
                    </div>
                    """, unsafe_allow_html=True)
                    
            else:
                score = max(0, 30 - int(response_time * 5))
                result = f"❌ SALAH (jawaban: {problem['answer']})"
                st.error(f"{result} - Score: {score}/100")
                
                st.markdown(f"""
                <div class="time-result {time_class}">
                    <strong>⏱️ Analisis Waktu Pengerjaan:</strong>
                </div>
                
                <div class="score-impact">
                    <div class="max-score">Base: 30</div>
                    <div class="time-penalty">-{int(response_time * 5)} poin</div>
                    <div class="final-score">Final: {score}</div>
                </div>
                
                <p>Anda menyelesaikan soal dalam {response_time:.2f} detik. 
                Untuk jawaban salah, setiap detik mengurangi 5 poin dari skor dasar 30.</p>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div style="background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; margin-top: 10px;">
                    <strong>🔄 Coba lagi!</strong> Fokus pada keseimbangan antara kecepatan dan akurasi.
                </div>
                """, unsafe_allow_html=True)
            
            del st.session_state.math_problem
            del st.session_state.math_start_time
            
            return {
                'test_type': 'math',
                'score': score,
                'response_time': round(response_time, 2),
                'correct': user_answer == problem['answer'],
                'result': result
            }
        
        time.sleep(0.1)
        st.rerun()
        
        return None

    def sequence_test_streamlit(self):
        """Sequence test"""
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

class TimerHelper:
    """Helper class untuk menstandarisasi timer di semua cognitive test"""
    
    @staticmethod
    def create_timer_warning(message, penalty):
        """Membuat peringatan timer standard"""
        return f"""
        <div class="timer-warning">
            ⏱️ PERHATIAN: Test ini menggunakan WAKTU! Semakin cepat menjawab, semakin tinggi skor Anda.
            <br>{message}
            <br>Setiap detik akan mengurangi {penalty} poin untuk jawaban benar.
        </div>
        """
    
    @staticmethod
    def create_timer_info(benar_penalty, salah_penalty, target):
        """Membuat info timer standard"""
        return f"""
        <div class="timer-info">
            ⏱️ <strong>Info Waktu:</strong> -{benar_penalty} poin per detik (benar), -{salah_penalty} poin per detik (salah)
            <br>Target waktu ideal: {target}
        </div>
        """
    
    @staticmethod
    def create_visual_timer(elapsed_time, max_time=10):
        """Membuat visual timer dengan progress bar"""
        progress_percent = min(100, (elapsed_time / max_time) * 100)
        
        # Color changes based on time
        if elapsed_time < max_time * 0.3:  # First 30%
            timer_color = "#28a745"
            time_class = "time-fast"
        elif elapsed_time < max_time * 0.6:  # 30-60%
            timer_color = "#ffc107"
            time_class = "time-medium"
        else:  # > 60%
            timer_color = "#dc3545"
            time_class = "time-slow"
        
        return f"""
        <div class="visual-timer">
            <div class="timer-progress" style="width: {progress_percent}%;"></div>
            <div class="timer-text" style="color: {timer_color};">⏱️ {elapsed_time:.1f} detik</div>
        </div>
        """
    
    @staticmethod
    def create_score_impact(base_score, penalty, current_score):
        """Membuat visualisasi dampak skor"""
        return f"""
        <div class="score-impact">
            <div class="max-score">Base: {base_score}</div>
            <div class="time-penalty">-{penalty} poin</div>
            <div class="final-score">Current: {current_score}</div>
        </div>
        """
    
    @staticmethod
    def create_time_result(response_time, penalty, max_score, final_score, test_type):
        """Membuat hasil analisis waktu"""
        
        # Determine time class based on test type
        time_class = "time-medium"  # default
        
        if test_type == 'math':
            time_class = "time-fast" if response_time < 3 else "time-medium" if response_time < 6 else "time-slow"
        elif test_type == 'sequence':
            time_class = "time-fast" if response_time < 5 else "time-medium" if response_time < 8 else "time-slow"
        elif test_type == 'attention':
            time_class = "time-fast" if response_time < 8 else "time-medium" if response_time < 12 else "time-slow"
        elif test_type == 'memory':
            time_class = "time-fast" if response_time < 10 else "time-medium" if response_time < 15 else "time-slow"
        elif test_type == 'reaction':
            time_class = "time-fast" if response_time < 0.5 else "time-medium" if response_time < 1.0 else "time-slow"
        
        return f"""
        <div class="time-result {time_class}">
            ⏱️ <strong>Analisis Waktu:</strong>
            <br>Waktu pengerjaan: {response_time:.2f} detik
            <br>Penalti waktu: -{penalty} poin
            <br>Skor maksimal tanpa penalti waktu: {max_score}/100
            <br>Skor akhir dengan penalti waktu: {final_score}/100
        </div>
        """
    
    @staticmethod
    def create_auto_refresh_timer(start_time, element_selector='.timer-text', progress_selector='.timer-progress', max_time=10):
        """Membuat JavaScript untuk auto-refresh timer"""
        return f"""
        <script>
            // Auto-refresh function
            function refreshTimer() {{
                const currentTime = new Date().getTime() / 1000;
                const startTime = {start_time};
                const elapsedTime = currentTime - startTime;
                const maxTime = {max_time};
                
                // Update progress bar
                const progressPercent = Math.min(100, (elapsedTime / maxTime) * 100);
                const progressBar = document.querySelector('{progress_selector}');
                const timerText = document.querySelector('{element_selector}');
                
                if(progressBar && timerText) {{
                    progressBar.style.width = progressPercent + '%';
                    timerText.textContent = '⏱️ ' + elapsedTime.toFixed(1) + ' detik';
                    
                    // Change color based on time
                    if(elapsedTime < maxTime * 0.3) {{
                        timerText.style.color = '#28a745';
                    }} else if(elapsedTime < maxTime * 0.6) {{
                        timerText.style.color = '#ffc107';
                    }} else {{
                        timerText.style.color = '#dc3545';
                    }}
                }}
                
                // Refresh every 100ms for smooth animation
                setTimeout(refreshTimer, 100);
            }}
            
            // Start the refresh cycle
            document.addEventListener('DOMContentLoaded', refreshTimer);
        </script>
        """
    
    @staticmethod
    def get_time_penalty_info(test_type, is_correct=True):
        """Mendapatkan informasi penalti waktu berdasarkan jenis tes"""
        if test_type == 'math':
            return 10 if is_correct else 5, 3
        elif test_type == 'sequence':
            return 8 if is_correct else 4, 5
        elif test_type == 'attention':
            return 5 if is_correct else 3, 8
        elif test_type == 'memory':
            return 3 if is_correct else 2, 10
        elif test_type == 'reaction':
            return 0, 0.5  # Reaction menggunakan threshold
        return 5, 5  # Default fallback

# Main Streamlit Application
def main():
    """Main Streamlit application"""
    
    st.markdown('<h1 class="main-header">🏭 Fit-to-Work Voice Readiness Checker</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'assessment_history' not in st.session_state:
        st.session_state.assessment_history = []
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'home'

    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    
    # Map session state to selectbox options
    page_mapping = {
        'home': "🏠 Home",
        'voice': "🎤 Voice Analysis Test",
        'cognitive': "🧠 Cognitive Test",
        'complete': "📊 Complete Assessment",
        'dashboard': "📈 Results Dashboard",
    }
    
    # Set selectbox based on current_page session state
    current_selectbox_value = page_mapping.get(st.session_state.current_page, "🏠 Home")
    
    # Sidebar selectbox
    if 'previous_page' not in st.session_state:
        st.session_state.previous_page = current_selectbox_value

    page = st.sidebar.selectbox(
        "Choose Mode:", 
        list(page_mapping.values()),
        index=list(page_mapping.values()).index(current_selectbox_value)
    )

    # Check if page has changed
    if page != st.session_state.previous_page:
        st.session_state.previous_page = page
        # Force immediate update
        reverse_mapping = {v: k for k, v in page_mapping.items()}
        st.session_state.current_page = reverse_mapping[page]
        st.rerun()
    
    # Update session state when selectbox changes
    reverse_mapping = {v: k for k, v in page_mapping.items()}
    new_page_key = reverse_mapping[page]
    
    st.session_state.current_page = new_page_key
    
    # Route based on current page
    if st.session_state.current_page == 'home':
        show_home_page()
    elif st.session_state.current_page == 'voice':
        show_real_voice_analysis()
    elif st.session_state.current_page == 'cognitive':
        show_real_cognitive_tests()
    elif st.session_state.current_page == 'complete':
        show_real_complete_assessment()
    else:  # dashboard
        show_results_dashboard()

def show_home_page():
    """Home page dengan overview"""
    st.markdown("### 🎯 System Overview")

    st.markdown("#### 📋 Description")
    st.markdown("""
    **Fit-to-Work Voice Readiness Checker** adalah sistem otomatis untuk mengevaluasi kesiapan pekerja 
    sebelum memulai shift kerja melalui analisis suara dan kognitif yang komprehensif. 
    
    Sistem ini dirancang untuk **meningkatkan keselamatan kerja** dengan mengidentifikasi 
    potensi masalah pada pekerja sebelum mereka memulai aktivitas yang berisiko.
    """)
    
    if AUDIO_MODE == "streamlit_native":
        st.info("""
        **Cloud Mode Requirements:**
        - Modern web browser dengan microphone support
        - Internet connection untuk Google Speech API
        - Permission untuk akses microphone saat diminta browser
        """)
    else:
        st.warning("""
        **Local Mode Requirements:**
        - Microphone yang berfungsi
        - Internet connection untuk Google Speech API
        - Permission untuk akses microphone
        """)
    
    st.markdown("#### 🚀 Quick Start")
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
    
    st.markdown("#### 🔧 System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Test microphone availability
        if AUDIO_MODE == "sounddevice":
            try:
                import sounddevice as sd
                devices = sd.query_devices()
                input_devices = [d for d in devices if d.get('max_input_channels', 0) > 0]
                if input_devices:
                    st.success("🎤 Microphone: OK")
                else:
                    st.error("🎤 Microphone: Not Found")
            except:
                st.error("🎤 Microphone: Error")
        else:
            st.info("🎤 Web Audio: Ready")
    
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
    """voice analysis page"""
    st.markdown("### 🎤 Voice Analysis")
    
    checker = RealVoiceChecker()
    
    # Step 1: Select target sentence
    st.markdown("#### Step 1: Select Target Sentence")
    target_sentence = st.selectbox(
        "Choose sentence to say:",
        checker.sample_sentences
    )
    
    # Step 2: audio recording
    st.markdown("#### Step 2: Record Your Voice")
    st.info(f"📝 **Say this sentence:** \"{target_sentence}\"")
    
    if AUDIO_MODE == "sounddevice":
        if st.button("🎤 Start Recording", type="primary"):
            # audio recording
            audio_data, temp_file = checker.record_audio_streamlit(duration=6)
            
            if audio_data is not None:
                # Save to temporary file
                temp_audio_file = checker.save_temp_audio(audio_data)
                
                try:
                    process_voice_analysis(checker, target_sentence, audio_data, temp_audio_file)
                finally:
                    checker.cleanup_temp_file(temp_audio_file)
    else:
        # Cloud mode - use st.audio_input
        audio_data, temp_audio_file = checker.record_audio_streamlit()
        
        if audio_data is not None:
            try:
                # For cloud mode, we already have the temp file path
                if temp_audio_file is None:
                    # Fallback: create temp file from audio data
                    temp_audio_file = checker.save_temp_audio(audio_data)
                
                # Process voice analysis
                process_voice_analysis(checker, target_sentence, audio_data, temp_audio_file)
            finally:
                # Cleanup temp file
                if temp_audio_file:
                    checker.cleanup_temp_file(temp_audio_file)

def process_voice_analysis(checker, target_sentence, audio_data, temp_audio_file):
    """Process voice analysis - shared function"""
    # Step 3: speech recognition
    st.markdown("#### Step 3: Speech Recognition")
    
    # Use file-based method only (compatible with version 17)
    recognized_text = checker.speech_to_text(temp_audio_file)
    
    # Step 4: Calculate pronunciation score
    pronunciation_score = checker.calculate_pronunciation_similarity(target_sentence, recognized_text)
    
    # Step 5: voice feature extraction
    st.markdown("#### Step 4: Voice Feature Extraction")
    features = checker.extract_voice_features(audio_data)
    
    # Step 6: Emotion analysis
    st.markdown("#### Step 5: Emotion Analysis")
    emotion_scores = checker.analyze_emotion_patterns(features)
    voice_result = checker.determine_work_readiness(emotion_scores, pronunciation_score)  # Pass pronunciation score
    
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
        
        # Show breakdown if pronunciation penalty applied
        if voice_result.get('pronunciation_penalty', 0) > 0:
            st.write(f"Base Score: {voice_result['base_score']:.1f}")
            st.write(f"Pronunciation Penalty: -{voice_result['pronunciation_penalty']}")
        
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

def show_real_cognitive_tests():
    """cognitive tests page dengan peringatan waktu untuk semua tes"""
    st.markdown("### 🧠 Cognitive Assessment")
    
    with st.expander("📋 PENTING - Bagaimana Tes Kognitif Dinilai", expanded=True):
        st.markdown("""
        ### Sistem Penilaian Berbasis Waktu
        
        Semua tes kognitif mengukur **kecepatan** dan **akurasi** secara bersamaan:
        
        | Tes | Mulai Dari | Penalti Waktu (Benar) | Penalti Waktu (Salah) | Target Waktu Ideal |
        | --- | --- | --- | --- | --- |
        | Math | 100 poin | -10 poin/detik | -5 poin/detik | < 3 detik |
        | Sequence | 100 poin | -8 poin/detik | -4 poin/detik | < 5 detik |
        | Attention | 100 poin | -5 poin/detik | -3 poin/detik | < 8 detik |
        | Memory | 100 poin | -3 poin/detik | -2 poin/detik | < 10 detik |
        | Reaction | Threshold | Berdasarkan waktu reaksi | N/A | < 0.5 detik |
        
        **Contoh:** Pada tes matematika, jawaban benar dalam 3 detik mendapat skor 70/100, sedangkan jawaban benar dalam 7 detik hanya mendapat 30/100.
        
        **Visual timer** akan muncul saat tes dimulai untuk menunjukkan waktu yang berjalan. Perhatikan timer ini!
        """)
        
        st.info("""
        **⚠️ PERHATIAN:** Timer mulai berjalan segera setelah tes dimuat. Bersiaplah sebelum memulai!
        """)
    
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
    
    # Display test-specific time info
    if selected_test == 'attention':
        st.markdown("""
        <div class="timer-info">
            ⏱️ <strong>Info Waktu:</strong> -5 poin per detik (benar), -3 poin per detik (salah)
            <br>Target waktu ideal: <8 detik
        </div>
        """, unsafe_allow_html=True)
    elif selected_test == 'memory':
        st.markdown("""
        <div class="timer-info">
            ⏱️ <strong>Info Waktu:</strong> -3 poin per detik (benar), -2 poin per detik (salah)
            <br>Target waktu ideal: <10 detik
        </div>
        """, unsafe_allow_html=True)
    elif selected_test == 'reaction':
        st.markdown("""
        <div class="timer-info">
            ⏱️ <strong>Info Waktu:</strong> Skor berdasarkan waktu reaksi
            <br>Target waktu ideal: <0.5 detik (Excellent), <1.0 detik (Good)
        </div>
        """, unsafe_allow_html=True)
    elif selected_test == 'math':
        st.markdown("""
        <div class="timer-info">
            ⏱️ <strong>Info Waktu:</strong> -10 poin per detik (benar), -5 poin per detik (salah)
            <br>Target waktu ideal: <3 detik
        </div>
        """, unsafe_allow_html=True)
    else:  # sequence
        st.markdown("""
        <div class="timer-info">
            ⏱️ <strong>Info Waktu:</strong> -8 poin per detik (benar), -4 poin per detik (salah)
            <br>Target waktu ideal: <5 detik
        </div>
        """, unsafe_allow_html=True)
    
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
        
        time_impact = 0
        if selected_test == 'math':
            time_impact = result['response_time'] * 10 if result['correct'] else result['response_time'] * 5
        elif selected_test == 'sequence':
            time_impact = result['response_time'] * 8 if result['correct'] else result['response_time'] * 4
        elif selected_test == 'attention':
            time_impact = result['response_time'] * 5 if result['correct'] else result['response_time'] * 3
        elif selected_test == 'memory':
            time_impact = result['response_time'] * 3 if result['correct'] else result['response_time'] * 2
        
        if selected_test != 'reaction':
            # Ensure time_class is always defined
            time_class = "time-medium"
            if selected_test == 'math':
                time_class = "time-fast" if result['response_time'] < 3 else "time-medium" if result['response_time'] < 6 else "time-slow"
            elif selected_test == 'sequence':
                time_class = "time-fast" if result['response_time'] < 5 else "time-medium" if result['response_time'] < 8 else "time-slow"
            elif selected_test == 'attention':
                time_class = "time-fast" if result['response_time'] < 8 else "time-medium" if result['response_time'] < 12 else "time-slow"
            elif selected_test == 'memory':
                time_class = "time-fast" if result['response_time'] < 10 else "time-medium" if result['response_time'] < 15 else "time-slow"
            # No else needed, time_class already set to "time-medium"
            
            # Ensure time_class is always defined
            time_class = "time-medium"
            if selected_test == 'math':
                time_class = "time-fast" if result['response_time'] < 3 else "time-medium" if result['response_time'] < 6 else "time-slow"
            elif selected_test == 'sequence':
                time_class = "time-fast" if result['response_time'] < 5 else "time-medium" if result['response_time'] < 8 else "time-slow"
            elif selected_test == 'attention':
                time_class = "time-fast" if result['response_time'] < 8 else "time-medium" if result['response_time'] < 12 else "time-slow"
            elif selected_test == 'memory':
                time_class = "time-fast" if result['response_time'] < 10 else "time-medium" if result['response_time'] < 15 else "time-slow"
            # No else needed, time_class already set to "time-medium"
            
            st.markdown(f"""
            <div class="time-result {time_class}">
                ⏱️ <strong>Analisis Waktu:</strong>
                <br>Waktu pengerjaan: {result['response_time']:.2f} detik
                <br>Penalti waktu: -{int(time_impact)} poin
                <br>Skor maksimal tanpa penalti waktu: {min(100, result['score'] + int(time_impact))}/100
                <br>Skor akhir dengan penalti waktu: {result['score']}/100
            </div>
            """, unsafe_allow_html=True)
            
            max_score = min(100, result['score'] + int(time_impact))
            final_score = result['score']
            
            st.markdown(f"""
            <div class="score-impact">
                <div class="max-score">Max: {max_score}</div>
                <div class="time-penalty">-{int(time_impact)}</div>
                <div class="final-score">Final: {final_score}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if selected_test == 'math':
                st.info("⏱️ Target waktu ideal untuk Math Test: < 3 detik")
            elif selected_test == 'sequence':
                st.info("⏱️ Target waktu ideal untuk Sequence Test: < 5 detik")
            elif selected_test == 'attention':
                st.info("⏱️ Target waktu ideal untuk Attention Test: < 8 detik")
            elif selected_test == 'memory':
                st.info("⏱️ Target waktu ideal untuk Memory Test: < 10 detik")
        
        if 'individual_test_results' not in st.session_state:
            st.session_state.individual_test_results = []
        
        st.session_state.individual_test_results.append(result)

def run_cognitive_battery_streamlit(self, num_tests=3):
    """Run cognitive battery dengan Streamlit UI dan peringatan waktu untuk semua tes"""
    st.subheader("🧠 COGNITIVE ASSESSMENT BATTERY")
    
    st.markdown("""
    <div class="timer-warning">
        ⏱️ PENTING: Semua tes kognitif menggunakan WAKTU dalam penilaian!
        <br>Kecepatan dan akurasi sama-sama penting. Semakin cepat menjawab dengan benar, semakin tinggi skor Anda.
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("ℹ️ Pengaruh Waktu pada Skor"):
        st.markdown("""
        ### Pengaruh Waktu pada Skor Tes
        
        Setiap tes kognitif memiliki penalti waktu yang berbeda:
        
        | Tes | Penalti Waktu (Jawaban Benar) | Penalti Waktu (Jawaban Salah) |
        | --- | --- | --- |
        | Attention | -5 poin per detik | -3 poin per detik |
        | Memory | -3 poin per detik | -2 poin per detik |
        | Math | -10 poin per detik | -5 poin per detik |
        | Sequence | -8 poin per detik | -4 poin per detik |
        | Reaction | Berdasarkan threshold waktu | N/A |
        
        **Contoh:** Pada tes matematika, jika Anda menjawab dengan benar dalam 3 detik, Anda akan kehilangan 30 poin dari skor maksimum 100, sehingga mendapatkan skor 70.
        """)
    
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
        
        # Display time info for current test
        if test_name == 'attention':
            st.markdown("""
            <div class="timer-info">
                ⏱️ <strong>Info Waktu:</strong> -5 poin per detik (benar), -3 poin per detik (salah)
                <br>Target waktu ideal: <8 detik
            </div>
            """, unsafe_allow_html=True)
        elif test_name == 'memory':
            st.markdown("""
            <div class="timer-info">
                ⏱️ <strong>Info Waktu:</strong> -3 poin per detik (benar), -2 poin per detik (salah)
                <br>Target waktu ideal: <10 detik
            </div>
            """, unsafe_allow_html=True)
        elif test_name == 'reaction':
            st.markdown("""
            <div class="timer-info">
                ⏱️ <strong>Info Waktu:</strong> Skor berdasarkan waktu reaksi
                <br>Target waktu ideal: <0.5 detik (Excellent), <1.0 detik (Good)
            </div>
            """, unsafe_allow_html=True)
        elif test_name == 'math':
            st.markdown("""
            <div class="timer-info">
                ⏱️ <strong>Info Waktu:</strong> -10 poin per detik (benar), -5 poin per detik (salah)
                <br>Target waktu ideal: <3 detik
            </div>
            """, unsafe_allow_html=True)
        else:  # sequence
            st.markdown("""
            <div class="timer-info">
                ⏱️ <strong>Info Waktu:</strong> -8 poin per detik (benar), -4 poin per detik (salah)
                <br>Target waktu ideal: <5 detik
            </div>
            """, unsafe_allow_html=True)
        
        result = self.cognitive_tests[test_name]()
        
        if result:
            # Show time analysis before continuing
            if test_name != 'reaction':
                time_impact = 0
                time_class = "time-medium"  # default
                if test_name == 'math':
                    time_impact = result['response_time'] * 10 if result['correct'] else result['response_time'] * 5
                    time_class = "time-fast" if result['response_time'] < 3 else "time-medium" if result['response_time'] < 6 else "time-slow"
                elif test_name == 'sequence':
                    time_impact = result['response_time'] * 8 if result['correct'] else result['response_time'] * 4
                    time_class = "time-fast" if result['response_time'] < 5 else "time-medium" if result['response_time'] < 8 else "time-slow"
                elif test_name == 'attention':
                    time_impact = result['response_time'] * 5 if result['correct'] else result['response_time'] * 3
                    time_class = "time-fast" if result['response_time'] < 8 else "time-medium" if result['response_time'] < 12 else "time-slow"
                elif test_name == 'memory':
                    time_impact = result['response_time'] * 3 if result['correct'] else result['response_time'] * 2
                    time_class = "time-fast" if result['response_time'] < 10 else "time-medium" if result['response_time'] < 15 else "time-slow"
                
                # Show time impact
                st.markdown(f"""
                <div class="time-result {time_class}">
                    ⏱️ <strong>Analisis Waktu:</strong> {result['response_time']:.2f} detik (-{int(time_impact)} poin)
                </div>
                """, unsafe_allow_html=True)
            
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
                
                # Calculate average response time
                avg_time = sum(r['response_time'] for r in results) / len(results)
                
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
                
                time_class = "time-fast" if avg_time < 5 else "time-medium" if avg_time < 10 else "time-slow"
                
                st.markdown(f"""
                <div class="time-result {time_class}">
                    ⏱️ <strong>Analisis Waktu Rata-Rata:</strong> {avg_time:.2f} detik
                </div>
                """, unsafe_allow_html=True)
                
                cognitive_summary = {
                    'total_tests': len(results),
                    'individual_results': results,
                    'average_score': round(avg_score, 1),
                    'average_time': round(avg_time, 2),
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
        avg_time = sum(r['response_time'] for r in results) / len(results)
        
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
            'average_time': round(avg_time, 2),
            'total_score': total_score,
            'status': status,
            'recommendation': recommendation,
            'timestamp': datetime.now().isoformat()
        }
    
    return None

def show_real_complete_assessment():
    """complete assessment workflow"""
    st.markdown("### 📊 Complete Assessment")
    
    # Initialize assessment state
    if 'complete_assessment_step' not in st.session_state:
        st.session_state.complete_assessment_step = 'start'
    
    if st.session_state.complete_assessment_step == 'start':
        st.markdown("#### 🎯 Complete Fit-to-Work Assessment")
        assessment_info = """
        This assessment includes:
        1. **Voice Analysis** (audio recording)
        2. **Cognitive Battery** (3 tests)
        3. **Final Integration** (weighted scoring)
        
        Total time: ~5-7 minutes
        """
        
        if AUDIO_MODE == "streamlit_native":
            assessment_info += "\n\n**Note:** Using web-based audio input for cloud compatibility."
        
        st.info(assessment_info)
        
        if st.button("🚀 Start Complete Assessment", type="primary"):
            st.session_state.complete_assessment_step = 'voice'
            st.rerun()
    
    elif st.session_state.complete_assessment_step == 'voice':
        st.markdown("#### 🎤 Step 1: Voice Analysis")
        
        checker = RealVoiceChecker()
        target_sentence = checker.get_random_sentence()
        
        st.info(f"📝 **Say this sentence:** \"{target_sentence}\"")
        
        if AUDIO_MODE == "sounddevice":
            if st.button("🎤 Record Voice", type="primary"):
                audio_data, temp_file = checker.record_audio_streamlit(duration=6)
                
                if audio_data is not None:
                    temp_audio_file = checker.save_temp_audio(audio_data)
                    
                    try:
                        # Process voice analysis
                        process_complete_voice_analysis(checker, target_sentence, audio_data, temp_audio_file)
                            
                    finally:
                        checker.cleanup_temp_file(temp_audio_file)
        else:
            # Cloud mode
            audio_data, temp_audio_file = checker.record_audio_streamlit()
            
            if audio_data is not None:
                try:
                    if temp_audio_file is None:
                        temp_audio_file = checker.save_temp_audio(audio_data)
                    
                    process_complete_voice_analysis(checker, target_sentence, audio_data, temp_audio_file)
                finally:
                    # Cleanup handled in process_complete_voice_analysis
                    pass
    
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
        show_final_assessment_results()

def process_complete_voice_analysis(checker, target_sentence, audio_data, temp_audio_file):
    """Process voice analysis for complete assessment"""
    if temp_audio_file is None and audio_data is not None:
        temp_audio_file = checker.save_temp_audio(audio_data)
    
    if temp_audio_file is None:
        st.error("❌ Failed to create temporary audio file")
        return
    
    try:
        recognized_text = checker.speech_to_text(temp_audio_file)
        pronunciation_score = checker.calculate_pronunciation_similarity(target_sentence, recognized_text)
        features = checker.extract_voice_features(audio_data)
        emotion_scores = checker.analyze_emotion_patterns(features)
        voice_result = checker.determine_work_readiness(emotion_scores, pronunciation_score)  # Include pronunciation
        
        # Store results
        st.session_state.complete_voice_result = {
            'target_sentence': target_sentence,
            'recognized_text': recognized_text,
            'pronunciation_score': pronunciation_score,
            'readiness_score': voice_result['readiness_score'],
            'base_score': voice_result.get('base_score', voice_result['readiness_score']),
            'pronunciation_penalty': voice_result.get('pronunciation_penalty', 0),
            'dominant_emotion': voice_result['dominant_emotion'],
            'emotion_breakdown': emotion_scores,
            'voice_features': features
        }
        
        st.success(f"✅ Voice analysis complete! Score: {voice_result['readiness_score']}/100")
        
        # Show penalty breakdown if applied
        if voice_result.get('pronunciation_penalty', 0) > 0:
            st.info(f"📢 Pronunciation penalty applied: -{voice_result['pronunciation_penalty']} points (Base: {voice_result['base_score']:.1f})")
        
        st.session_state.complete_assessment_step = 'cognitive'
        
        if st.button("Continue to Cognitive Tests"):
            st.rerun()
            
    finally:
        checker.cleanup_temp_file(temp_audio_file)

def show_final_assessment_results():
    """Show final assessment results"""
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
            'recommendation': recommendation,
            'audio_mode': AUDIO_MODE
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
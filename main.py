import sounddevice as sd
import soundfile as sf
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
import tempfile
import os
import json
import pandas as pd
from datetime import datetime
import speech_recognition as sr

class IntegratedVoiceChecker:
    """
    Integrated system: Speech Recognition + Emotion Analysis + Data Collection
    """
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        
        # Speech Recognition Setup
        self.recognizer = sr.Recognizer()
        print("🔧 Initializing speech recognition...")
        
        # Emotion Analysis Setup
        self.emotion_labels = {
            'ready': 'Siap & Fokus',
            'tired': 'Lelah/Mengantuk', 
            'stressed': 'Stress/Tegang',
            'calm': 'Tenang & Stabil',
            'uncertain': 'Ragu/Tidak Yakin'
        }
        
        # Presentation Data Collection
        self.test_results = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Sample sentences
        self.sample_sentences = [
            "Saya siap kerja",
            "Keselamatan utama", 
            "Kondisi sehat",
            "Peralatan aman",
            "Tim komunikasi baik"
        ]
        
        print("🧠 Integrated Voice Checker initialized")
        print(f"📊 Session ID: {self.session_id}")
    
    def get_varied_instructions(self, test_number):
        """Get varied instructions untuk testing yang lebih menarik"""
        instruction_sets = [
            {
                'mood': 'energetic',
                'instruction': '💪 Ucapkan dengan penuh semangat dan energi!',
                'example': 'Bayangkan Anda baru dapat kabar baik'
            },
            {
                'mood': 'calm',
                'instruction': '😌 Ucapkan dengan tenang dan santai',
                'example': 'Seperti sedang berbicara dengan teman'
            },
            {
                'mood': 'professional',
                'instruction': '👔 Ucapkan dengan nada formal dan profesional',
                'example': 'Seperti sedang presentasi di kantor'
            },
            {
                'mood': 'tired',
                'instruction': '😴 Ucapkan seolah Anda agak lelah/mengantuk',
                'example': 'Simulasikan kondisi setelah kerja lembur'
            },
            {
                'mood': 'confident',
                'instruction': '🦾 Ucapkan dengan penuh percaya diri',
                'example': 'Seperti sedang meyakinkan atasan'
            },
            {
                'mood': 'hesitant',
                'instruction': '🤔 Ucapkan dengan sedikit ragu-ragu',
                'example': 'Seperti sedang tidak yakin dengan kondisi'
            },
            {
                'mood': 'normal',
                'instruction': '🗣️ Ucapkan secara natural dan biasa',
                'example': 'Seperti berbicara sehari-hari'
            },
            {
                'mood': 'stressed',
                'instruction': '😰 Ucapkan dengan nada agak tegang/stress',
                'example': 'Simulasikan kondisi deadline mendekat'
            }
        ]
        
        # Cycle through different instruction sets
        return instruction_sets[test_number % len(instruction_sets)]
    
    def get_contextual_sentences(self):
        """Get sentences dengan context yang bervariasi"""
        sentence_contexts = [
            {
                'sentence': 'Saya siap kerja',
                'context': 'Pernyataan kesiapan dasar',
                'focus': 'Overall readiness'
            },
            {
                'sentence': 'Keselamatan utama',
                'context': 'Komitmen safety first',
                'focus': 'Safety awareness'
            },
            {
                'sentence': 'Kondisi sehat',
                'context': 'Deklarasi kesehatan fisik',
                'focus': 'Health status'
            },
            {
                'sentence': 'Peralatan aman',
                'context': 'Konfirmasi equipment check',
                'focus': 'Equipment readiness'
            },
            {
                'sentence': 'Tim komunikasi baik',
                'context': 'Kesiapan koordinasi tim',
                'focus': 'Team coordination'
            }
        ]
        
        import random
        return random.choice(sentence_contexts)
    
    def record_audio(self, duration=6):
        """Record audio for analysis"""
        print(f"🎤 Recording selama {duration} detik...")
        print("📢 Ucapkan kalimat target dengan jelas dan natural!")
        
        audio_data = sd.rec(int(duration * self.sample_rate), 
                          samplerate=self.sample_rate, 
                          channels=1, 
                          dtype='float32')
        sd.wait()
        print("✅ Recording selesai!")
        return audio_data.flatten()
    
    def save_temp_audio(self, audio_data):
        """Save audio to temporary file"""
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(temp_file.name, audio_data, self.sample_rate)
        return temp_file.name
    
    def speech_to_text(self, audio_file_path):
        """Speech recognition with ambient noise filtering"""
        try:
            print("🤖 Memproses speech recognition...")
            
            with sr.AudioFile(audio_file_path) as source:
                print("🔧 Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
                print("🔧 Reading audio...")
                audio = self.recognizer.listen(source)
                print("✅ Audio loaded successfully")
            
            print("📡 Calling Google Speech API...")
            
            # Test dengan bahasa Indonesia dan English
            languages = [('id-ID', 'Indonesian'), ('en-US', 'English')]
            
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
    
    def calculate_pronunciation_similarity(self, target_sentence, recognized_text):
        """Calculate similarity between target and recognized text"""
        if recognized_text in ["TIDAK_TERDETEKSI", "ERROR"]:
            return 0
        
        import difflib
        
        # Normalize texts
        target = target_sentence.lower().strip()
        recognized = recognized_text.lower().strip()
        
        # Calculate similarity
        similarity = difflib.SequenceMatcher(None, target, recognized).ratio()
        return round(similarity * 100, 1)
    
    def extract_voice_features(self, audio_data):
        """
        Extract audio features untuk emotion analysis
        """
        print("🔍 Extracting voice features...")
        
        features = {}
        
        # 1. Pitch Analysis (Fundamental Frequency)
        try:
            # Extract pitch using librosa
            pitches, magnitudes = librosa.piptrack(y=audio_data, sr=self.sample_rate, 
                                                  threshold=0.1, fmin=50, fmax=400)
            
            # Get pitch values (remove zeros)
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
            print(f"⚠️ Pitch analysis warning: {e}")
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
            features['pitch_range'] = 0
        
        # 2. Energy and Amplitude Analysis
        features['rms_energy'] = np.sqrt(np.mean(audio_data**2))
        features['max_amplitude'] = np.max(np.abs(audio_data))
        features['energy_variance'] = np.var(audio_data**2)
        
        # 3. Speaking Rate Analysis
        # Detect speech segments
        frame_length = int(0.025 * self.sample_rate)  # 25ms frames
        hop_length = int(0.01 * self.sample_rate)     # 10ms hop
        
        # Energy-based voice activity detection
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
        
        # 4. Spectral Features (Frequency characteristics)
        # MFCCs (Mel-Frequency Cepstral Coefficients)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfccs, axis=1)
        features['mfcc_std'] = np.std(mfccs, axis=1)
        
        # Spectral centroid (brightness of sound)
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        # Zero crossing rate (roughness)
        zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # 5. Temporal Features
        # Pause detection
        silence_threshold = features['rms_energy'] * 0.1
        silent_frames = np.sum(energy_frames < silence_threshold)
        features['silence_ratio'] = silent_frames / total_frames if total_frames > 0 else 0
        
        print(f"✅ Extracted {len(features)} voice features")
        return features
    
    def analyze_emotion_patterns(self, features):
        """
        Analyze extracted features to determine emotional state
        """
        print("🧠 Analyzing emotion patterns...")
        
        emotion_scores = {}
        
        # Analyze READY/FOCUSED patterns
        ready_score = 0
        
        # Ready indicators:
        # - Stable pitch (not too high/low variance)
        # - Good energy level (not too low/high)
        # - Consistent speech rate
        # - Clear articulation (good spectral clarity)
        
        if 100 < features['pitch_mean'] < 300:  # Normal pitch range
            ready_score += 20
        
        if features['pitch_std'] < 50:  # Stable pitch
            ready_score += 15
        
        if 0.01 < features['rms_energy'] < 0.1:  # Good energy level
            ready_score += 20
        
        if features['speech_rate'] > 0.3:  # Active speech
            ready_score += 15
        
        if features['spectral_centroid_mean'] > 1000:  # Clear articulation
            ready_score += 15
        
        if features['silence_ratio'] < 0.7:  # Not too much silence
            ready_score += 15
        
        emotion_scores['ready'] = min(ready_score, 100)
        
        # Analyze TIRED/SLEEPY patterns
        tired_score = 0
        
        # Tired indicators:
        # - Lower pitch
        # - Low energy
        # - Slow speech rate
        # - More pauses/silence
        
        if features['pitch_mean'] < 150:  # Lower pitch
            tired_score += 25
        
        if features['rms_energy'] < 0.02:  # Low energy
            tired_score += 30
        
        if features['speech_rate'] < 0.4:  # Slow speech
            tired_score += 25
        
        if features['silence_ratio'] > 0.5:  # Many pauses
            tired_score += 20
        
        emotion_scores['tired'] = min(tired_score, 100)
        
        # Analyze STRESSED/TENSE patterns
        stressed_score = 0
        
        # Stress indicators:
        # - Higher pitch
        # - High pitch variance
        # - High energy variance
        # - Fast/irregular speech
        
        if features['pitch_mean'] > 250:  # Higher pitch
            stressed_score += 25
        
        if features['pitch_std'] > 60:  # Unstable pitch
            stressed_score += 25
        
        if features['energy_variance'] > 0.01:  # Energy fluctuation
            stressed_score += 25
        
        if features['zcr_mean'] > 0.1:  # Voice roughness
            stressed_score += 25
        
        emotion_scores['stressed'] = min(stressed_score, 100)
        
        # Analyze CALM/STABLE patterns
        calm_score = 0
        
        # Calm indicators:
        # - Stable moderate pitch
        # - Consistent energy
        # - Steady speech rate
        # - Smooth spectral characteristics
        
        if 120 < features['pitch_mean'] < 220:  # Moderate pitch
            calm_score += 25
        
        if features['pitch_std'] < 30:  # Very stable pitch
            calm_score += 25
        
        if features['energy_variance'] < 0.005:  # Consistent energy
            calm_score += 25
        
        if 0.4 < features['speech_rate'] < 0.7:  # Steady speech
            calm_score += 25
        
        emotion_scores['calm'] = min(calm_score, 100)
        
        # Analyze UNCERTAIN patterns
        uncertain_score = 0
        
        # Uncertainty indicators:
        # - Variable pitch (hesitation)
        # - Irregular energy
        # - Many pauses
        # - Lower confidence in articulation
        
        if features['pitch_std'] > 40:  # Variable pitch
            uncertain_score += 30
        
        if features['silence_ratio'] > 0.6:  # Many hesitation pauses
            uncertain_score += 30
        
        if features['speech_rate'] < 0.5:  # Hesitant speech
            uncertain_score += 20
        
        if features['spectral_centroid_mean'] < 1500:  # Less clear articulation
            uncertain_score += 20
        
        emotion_scores['uncertain'] = min(uncertain_score, 100)
        
        return emotion_scores
    
    def determine_work_readiness(self, emotion_scores):
        """
        Determine overall work readiness based on emotion analysis
        """
        print("⚖️ Determining work readiness...")
        
        # Calculate readiness score
        positive_emotions = emotion_scores['ready'] + emotion_scores['calm']
        negative_emotions = emotion_scores['tired'] + emotion_scores['stressed'] + emotion_scores['uncertain']
        
        # Weighted calculation
        readiness_score = (positive_emotions * 0.7) - (negative_emotions * 0.3)
        readiness_score = max(0, min(100, readiness_score))
        
        # Determine status
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
        
        # Find dominant emotion
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
    
    def record_audio_for_emotion(self, duration=5):
        """Record audio khusus untuk emotion analysis"""
        print("🎤 Recording untuk analisis emosi...")
        print("💡 Tips: Bicara secara natural dan ekspresif")
        print("📢 Ceritakan bagaimana perasaan Anda hari ini!")
        
        audio_data = sd.rec(int(duration * self.sample_rate), 
                          samplerate=self.sample_rate, 
                          channels=1, 
                          dtype='float32')
        sd.wait()
        print("✅ Recording selesai!")
        return audio_data.flatten()
    
    def convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    def collect_test_data(self, target_sentence, recognized_text, pronunciation_score, 
                         emotion_result, audio_features, user_notes=""):
        """Collect test data for presentation analysis"""
        # Convert all numpy types to native Python types
        safe_emotion_scores = self.convert_numpy_types(emotion_result['emotion_breakdown'])
        safe_audio_features = self.convert_numpy_types({
            'pitch_mean': round(float(audio_features['pitch_mean']), 2),
            'pitch_std': round(float(audio_features['pitch_std']), 2),
            'rms_energy': round(float(audio_features['rms_energy']), 6),
            'speech_rate': round(float(audio_features['speech_rate']), 3),
            'spectral_centroid_mean': round(float(audio_features['spectral_centroid_mean']), 2)
        })
        
        test_sample = {
            'session_id': self.session_id,
            'test_number': len(self.test_results) + 1,
            'timestamp': datetime.now().isoformat(),
            'target_sentence': target_sentence,
            'recognized_text': recognized_text,
            'pronunciation_score': float(pronunciation_score),
            'readiness_score': float(emotion_result['readiness_score']),
            'readiness_status': emotion_result['status'],
            'dominant_emotion': emotion_result['dominant_emotion'],
            'emotion_scores': safe_emotion_scores,
            'audio_features': safe_audio_features,
            'user_notes': user_notes
        }
        
        self.test_results.append(test_sample)
        print(f"📊 Test data collected (Sample #{len(self.test_results)})")
        return test_sample
    
    def save_session_data(self):
        """Save collected session data untuk presentasi"""
        if not self.test_results:
            print("⚠️ No test data to save")
            return None
        
        try:
            # Create comprehensive session report dengan converted data
            session_data = {
                'session_info': {
                    'session_id': self.session_id,
                    'timestamp': datetime.now().isoformat(),
                    'total_tests': len(self.test_results),
                    'system_version': 'Voice Readiness Checker v1.0'
                },
                'session_metrics': self.calculate_session_metrics(),
                'test_results': self.convert_numpy_types(self.test_results)
            }
            
            # Save to JSON with safe serialization
            filename_json = f"session_data_{self.session_id}.json"
            with open(filename_json, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            
            # Save to CSV for easy analysis
            df = pd.DataFrame(self.test_results)
            filename_csv = f"session_data_{self.session_id}.csv"
            df.to_csv(filename_csv, index=False)
            
            print(f"💾 Session data saved successfully:")
            print(f"   📄 {filename_json}")
            print(f"   📊 {filename_csv}")
            
            return session_data
            
        except Exception as e:
            print(f"❌ Error saving session data: {e}")
            
            # Fallback: Save CSV only (more robust)
            try:
                df = pd.DataFrame(self.test_results)
                filename_csv = f"session_data_{self.session_id}_backup.csv"
                df.to_csv(filename_csv, index=False)
                print(f"💾 Backup saved to: {filename_csv}")
                return None
            except Exception as e2:
                print(f"❌ Backup save also failed: {e2}")
                return None
    
    def calculate_session_metrics(self):
        """Calculate metrics untuk presentasi"""
        if not self.test_results:
            return {}
        
        # Pronunciation metrics
        pronunciation_scores = [float(r['pronunciation_score']) for r in self.test_results]
        
        # Readiness metrics  
        readiness_scores = [float(r['readiness_score']) for r in self.test_results]
        
        # Recognition success rate
        successful_recognitions = sum(1 for r in self.test_results 
                                    if r['recognized_text'] not in ['TIDAK_TERDETEKSI', 'ERROR'])
        
        # Status distribution
        status_counts = {}
        for r in self.test_results:
            status = r['readiness_status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Emotion distribution
        emotion_counts = {}
        for r in self.test_results:
            emotion = r['dominant_emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        metrics = {
            'pronunciation_analysis': {
                'mean_score': round(float(np.mean(pronunciation_scores)), 1),
                'std_score': round(float(np.std(pronunciation_scores)), 1),
                'min_score': round(float(np.min(pronunciation_scores)), 1),
                'max_score': round(float(np.max(pronunciation_scores)), 1)
            },
            'readiness_analysis': {
                'mean_score': round(float(np.mean(readiness_scores)), 1),
                'std_score': round(float(np.std(readiness_scores)), 1),
                'status_distribution': status_counts
            },
            'system_performance': {
                'total_tests': len(self.test_results),
                'successful_recognitions': successful_recognitions,
                'recognition_rate': round(float(successful_recognitions / len(self.test_results) * 100), 1)
            },
            'emotion_distribution': emotion_counts
        }
        
        return metrics

# Integrated test function
def test_integrated_voice_checker():
    """Complete integrated test: Speech + Emotion + Data Collection"""
    print("=== VOICE READINESS CHECKER - INTEGRATED SYSTEM ===")
    print("Speech Recognition + Emotion Analysis + Data Collection\n")
    
    checker = IntegratedVoiceChecker()
    
    print("🎯 SISTEM TERINTEGRASI:")
    print("1. Speech Recognition & Pronunciation Assessment")
    print("2. Voice Emotion/Mood Analysis") 
    print("3. Automatic Data Collection untuk Presentasi")
    print()
    
    try:
        while True:
            # Get varied instruction dan target sentence
            test_number = len(checker.test_results) + 1
            instruction_set = checker.get_varied_instructions(test_number - 1)
            sentence_context = checker.get_contextual_sentences()
            
            target_sentence = sentence_context['sentence']
            
            print(f"📝 TEST #{test_number} - {instruction_set['mood'].upper()} MODE")
            print("=" * 60)
            print(f"🎯 Target: \"{target_sentence}\"")
            print(f"📖 Context: {sentence_context['context']}")
            print(f"🎭 Mood: {instruction_set['instruction']}")
            print(f"💡 Tip: {instruction_set['example']}")
            print()
            
            input("Tekan Enter untuk mulai recording...")
            
            # Record audio
            audio_data = checker.record_audio(duration=6)
            
            # Save to temporary file
            temp_audio_file = checker.save_temp_audio(audio_data)
            
            # Step 1: Speech Recognition
            recognized_text = checker.speech_to_text(temp_audio_file)
            
            # Step 2: Calculate pronunciation score
            pronunciation_score = checker.calculate_pronunciation_similarity(target_sentence, recognized_text)
            
            # Step 3: Extract voice features
            features = checker.extract_voice_features(audio_data)
            
            # Step 4: Analyze emotions
            emotion_scores = checker.analyze_emotion_patterns(features)
            emotion_result = checker.determine_work_readiness(emotion_scores)
            
            # Step 5: Collect data dengan mood context
            test_sample = checker.collect_test_data(
                target_sentence=target_sentence,
                recognized_text=recognized_text,
                pronunciation_score=pronunciation_score,
                emotion_result=emotion_result,
                audio_features=features,
                user_notes=f"Test #{test_number} - {instruction_set['mood']} mode"
            )
            
            # Display results
            print(f"\n📊 HASIL ANALISIS LENGKAP:")
            print("=" * 60)
            
            print(f"\n🗣️ SPEECH RECOGNITION:")
            print(f"   Target      : \"{target_sentence}\"")
            print(f"   Recognized  : \"{recognized_text}\"")
            print(f"   Pronunciation: {pronunciation_score}%")
            
            print(f"\n🧠 EMOTION ANALYSIS:")
            print(f"   {emotion_result['color']} Status: {emotion_result['status']}")
            print(f"   🎯 Readiness Score: {emotion_result['readiness_score']}/100")
            print(f"   🎭 Dominant Emotion: {checker.emotion_labels[emotion_result['dominant_emotion']]}")
            
            print(f"\n💡 REKOMENDASI:")
            print(f"   {emotion_result['recommendation']}")
            
            print(f"\n📈 EMOTION BREAKDOWN:")
            for emotion, score in emotion_scores.items():
                label = checker.emotion_labels[emotion]
                bar = "█" * (score // 10) + "░" * (10 - score // 10)
                print(f"   {label:15} [{bar}] {score:3.0f}%")
            
            print(f"\n🔍 VOICE FEATURES:")
            print(f"   Pitch      : {features['pitch_mean']:.1f} Hz")
            print(f"   Energy     : {features['rms_energy']:.4f}")
            print(f"   Speech Rate: {features['speech_rate']:.2f}")
            print(f"   Clarity    : {features['spectral_centroid_mean']:.0f} Hz")
            
            # Cleanup temp file
            try:
                os.unlink(temp_audio_file)
            except:
                pass
            
            # Ask for next test
            print(f"\n📊 Current session: {len(checker.test_results)} tests completed")
            
            continue_test = input("\nLanjut test lagi? (y/n/save): ").lower().strip()
            
            if continue_test == 'save':
                # Save session data
                session_data = checker.save_session_data()
                print("\n✅ Session data saved successfully!")
                break
            elif continue_test == 'n':
                # Save session data before exit
                if checker.test_results:
                    checker.save_session_data()
                    print("\n💾 Session data auto-saved before exit")
                break
            
    except KeyboardInterrupt:
        print("\n\n⏹️ Test interrupted by user")
        if checker.test_results:
            checker.save_session_data()
            print("💾 Session data saved before exit")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        if checker.test_results:
            checker.save_session_data()
            print("💾 Session data saved despite error")

if __name__ == "__main__":
    test_integrated_voice_checker()
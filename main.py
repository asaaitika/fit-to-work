import random
import time
import json
import pandas as pd
from datetime import datetime
import numpy as np

class CognitiveAssessment:
    """
    Cognitive assessment untuk mengukur kesiapan mental bekerja
    """
    def __init__(self):
        self.cognitive_tests = {
            'attention': self.attention_test,
            'memory': self.memory_test, 
            'reaction': self.reaction_time_test,
            'math': self.simple_math_test,
            'sequence': self.sequence_test
        }
        
        print("🧠 Cognitive Assessment Module initialized")
    
    def attention_test(self):
        """Test fokus dan perhatian"""
        print("\n🎯 ATTENTION TEST")
        print("Hitung berapa huruf 'A' dalam teks berikut:")
        
        # Generate random text dengan huruf A
        texts = [
            "BANANA ADALAH BUAH YANG MANIS DAN BERGIZI TINGGI",
            "APLIKASI ANDROID SANGAT MEMBANTU AKTIVITAS HARIAN",
            "AREA PARKIR YANG AMAN MEMBERIKAN RASA TENANG",
            "ALARM KEBAKARAN AKTIF SETIAP SAAT DI AREA KERJA"
        ]
        
        selected_text = random.choice(texts)
        correct_count = selected_text.count('A')
        
        print(f"📖 Text: {selected_text}")
        
        start_time = time.time()
        try:
            user_answer = int(input("Berapa huruf 'A'? "))
            response_time = time.time() - start_time
            
            if user_answer == correct_count:
                score = max(0, 100 - int(response_time * 5))  # Penalty untuk lambat
                result = "✅ BENAR"
            else:
                score = max(0, 50 - int(response_time * 3))
                result = f"❌ SALAH (jawaban: {correct_count})"
            
            return {
                'test_type': 'attention',
                'score': score,
                'response_time': round(response_time, 2),
                'correct': user_answer == correct_count,
                'result': result
            }
            
        except ValueError:
            return {
                'test_type': 'attention',
                'score': 0,
                'response_time': 999,
                'correct': False,
                'result': "❌ Invalid input"
            }
    
    def memory_test(self):
        """Test memori jangka pendek"""
        print("\n🧠 MEMORY TEST")
        print("Hafalkan sequence angka berikut (akan hilang dalam 5 detik):")
        
        # Generate random sequence
        sequence_length = random.randint(4, 6)
        sequence = [random.randint(1, 9) for _ in range(sequence_length)]
        sequence_str = " - ".join(map(str, sequence))
        
        print(f"📋 Sequence: {sequence_str}")
        time.sleep(5)
        
        # Clear screen effect
        print("\n" * 10)
        print("⏰ Waktu habis! Sekarang ketik sequence yang tadi:")
        
        start_time = time.time()
        try:
            user_input = input("Sequence (pisahkan dengan spasi): ")
            response_time = time.time() - start_time
            
            user_sequence = [int(x.strip()) for x in user_input.split()]
            
            if user_sequence == sequence:
                score = max(0, 100 - int(response_time * 3))
                result = "✅ BENAR"
            else:
                # Partial credit untuk sebagian benar
                correct_positions = sum(1 for i, (a, b) in enumerate(zip(user_sequence, sequence)) if a == b)
                score = max(0, int(correct_positions / len(sequence) * 70) - int(response_time * 2))
                result = f"❌ SALAH (benar: {sequence})"
            
            return {
                'test_type': 'memory',
                'score': score,
                'response_time': round(response_time, 2),
                'correct': user_sequence == sequence,
                'result': result
            }
            
        except (ValueError, IndexError):
            return {
                'test_type': 'memory',
                'score': 0,
                'response_time': 999,
                'correct': False,
                'result': "❌ Invalid input"
            }
    
    def reaction_time_test(self):
        """Test waktu reaksi"""
        print("\n⚡ REACTION TIME TEST")
        print("Tekan ENTER secepat mungkin saat melihat '🚨 GO!'")
        print("Tunggu instruksi...")
        
        # Random delay
        delay = random.uniform(2, 5)
        time.sleep(delay)
        
        print("🚨 GO!")
        start_time = time.time()
        input()
        reaction_time = time.time() - start_time
        
        # Scoring berdasarkan reaction time
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
        
        return {
            'test_type': 'reaction',
            'score': score,
            'response_time': round(reaction_time, 3),
            'correct': True,
            'result': f"{result} ({reaction_time:.3f}s)"
        }
    
    def simple_math_test(self):
        """Test kalkulasi sederhana"""
        print("\n🔢 MATH TEST")
        print("Hitung dengan cepat:")
        
        # Generate random math problem
        problems = [
            (lambda: (random.randint(10, 50), random.randint(5, 20)), lambda a, b: a + b, "+"),
            (lambda: (random.randint(30, 80), random.randint(5, 25)), lambda a, b: a - b, "-"),
            (lambda: (random.randint(2, 12), random.randint(2, 9)), lambda a, b: a * b, "×")
        ]
        
        generator, operation, symbol = random.choice(problems)
        a, b = generator()
        correct_answer = operation(a, b)
        
        print(f"📊 {a} {symbol} {b} = ?")
        
        start_time = time.time()
        try:
            user_answer = int(input("Jawaban: "))
            response_time = time.time() - start_time
            
            if user_answer == correct_answer:
                score = max(0, 100 - int(response_time * 10))
                result = "✅ BENAR"
            else:
                score = max(0, 30 - int(response_time * 5))
                result = f"❌ SALAH (jawaban: {correct_answer})"
            
            return {
                'test_type': 'math',
                'score': score,
                'response_time': round(response_time, 2),
                'correct': user_answer == correct_answer,
                'result': result
            }
            
        except ValueError:
            return {
                'test_type': 'math',
                'score': 0,
                'response_time': 999,
                'correct': False,
                'result': "❌ Invalid input"
            }
    
    def sequence_test(self):
        """Test pola dan sequence"""
        print("\n🔄 SEQUENCE TEST")
        print("Lanjutkan pola berikut:")
        
        patterns = [
            ([2, 4, 6, 8], 10, "Bilangan genap"),
            ([1, 3, 5, 7], 9, "Bilangan ganjil"),
            ([5, 10, 15, 20], 25, "Kelipatan 5"),
            ([1, 4, 9, 16], 25, "Kuadrat"),
            ([2, 6, 18, 54], 162, "×3"),
        ]
        
        sequence, answer, description = random.choice(patterns)
        
        print(f"📋 Pola: {' - '.join(map(str, sequence))} - ?")
        
        start_time = time.time()
        try:
            user_answer = int(input("Angka selanjutnya: "))
            response_time = time.time() - start_time
            
            if user_answer == answer:
                score = max(0, 100 - int(response_time * 8))
                result = f"✅ BENAR ({description})"
            else:
                score = max(0, 40 - int(response_time * 4))
                result = f"❌ SALAH (jawaban: {answer} - {description})"
            
            return {
                'test_type': 'sequence',
                'score': score,
                'response_time': round(response_time, 2),
                'correct': user_answer == answer,
                'result': result
            }
            
        except ValueError:
            return {
                'test_type': 'sequence',
                'score': 0,
                'response_time': 999,
                'correct': False,
                'result': "❌ Invalid input"
            }
    
    def run_cognitive_battery(self, num_tests=3):
        """Jalankan battery cognitive tests"""
        print("🧠 COGNITIVE ASSESSMENT BATTERY")
        print("=" * 50)
        print(f"Akan menjalankan {num_tests} tes kognitif untuk mengukur kesiapan mental")
        print()
        
        # Select random tests
        available_tests = list(self.cognitive_tests.keys())
        selected_tests = random.sample(available_tests, min(num_tests, len(available_tests)))
        
        results = []
        
        for i, test_name in enumerate(selected_tests, 1):
            print(f"🔄 TEST {i}/{len(selected_tests)}")
            result = self.cognitive_tests[test_name]()
            results.append(result)
            print(f"📊 Score: {result['score']}/100 - {result['result']}")
            
            if i < len(selected_tests):
                print("\n⏸️ 3 detik istirahat...")
                time.sleep(3)
                print()
        
        # Calculate overall cognitive score
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

# Complete Fit-to-Work System Integration
class CompleteFitToWorkChecker:
    """
    Complete integrated fit-to-work assessment system
    """
    def __init__(self):
        # Initialize voice checker components directly
        import sounddevice as sd
        import soundfile as sf
        import speech_recognition as sr
        import tempfile
        import os
        
        self.sample_rate = 16000
        self.recognizer = sr.Recognizer()
        
        # Emotion labels
        self.emotion_labels = {
            'ready': 'Siap & Fokus',
            'tired': 'Lelah/Mengantuk', 
            'stressed': 'Stress/Tegang',
            'calm': 'Tenang & Stabil',
            'uncertain': 'Ragu/Tidak Yakin'
        }
        
        # Sample sentences
        self.sample_sentences = [
            "Saya siap kerja",
            "Keselamatan utama", 
            "Kondisi sehat",
            "Peralatan aman",
            "Tim komunikasi baik"
        ]
        
        self.cognitive_assessment = CognitiveAssessment()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.complete_results = []
        
        print("🏭 COMPLETE FIT-TO-WORK CHECKER INITIALIZED")
        print(f"📊 Session ID: {self.session_id}")
    
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
        import sounddevice as sd
        
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
        import soundfile as sf
        import tempfile
        
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(temp_file.name, audio_data, self.sample_rate)
        return temp_file.name
    
    def speech_to_text(self, audio_file_path):
        """Speech recognition with ambient noise filtering"""
        import speech_recognition as sr
        
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
        """Extract audio features untuk emotion analysis"""
        import librosa
        import numpy as np
        
        print("🔍 Extracting voice features...")
        
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
            print(f"⚠️ Pitch analysis warning: {e}")
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
            features['pitch_range'] = 0
        
        # 2. Energy Analysis
        features['rms_energy'] = np.sqrt(np.mean(audio_data**2))
        features['max_amplitude'] = np.max(np.abs(audio_data))
        features['energy_variance'] = np.var(audio_data**2)
        
        # 3. Speaking Rate Analysis
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
        
        # 4. Spectral Features
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
            print(f"⚠️ Spectral features warning: {e}")
            features['spectral_centroid_mean'] = 1500
            features['zcr_mean'] = 0.05
        
        # 5. Temporal Features
        silence_threshold = features['rms_energy'] * 0.1
        silent_frames = np.sum(energy_frames < silence_threshold)
        features['silence_ratio'] = silent_frames / total_frames if total_frames > 0 else 0
        
        print(f"✅ Extracted {len(features)} voice features")
        return features
    
    def analyze_emotion_patterns(self, features):
        """Analyze extracted features to determine emotional state"""
        print("🧠 Analyzing emotion patterns...")
        
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
        """Determine overall work readiness based on emotion analysis"""
        print("⚖️ Determining work readiness...")
        
        positive_emotions = emotion_scores['ready'] + emotion_scores['calm']
        negative_emotions = emotion_scores['tired'] + emotion_scores['stressed'] + emotion_scores['uncertain']
        
        readiness_score = (positive_emotions * 0.7) - (negative_emotions * 0.3)
        readiness_score = max(0, min(100, readiness_score))
        
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
    
    def calculate_final_readiness_score(self, voice_result, cognitive_result):
        """Calculate final fit-to-work score"""
        # Weighted combination
        voice_weight = 0.6  # 60% voice analysis
        cognitive_weight = 0.4  # 40% cognitive assessment
        
        final_score = (
            voice_result['readiness_score'] * voice_weight +
            cognitive_result['average_score'] * cognitive_weight
        )
        
        # Determine final status
        if final_score >= 75:
            status = "🟢 FIT TO WORK"
            color = "green"
            recommendation = "Pekerja siap dan aman untuk bekerja"
        elif final_score >= 60:
            status = "🟡 CONDITIONAL FIT"
            color = "yellow"
            recommendation = "Bisa bekerja dengan pengawasan atau tugas ringan"
        else:
            status = "🔴 NOT FIT TO WORK"
            color = "red"
            recommendation = "Tidak disarankan bekerja, perlu istirahat atau konsultasi"
        
        return {
            'final_score': round(final_score, 1),
            'status': status,
            'color': color,
            'recommendation': recommendation,
            'voice_contribution': round(voice_result['readiness_score'] * voice_weight, 1),
            'cognitive_contribution': round(cognitive_result['average_score'] * cognitive_weight, 1)
        }
    
    def run_complete_assessment(self):
        """Jalankan complete fit-to-work assessment"""
        print("🏭 COMPLETE FIT-TO-WORK ASSESSMENT")
        print("=" * 60)
        print("Sistem akan mengevaluasi kesiapan kerja melalui:")
        print("1. 🗣️ Voice & Speech Analysis")
        print("2. 🧠 Cognitive Assessment")
        print("3. 📊 Final Integration & Recommendation")
        print()
        
        try:
            # Step 1: Voice Analysis (simplified single test)
            print("🎤 TAHAP 1: VOICE & SPEECH ANALYSIS")
            print("-" * 40)
            
            sentence_context = self.get_contextual_sentences()
            target_sentence = sentence_context['sentence']
            
            print(f"📝 Ucapkan kalimat: \"{target_sentence}\"")
            print("💡 Bicara dengan natural dan jelas")
            input("Tekan Enter untuk mulai recording...")
            
            # Voice analysis process
            audio_data = self.record_audio(duration=6)
            temp_audio_file = self.save_temp_audio(audio_data)
            recognized_text = self.speech_to_text(temp_audio_file)
            pronunciation_score = self.calculate_pronunciation_similarity(target_sentence, recognized_text)
            features = self.extract_voice_features(audio_data)
            emotion_scores = self.analyze_emotion_patterns(features)
            voice_result = self.determine_work_readiness(emotion_scores)
            
            print(f"✅ Voice Analysis Complete - Score: {voice_result['readiness_score']}/100")
            
            # Cleanup
            import os
            try:
                os.unlink(temp_audio_file)
            except:
                pass
            
            print()
            
            # Step 2: Cognitive Assessment
            print("🧠 TAHAP 2: COGNITIVE ASSESSMENT")
            print("-" * 40)
            cognitive_result = self.cognitive_assessment.run_cognitive_battery(num_tests=3)
            
            print(f"✅ Cognitive Assessment Complete - Score: {cognitive_result['average_score']}/100")
            print()
            
            # Step 3: Final Integration
            print("📊 TAHAP 3: FINAL INTEGRATION")
            print("-" * 40)
            final_result = self.calculate_final_readiness_score(voice_result, cognitive_result)
            
            # Compile complete results
            complete_assessment = {
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'voice_analysis': {
                    'target_sentence': target_sentence,
                    'recognized_text': recognized_text,
                    'pronunciation_score': pronunciation_score,
                    'readiness_score': voice_result['readiness_score'],
                    'dominant_emotion': voice_result['dominant_emotion'],
                    'emotion_breakdown': emotion_scores
                },
                'cognitive_analysis': cognitive_result,
                'final_assessment': final_result
            }
            
            # Display final results
            self.display_final_results(complete_assessment)
            
            # Save results
            self.save_complete_assessment(complete_assessment)
            
            return complete_assessment
            
        except Exception as e:
            print(f"❌ Error in complete assessment: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def display_final_results(self, assessment):
        """Display comprehensive final results"""
        print("\n" + "=" * 60)
        print("🏭 HASIL AKHIR FIT-TO-WORK ASSESSMENT")
        print("=" * 60)
        
        voice = assessment['voice_analysis']
        cognitive = assessment['cognitive_analysis']
        final = assessment['final_assessment']
        
        print(f"\n📊 SKOR KOMPONEN:")
        print(f"   🗣️ Voice & Speech  : {voice['readiness_score']}/100")
        print(f"   🧠 Cognitive       : {cognitive['average_score']}/100")
        
        print(f"\n🎯 SKOR AKHIR: {final['final_score']}/100")
        print(f"📋 STATUS: {final['status']}")
        print(f"💡 REKOMENDASI: {final['recommendation']}")
        
        print(f"\n📈 KONTRIBUSI SKOR:")
        print(f"   Voice contribution    : {final['voice_contribution']}/100 (60%)")
        print(f"   Cognitive contribution: {final['cognitive_contribution']}/100 (40%)")
        
        print(f"\n🗣️ DETAIL VOICE ANALYSIS:")
        print(f"   Target    : \"{voice['target_sentence']}\"")
        print(f"   Recognized: \"{voice['recognized_text']}\"")
        print(f"   Pronunciation: {voice['pronunciation_score']}%")
        print(f"   Dominant Emotion: {voice['dominant_emotion']}")
        
        print(f"\n🧠 DETAIL COGNITIVE ANALYSIS:")
        for i, test in enumerate(cognitive['individual_results'], 1):
            print(f"   Test {i} ({test['test_type']}): {test['score']}/100 - {test['result']}")
    
    def save_complete_assessment(self, assessment):
        """Save complete assessment results"""
        try:
            filename = f"complete_assessment_{self.session_id}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(assessment, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"\n💾 Complete assessment saved: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Error saving assessment: {e}")
            return None

# Main test function
def test_complete_fit_to_work_system():
    """Test complete integrated fit-to-work system"""
    print("🏭 COMPLETE FIT-TO-WORK SYSTEM TEST")
    print("=" * 60)
    print("Sistem lengkap untuk assessment kesiapan kerja")
    print()
    
    checker = CompleteFitToWorkChecker()
    
    print("⚠️ DISCLAIMER:")
    print("Ini adalah prototype untuk tujuan edukasi.")
    print("Untuk penggunaan real workplace, perlu validasi medical professional.")
    print()
    
    input("Tekan Enter untuk memulai complete assessment...")
    
    result = checker.run_complete_assessment()
    
    if result:
        print("\n✅ COMPLETE ASSESSMENT FINISHED!")
        print("📊 Semua data tersimpan untuk analisis lebih lanjut")
    else:
        print("\n❌ Assessment tidak berhasil diselesaikan")

if __name__ == "__main__":
    test_complete_fit_to_work_system()
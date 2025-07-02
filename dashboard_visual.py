# Visualization Dashboard & Presentation Materials Generator
# Install: pip install matplotlib seaborn plotly streamlit

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob
import os

class VoiceReadinessVisualizer:
    """
    Create comprehensive visualizations untuk presentasi capstone
    """
    def __init__(self):
        self.presentation_data = {}
        self.session_data = []
        self.complete_assessments = []
        
        print("📊 Voice Readiness Visualizer initialized")
        self.load_all_data()
    
    def load_all_data(self):
        """Load semua data dari files yang sudah disimpan"""
        print("📂 Loading data files...")

        # Ensure directories exist
        os.makedirs("data/sessions", exist_ok=True)
        os.makedirs("data/assessments", exist_ok=True)
        
        # Load session data
        session_files = glob.glob("data/sessions/session_data_*.json")
        for file in session_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.session_data.append(data)
                print(f"✅ Loaded: {file}")
            except Exception as e:
                print(f"⚠️ Could not load {file}: {e}")
        
        # Load complete assessments
        complete_files = glob.glob("data/assessments/complete_assessment_*.json")
        for file in complete_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.complete_assessments.append(data)
                print(f"✅ Loaded: {file}")
            except Exception as e:
                print(f"⚠️ Could not load {file}: {e}")
        
        print(f"📊 Data loaded: {len(self.session_data)} sessions, {len(self.complete_assessments)} complete assessments")
    
    def create_comprehensive_visualizations(self):
        """Create all visualizations untuk presentasi"""
        print("🎨 Creating comprehensive visualizations...")
        
        # Set style untuk professional presentation
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create figure dengan multiple subplots
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Project Overview
        self.plot_project_overview(fig)
        
        # 2. Model Performance Metrics
        self.plot_model_performance(fig)
        
        # 3. Voice Analysis Results
        self.plot_voice_analysis(fig)
        
        # 4. Emotion Distribution
        self.plot_emotion_distribution(fig)
        
        # 5. Cognitive Assessment Results
        self.plot_cognitive_assessment(fig)
        
        # 6. Complete System Integration
        self.plot_system_integration(fig)
        
        # 7. User Journey & Workflow
        self.plot_user_workflow(fig)
        
        # 8. Future Improvements
        self.plot_future_roadmap(fig)
        
        plt.tight_layout()
        plt.savefig('assets/images/voice_readiness_presentation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Comprehensive visualization saved: voice_readiness_presentation.png")
    
    def plot_project_overview(self, fig):
        """Plot 1: Project Overview"""
        ax1 = plt.subplot(4, 2, 1)
        
        # System components
        components = ['Speech\nRecognition', 'Emotion\nAnalysis', 'Cognitive\nAssessment', 'Final\nIntegration']
        values = [100, 100, 100, 100]  # All implemented
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
        
        bars = ax1.bar(components, values, color=colors, alpha=0.8)
        ax1.set_title('🏭 Fit-to-Work Voice Readiness Checker\nSystem Components', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Implementation Status (%)')
        ax1.set_ylim(0, 120)
        
        # Add percentage labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{value}%', ha='center', va='bottom', fontweight='bold')
        
        ax1.grid(True, alpha=0.3)
    
    def plot_model_performance(self, fig):
        """Plot 2: Model Performance Metrics"""
        ax2 = plt.subplot(4, 2, 2)
        
        # Calculate metrics dari session data
        if self.session_data:
            session_metrics = []
            for session in self.session_data:
                if 'session_metrics' in session:
                    session_metrics.append(session['session_metrics'])
            
            if session_metrics:
                # Extract metrics
                recognition_rates = [m['system_performance']['recognition_rate'] for m in session_metrics]
                pronunciation_scores = [m['pronunciation_analysis']['mean_score'] for m in session_metrics]
                
                metrics = ['Speech\nRecognition', 'Pronunciation\nAccuracy']
                values = [np.mean(recognition_rates), np.mean(pronunciation_scores)]
                
                bars = ax2.bar(metrics, values, color=['#3498db', '#2ecc71'], alpha=0.8)
                ax2.set_title('📊 Model Performance Metrics', fontsize=14, fontweight='bold')
                ax2.set_ylabel('Accuracy (%)')
                ax2.set_ylim(0, 100)
                
                # Add value labels
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                            f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax2.grid(True, alpha=0.3)
    
    def plot_voice_analysis(self, fig):
        """Plot 3: Voice Analysis Results"""
        ax3 = plt.subplot(4, 2, 3)
        
        if self.complete_assessments:
            voice_scores = []
            pronunciation_scores = []
            
            for assessment in self.complete_assessments:
                voice_scores.append(assessment['voice_analysis']['readiness_score'])
                pronunciation_scores.append(assessment['voice_analysis']['pronunciation_score'])
            
            x = range(1, len(voice_scores) + 1)
            ax3.plot(x, voice_scores, marker='o', linewidth=2, label='Voice Readiness', color='#3498db')
            ax3.plot(x, pronunciation_scores, marker='s', linewidth=2, label='Pronunciation', color='#e74c3c')
            
            ax3.set_title('🗣️ Voice Analysis Results Over Time', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Test Number')
            ax3.set_ylabel('Score')
            ax3.set_ylim(0, 100)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
    
    def plot_emotion_distribution(self, fig):
        """Plot 4: Emotion Distribution"""
        ax4 = plt.subplot(4, 2, 4)
        
        if self.complete_assessments:
            emotions = {}
            for assessment in self.complete_assessments:
                dominant_emotion = assessment['voice_analysis']['dominant_emotion']
                emotions[dominant_emotion] = emotions.get(dominant_emotion, 0) + 1
            
            if emotions:
                emotion_labels = ['Ready', 'Tired', 'Stressed', 'Calm', 'Uncertain']
                emotion_colors = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db', '#9b59b6']
                
                # Filter only emotions yang ada di data
                existing_emotions = []
                existing_counts = []
                existing_colors = []
                
                for i, emotion in enumerate(['ready', 'tired', 'stressed', 'calm', 'uncertain']):
                    if emotion in emotions:
                        existing_emotions.append(emotion_labels[i])
                        existing_counts.append(emotions[emotion])
                        existing_colors.append(emotion_colors[i])
                
                if existing_emotions:
                    pie_result = ax4.pie(existing_counts, labels=existing_emotions, 
                                         colors=existing_colors, autopct='%1.1f%%', 
                                         startangle=90)
                    
                    ax4.set_title('🎭 Dominant Emotion Distribution', fontsize=14, fontweight='bold')
    
    def plot_cognitive_assessment(self, fig):
        """Plot 5: Cognitive Assessment Results"""
        ax5 = plt.subplot(4, 2, 5)
        
        if self.complete_assessments:
            cognitive_scores = []
            test_types = {}
            
            for assessment in self.complete_assessments:
                cognitive_scores.append(assessment['cognitive_analysis']['average_score'])
                
                # Count test types
                for test in assessment['cognitive_analysis']['individual_results']:
                    test_type = test['test_type']
                    if test_type not in test_types:
                        test_types[test_type] = []
                    test_types[test_type].append(test['score'])
            
            # Plot cognitive scores over time
            x = range(1, len(cognitive_scores) + 1)
            ax5.plot(x, cognitive_scores, marker='o', linewidth=3, color='#9b59b6', markersize=8)
            ax5.set_title('🧠 Cognitive Assessment Scores', fontsize=14, fontweight='bold')
            ax5.set_xlabel('Assessment Number')
            ax5.set_ylabel('Cognitive Score')
            ax5.set_ylim(0, 100)
            ax5.grid(True, alpha=0.3)
            
            # Add average line
            if cognitive_scores:
                avg_score = float(np.mean(cognitive_scores))
                ax5.axhline(y=avg_score, color='red', linestyle='--', alpha=0.7, 
                           label=f'Average: {avg_score:.1f}')
                ax5.legend()
    
    def plot_system_integration(self, fig):
        """Plot 6: Complete System Integration"""
        ax6 = plt.subplot(4, 2, 6)
        
        if self.complete_assessments:
            final_scores = []
            voice_contributions = []
            cognitive_contributions = []
            
            for assessment in self.complete_assessments:
                final_scores.append(assessment['final_assessment']['final_score'])
                voice_contributions.append(assessment['final_assessment']['voice_contribution'])
                cognitive_contributions.append(assessment['final_assessment']['cognitive_contribution'])
            
            x = range(1, len(final_scores) + 1)
            
            # Stacked bar chart
            width = 0.6
            ax6.bar(x, voice_contributions, width, label='Voice (60%)', color='#3498db', alpha=0.8)
            ax6.bar(x, cognitive_contributions, width, bottom=voice_contributions, 
                   label='Cognitive (40%)', color='#e74c3c', alpha=0.8)
            
            # Plot final scores line
            ax6_twin = ax6.twinx()
            ax6_twin.plot(x, final_scores, color='black', marker='o', linewidth=2, 
                         markersize=6, label='Final Score')
            
            ax6.set_title('📊 System Integration: Component Contributions', fontsize=14, fontweight='bold')
            ax6.set_xlabel('Assessment Number')
            ax6.set_ylabel('Component Contribution')
            ax6_twin.set_ylabel('Final Score')
            ax6.legend(loc='upper left')
            ax6_twin.legend(loc='upper right')
            ax6.grid(True, alpha=0.3)
    
    def plot_user_workflow(self, fig):
        """Plot 7: User Journey & Workflow"""
        ax7 = plt.subplot(4, 2, 7)
        
        # Workflow steps dengan estimated time
        steps = ['Voice\nRecording', 'Speech\nRecognition', 'Emotion\nAnalysis', 
                'Cognitive\nTests', 'Final\nAssessment']
        times = [6, 5, 3, 120, 5]  # seconds
        cumulative_times = np.cumsum([0] + times)
        
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
        
        bars = ax7.barh(steps, times, color=colors, alpha=0.8)
        ax7.set_title('⏱️ User Journey Timeline', fontsize=14, fontweight='bold')
        ax7.set_xlabel('Duration (seconds)')
        
        # Add time labels
        for i, (bar, time) in enumerate(zip(bars, times)):
            width = bar.get_width()
            if time >= 60:
                label = f'{time//60}m {time%60}s'
            else:
                label = f'{time}s'
            ax7.text(width + 2, bar.get_y() + bar.get_height()/2, 
                    label, ha='left', va='center', fontweight='bold')
        
        total_time = sum(times)
        ax7.text(0.5, 0.95, f'Total Assessment Time: {total_time//60}m {total_time%60}s', 
                transform=ax7.transAxes, ha='center', fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    def plot_future_roadmap(self, fig):
        """Plot 8: Future Improvements Roadmap"""
        ax8 = plt.subplot(4, 2, 8)
        
        improvements = ['ML Model\nTraining', 'Multi-language\nSupport', 'Wearable\nIntegration', 
                       'Deep Learning\nEmotion', 'Real-time\nMonitoring']
        priorities = [90, 75, 60, 85, 70]  # Priority scores
        complexities = [80, 40, 70, 90, 85]  # Implementation complexity
        
        scatter = ax8.scatter(complexities, priorities, s=200, alpha=0.7, 
                             c=range(len(improvements)), cmap='viridis')
        
        for i, improvement in enumerate(improvements):
            ax8.annotate(improvement, (complexities[i], priorities[i]), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=10, ha='left')
        
        ax8.set_title('🚀 Future Improvements Roadmap', fontsize=14, fontweight='bold')
        ax8.set_xlabel('Implementation Complexity')
        ax8.set_ylabel('Priority Level')
        ax8.set_xlim(30, 100)
        ax8.set_ylim(50, 100)
        ax8.grid(True, alpha=0.3)
        
        # Add quadrant labels
        ax8.text(0.75, 0.95, 'High Priority\nHigh Complexity', transform=ax8.transAxes, 
                ha='center', va='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3))
        ax8.text(0.25, 0.95, 'High Priority\nLow Complexity', transform=ax8.transAxes, 
                ha='center', va='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.3))
    
    def generate_presentation_summary(self):
        """Generate comprehensive summary untuk presentasi"""
        summary = {
            'project_overview': {
                'title': 'Fit-to-Work Voice Readiness Checker',
                'objective': 'Automated assessment of worker readiness through voice analysis for workplace safety',
                'total_tests_conducted': len(self.complete_assessments),
                'total_data_points': sum(len(session.get('test_results', [])) for session in self.session_data)
            },
            'technical_implementation': {
                'components': [
                    'Speech Recognition (Google Speech API)',
                    'Voice Emotion Analysis (Feature Extraction)',
                    'Cognitive Assessment Battery',
                    'Integrated Scoring System'
                ],
                'technologies': [
                    'Python + librosa for audio processing',
                    'SpeechRecognition for speech-to-text',
                    'NumPy/SciPy for signal processing',
                    'Machine Learning for pattern recognition'
                ]
            },
            'dataset_characteristics': {
                'source': 'Real-time user voice recordings',
                'preprocessing': [
                    'Audio normalization (16kHz sampling)',
                    'Ambient noise filtering',
                    'Feature extraction (MFCC, pitch, energy)',
                    'Statistical analysis of voice patterns'
                ],
                'features_extracted': [
                    'Pitch analysis (mean, std, range)',
                    'Energy characteristics (RMS, amplitude)',
                    'Spectral features (MFCC, centroid)',
                    'Temporal features (speech rate, pauses)'
                ]
            }
        }
        
        # Calculate performance metrics
        if self.session_data:
            all_metrics = []
            for session in self.session_data:
                if 'session_metrics' in session:
                    all_metrics.append(session['session_metrics'])
            
            if all_metrics:
                recognition_rates = [m['system_performance']['recognition_rate'] for m in all_metrics]
                pronunciation_scores = [m['pronunciation_analysis']['mean_score'] for m in all_metrics]
                
                summary['model_performance'] = {
                    'speech_recognition_accuracy': f"{np.mean(recognition_rates):.1f}%",
                    'pronunciation_assessment_accuracy': f"{np.mean(pronunciation_scores):.1f}%",
                    'system_reliability': f"{len([r for r in recognition_rates if r > 80]) / len(recognition_rates) * 100:.1f}%"
                }
        
        # Calculate complete assessment metrics
        if self.complete_assessments:
            final_scores = [a['final_assessment']['final_score'] for a in self.complete_assessments]
            voice_scores = [a['voice_analysis']['readiness_score'] for a in self.complete_assessments]
            cognitive_scores = [a['cognitive_analysis']['average_score'] for a in self.complete_assessments]
            
            summary['system_effectiveness'] = {
                'average_final_score': f"{np.mean(final_scores):.1f}/100",
                'average_voice_score': f"{np.mean(voice_scores):.1f}/100",
                'average_cognitive_score': f"{np.mean(cognitive_scores):.1f}/100",
                'score_consistency': f"{100 - np.std(final_scores):.1f}%"
            }
        
        summary['challenges_and_solutions'] = [
            {
                'challenge': 'Speech Recognition Accuracy in Noisy Environments',
                'solution': 'Implemented ambient noise filtering and adaptive thresholds',
                'learning': 'Audio preprocessing significantly impacts recognition quality'
            },
            {
                'challenge': 'Individual Voice Pattern Variations',
                'solution': 'Multi-feature approach combining pitch, energy, and spectral analysis',
                'learning': 'Robust feature extraction compensates for individual differences'
            },
            {
                'challenge': 'Real-time Processing Requirements',
                'solution': 'Optimized feature extraction pipeline and efficient algorithms',
                'learning': 'Balance between accuracy and processing speed is crucial'
            }
        ]
        
        summary['future_improvements'] = [
            'Machine Learning model training with larger datasets',
            'Multi-language support for diverse workplaces',
            'Integration with wearable devices for continuous monitoring',
            'Advanced emotion recognition using deep learning',
            'Personalized adaptation based on individual patterns'
        ]
        
        # Save summary
        with open('presentation_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print("📋 Presentation summary saved: presentation_summary.json")
        return summary
    
    def create_demo_data_summary(self):
        """Create summary tabel untuk presentasi"""
        if not self.complete_assessments:
            print("⚠️ No complete assessment data available")
            return None
        
        # Create DataFrame untuk easy analysis
        demo_data = []
        for i, assessment in enumerate(self.complete_assessments, 1):
            demo_data.append({
                'Test_Number': i,
                'Voice_Score': assessment['voice_analysis']['readiness_score'],
                'Cognitive_Score': assessment['cognitive_analysis']['average_score'],
                'Final_Score': assessment['final_assessment']['final_score'],
                'Status': assessment['final_assessment']['status'],
                'Dominant_Emotion': assessment['voice_analysis']['dominant_emotion'],
                'Pronunciation_Accuracy': assessment['voice_analysis']['pronunciation_score']
            })
        
        df = pd.DataFrame(demo_data)
        df.to_csv('demo_results_summary.csv', index=False)
        
        print("📊 Demo results summary saved: demo_results_summary.csv")
        print("\n📋 QUICK STATISTICS:")
        print(f"   Total Assessments: {len(df)}")
        print(f"   Average Final Score: {df['Final_Score'].mean():.1f}")
        print(f"   Fit to Work: {len(df[df['Status'].str.contains('FIT TO WORK')])}")
        print(f"   Not Fit to Work: {len(df[df['Status'].str.contains('NOT FIT')])}")
        
        return df

def main():
    """Main function untuk generate semua materials presentasi"""
    print("🎨 GENERATING PRESENTATION MATERIALS")
    print("=" * 50)
    
    visualizer = VoiceReadinessVisualizer()
    
    # 1. Create comprehensive visualizations
    visualizer.create_comprehensive_visualizations()
    
    # 2. Generate presentation summary
    summary = visualizer.generate_presentation_summary()
    
    # 3. Create demo data summary
    demo_df = visualizer.create_demo_data_summary()
    
    print("\n✅ PRESENTATION MATERIALS GENERATED:")
    print("   📊 voice_readiness_presentation.png (comprehensive charts)")
    print("   📋 presentation_summary.json (detailed project info)")
    print("   📈 demo_results_summary.csv (test results table)")
    
    print("\n🎯 READY FOR PRESENTATION!")
    print("Semua materials siap untuk capstone presentation!")

if __name__ == "__main__":
    main()
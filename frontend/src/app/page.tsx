'use client';
import { useState } from 'react';
import styles from './page.module.css';

export default function Home() {
  const [symptoms, setSymptoms] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState('');
  const [feedbackSent, setFeedbackSent] = useState(false);

  const handleAnalyze = async (e: any) => {
    e.preventDefault();
    if (!symptoms.trim()) return;

    setLoading(true);
    setError('');
    setResult(null);
    setFeedbackSent(false);

    try {
      const response = await fetch('http://localhost:8000/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: symptoms })
      });

      if (!response.ok) {
        throw new Error('Analysis failed. Please ensure the backend is running.');
      }

      const data = await response.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'An error occurred.');
    } finally {
      setLoading(false);
    }
  };

  const submitFeedback = async (isCorrect: boolean) => {
    if (!result) return;
    try {
      await fetch('http://localhost:8000/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          diagnosis: result.diagnosis,
          is_correct: isCorrect,
          notes: ''
        })
      });
      setFeedbackSent(true);
    } catch (err) {
      console.error('Feedback submission failed', err);
    }
  };

  return (
    <main className={styles.container}>
      <div className={styles.header}>
        <h1 className={styles.title}>MedGuard AI</h1>
        <p className={styles.subtitle}>
          Secure, probabilistic clinical decision support verified by AI reasoning.
        </p>
      </div>

      <div className={styles.inputSection}>
        <form onSubmit={handleAnalyze}>
          <textarea
            className={styles.textarea}
            placeholder="Describe your symptoms in natural language (e.g., 'I have had a severe headache behind my eyes for 3 days and a slight fever...')"
            value={symptoms}
            onChange={(e) => setSymptoms(e.target.value)}
            disabled={loading}
          />
          <div className={styles.btnContainer}>
            <button type="submit" className={styles.primaryBtn} disabled={loading || !symptoms.trim()}>
              {loading ? (
                <><span className={styles.loader}></span> Analyzing...</>
              ) : (
                'Run Diagnostic'
              )}
            </button>
          </div>
        </form>
        {error && <div style={{color: 'var(--danger-color)', marginTop: '1rem'}}>{error}</div>}
      </div>

      {result && (
        <div className={styles.resultSection}>
          <div className={styles.card}>
            <div className={styles.diagnosisHeader}>
              <div>
                <span style={{color: 'var(--text-secondary)', fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '1px'}}>Predicted Diagnosis</span>
                <h2 className={styles.diagnosisTitle}>{result.diagnosis}</h2>
              </div>
              <div className={styles.confidenceBadge}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
                {(result.confidence * 100).toFixed(1)}% Confidence
              </div>
            </div>

            <h3 style={{fontSize: '1rem', color: 'var(--text-secondary)', marginBottom: '0.8rem'}}>Extracted Symptoms:</h3>
            <div className={styles.extractedList}>
              {result.symptoms_extracted.length > 0 ? (
                result.symptoms_extracted.map((sym: string, i: number) => (
                  <span key={i} className={styles.pill}>{sym.replace(/_/g, ' ')}</span>
                ))
              ) : (
                <span className={styles.pill}>None recognized</span>
              )}
            </div>

            <h3 style={{fontSize: '1rem', color: 'var(--text-secondary)', marginBottom: '0.8rem'}}>Clinical Reasoning:</h3>
            <div className={styles.explanation}>
              {result.explanation}
            </div>

            <div className={styles.feedbackSection}>
              {!feedbackSent ? (
                <>
                  <span className={styles.feedbackLabel}>Is this analysis helpful and accurate?</span>
                  <button onClick={() => submitFeedback(true)} className={`${styles.iconBtn} ${styles.up}`}>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path></svg>
                    Yes
                  </button>
                  <button onClick={() => submitFeedback(false)} className={`${styles.iconBtn} ${styles.down}`}>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"></path></svg>
                    No
                  </button>
                </>
              ) : (
                <div className={styles.thankYouMessage}>
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{marginRight: '8px', verticalAlign: 'middle'}}><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
                  Thank you for your feedback! This helps improve the model.
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </main>
  );
}

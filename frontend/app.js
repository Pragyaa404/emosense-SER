// ── CONFIG ───────────────────────────────────────────────────────────────────
// IMPORTANT: Replace this with your actual HuggingFace Space URL once deployed
// Example: "https://your-username-emosense.hf.space"
const API_BASE_URL = "https://pragyaa404-emosense-ser.hf.space";

// ── STATE ────────────────────────────────────────────────────────────────────
let selectedFile = null;       // File object from upload or recording
let lastPredictedEmotion = null;
const EMOTION_LABELS = ["happy", "sadness", "anger", "neutral"];
const EMOTION_COLORS = {
  happy: "var(--c2)",
  sadness: "var(--c4)",
  anger: "var(--c1)",
  neutral: "var(--c3)"
};

// ── TAB SWITCHING ────────────────────────────────────────────────────────────
function switchTab(tab, btn) {
  document.querySelectorAll('.demo-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.demo-tab').forEach(b => b.classList.remove('active'));
  document.getElementById('panel-' + tab).classList.add('active');
  btn.classList.add('active');

  selectedFile = null;
  document.getElementById('analyzeBtn').disabled = true;
  document.getElementById('resultBox').classList.remove('show');
  document.getElementById('uploadStatus').style.display = 'none';
  document.getElementById('errorBox').style.display = 'none';
}

// ── FILE UPLOAD ──────────────────────────────────────────────────────────────
function handleFile(input) {
  if (input.files[0]) {
    selectedFile = input.files[0];
    const status = document.getElementById('uploadStatus');
    status.style.display = 'block';
    status.textContent = '✓ File loaded: ' + selectedFile.name;
    document.getElementById('analyzeBtn').disabled = false;
  }
}

// ── AUDIO RECORDING (uses real browser microphone) ────────────────────────────
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let recInterval = null;
let recSeconds = 0;

async function toggleRecord() {
  if (!isRecording) {
    // Start recording
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);
      audioChunks = [];

      mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        selectedFile = new File([audioBlob], "recording.webm", { type: 'audio/webm' });

        const audioUrl = URL.createObjectURL(audioBlob);
        const playback = document.getElementById('audioPlayback');
        playback.src = audioUrl;
        playback.style.display = 'block';

        document.getElementById('analyzeBtn').disabled = false;
        document.getElementById('recordLabel').textContent = 'Recording saved! Click to record again';
      };

      mediaRecorder.start();
      isRecording = true;
      recSeconds = 0;

      const btn = document.getElementById('recordBtn');
      btn.classList.add('recording');
      btn.textContent = '⏹️';
      document.getElementById('recordLabel').textContent = 'Recording... click to stop';

      recInterval = setInterval(() => {
        recSeconds++;
        document.getElementById('recTimer').textContent = recSeconds + 's recorded';
      }, 1000);

    } catch (err) {
      showError('Microphone access denied or unavailable. Please allow microphone access and try again.');
    }

  } else {
    // Stop recording
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(track => track.stop());
    isRecording = false;
    clearInterval(recInterval);

    const btn = document.getElementById('recordBtn');
    btn.classList.remove('recording');
    btn.textContent = '🎙️';
  }
}

// ── ERROR HANDLING ───────────────────────────────────────────────────────────
function showError(message) {
  const errorBox = document.getElementById('errorBox');
  errorBox.textContent = message;
  errorBox.style.display = 'block';
}

function hideError() {
  document.getElementById('errorBox').style.display = 'none';
}

// ── ANALYZE — calls the real /predict endpoint ────────────────────────────────
async function analyze() {
  if (!selectedFile) {
    showError('Please upload or record audio first.');
    return;
  }

  hideError();
  const btn = document.getElementById('analyzeBtn');
  btn.textContent = 'Analyzing...';
  btn.disabled = true;

  try {
    const formData = new FormData();
    formData.append('file', selectedFile);

    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }

    const result = await response.json();
    displayResult(result);

  } catch (err) {
    showError(
      'Could not reach the prediction server. Make sure the backend is deployed and API_BASE_URL in app.js is set correctly. Error: ' + err.message
    );
  } finally {
    btn.textContent = 'Analyze Emotion →';
    btn.disabled = false;
  }
}

// ── DISPLAY RESULT ───────────────────────────────────────────────────────────
function displayResult(result) {
  lastPredictedEmotion = result.emotion;

  const box = document.getElementById('resultBox');
  const emotionEl = document.getElementById('resultEmotion');
  const confBars = document.getElementById('confBars');

  emotionEl.textContent = result.emotion;
  emotionEl.style.color = EMOTION_COLORS[result.emotion] || 'var(--text)';

  confBars.innerHTML = '';
  // result.confidence is an object like { happy: 91.2, sadness: 3.1, ... }
  Object.entries(result.confidence)
    .sort((a, b) => b[1] - a[1])
    .forEach(([name, score]) => {
      confBars.innerHTML += `
        <div class="conf-row">
          <span class="conf-name">${name}</span>
          <div class="conf-track"><div class="conf-fill" style="width:${score}%;background:${EMOTION_COLORS[name] || '#888'}"></div></div>
          <span class="conf-pct">${score}%</span>
        </div>`;
    });

  // Build feedback buttons for continuous learning
  const feedbackBtns = document.getElementById('feedbackBtns');
  feedbackBtns.innerHTML = '';
  EMOTION_LABELS.forEach(emotion => {
    const isCorrectGuess = emotion === result.emotion;
    feedbackBtns.innerHTML += `
      <button class="feedback-btn ${isCorrectGuess ? 'correct' : ''}" onclick="submitFeedback('${emotion}')">
        ${isCorrectGuess ? '✓ ' : ''}${emotion}
      </button>`;
  });
  document.getElementById('feedbackStatus').textContent = '';

  box.classList.add('show');
}

// ── FEEDBACK — calls /feedback endpoint for continuous learning ───────────────
async function submitFeedback(correctEmotion) {
  if (!selectedFile || !lastPredictedEmotion) return;

  try {
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('predicted_emotion', lastPredictedEmotion);
    formData.append('correct_emotion', correctEmotion);

    const response = await fetch(`${API_BASE_URL}/feedback`, {
      method: 'POST',
      body: formData
    });

    if (response.ok) {
      document.getElementById('feedbackStatus').textContent =
        `✓ Thanks! Saved as "${correctEmotion}" for future model improvement.`;
    } else {
      document.getElementById('feedbackStatus').textContent =
        '⚠ Could not save feedback right now.';
    }
  } catch (err) {
    document.getElementById('feedbackStatus').textContent =
      '⚠ Could not reach server to save feedback.';
  }
}

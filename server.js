require('dotenv').config();
const express = require('express');
const cors = require('cors');
const rateLimit = require('express-rate-limit');
const NodeCache = require('node-cache');
const faceapi = require('@vladmandic/face-api');
const canvas = require('canvas');
const tf = require('@tensorflow/tfjs-node');
const crypto = require('crypto');

const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const app = express();
const PORT = process.env.PORT || 3001;

// Required for Railway / Render / Heroku reverse proxy — lets express-rate-limit
// see the real client IP instead of the proxy IP, avoiding false rate-limit crashes
app.set('trust proxy', 1);

const cache = new NodeCache({ stdTTL: 3600, checkperiod: 120 });

let totalCount = 0;

app.use(cors({
  origin: '*',
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type'],
  credentials: false
}));

app.options('*', cors());

// Body parser BEFORE rate limiter so req.body is available if limiter needs it
app.use(express.json({ limit: "25mb" }));

const limiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 50,
  message: { error: 'Too many requests, please try again later.' },
  standardHeaders: true,
  legacyHeaders: false,
});

app.use('/analyze', limiter);
app.use('/landmarks', limiter);

let modelsLoaded = false;
let requestCount = 0;
let cacheHits = 0;
let totalProcessingTime = 0;

async function loadModels() {
  if (modelsLoaded) {
    console.log('Models already loaded');
    return;
  }
  
  console.log('Loading AI models...');
  const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model';
  
  try {
    await Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
      faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
      faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL),
    ]);
    
    modelsLoaded = true;
    console.log('✅ Models loaded successfully');
  } catch (error) {
    console.error('❌ Failed to load models:', error);
    throw error;
  }
}

function calculateEyeConstriction(landmarks) {
  const leftEye = landmarks.getLeftEye();
  const rightEye = landmarks.getRightEye();
  
  const leftEyeWidth = Math.abs(leftEye[3].x - leftEye[0].x);
  const leftEyeHeight = Math.abs(leftEye[1].y - leftEye[5].y);
  const rightEyeWidth = Math.abs(rightEye[3].x - rightEye[0].x);
  const rightEyeHeight = Math.abs(rightEye[1].y - rightEye[5].y);
  
  const leftAspectRatio = leftEyeWidth / leftEyeHeight;
  const rightAspectRatio = rightEyeWidth / rightEyeHeight;
  const avgAspectRatio = (leftAspectRatio + rightAspectRatio) / 2;
  
  const eyeConstriction = Math.max(0, Math.min(1, (avgAspectRatio - 3) / 3));
  
  return Math.round(eyeConstriction * 100);
}

function calculateCheekRaise(landmarks) {
  const leftEye = landmarks.getLeftEye();
  const rightEye = landmarks.getRightEye();
  const nose = landmarks.getNose();
  const mouth = landmarks.getMouth();
  
  const leftEyeBottom = leftEye[4].y;
  const rightEyeBottom = rightEye[4].y;
  const avgEyeBottom = (leftEyeBottom + rightEyeBottom) / 2;
  
  const noseBridge = nose[0].y;
  const upperLip = mouth[14].y;
  
  const faceHeight = upperLip - noseBridge;
  const eyeToNose = avgEyeBottom - noseBridge;
  
  const cheekRaise = eyeToNose / faceHeight;
  const normalized = Math.max(0, Math.min(1, cheekRaise * 1.5));
  
  return Math.round(normalized * 100);
}

function calculateMouthCurve(landmarks) {
  const mouth = landmarks.getMouth();
  
  const leftCorner = mouth[0];
  const rightCorner = mouth[6];
  const topCenter = mouth[14];
  const bottomCenter = mouth[18];
  
  const mouthWidth = Math.abs(rightCorner.x - leftCorner.x);
  const mouthHeight = Math.abs(bottomCenter.y - topCenter.y);
  
  const aspectRatio = mouthWidth / mouthHeight;
  const normalized = Math.max(0, Math.min(1, (aspectRatio - 2) / 3));
  
  return Math.round(normalized * 100);
}

function calculateSymmetry(landmarks) {
  const jaw = landmarks.getJawOutline();
  const nose = landmarks.getNose();
  
  const centerX = nose[3].x;
  
  let leftDistance = 0;
  let rightDistance = 0;
  
  for (let i = 0; i < jaw.length / 2; i++) {
    leftDistance += Math.abs(jaw[i].x - centerX);
    rightDistance += Math.abs(jaw[jaw.length - 1 - i].x - centerX);
  }
  
  const symmetryRatio = Math.min(leftDistance, rightDistance) / Math.max(leftDistance, rightDistance);
  
  return Math.round(symmetryRatio * 100);
}

function hashImage(base64Data) {
  return crypto.createHash('md5').update(base64Data).digest('hex');
}

app.get('/count', (req, res) => {
  res.json({ count: totalCount });
});

app.post('/count/increment', (req, res) => {
  totalCount++;
  console.log(`📊 Count incremented to: ${totalCount}`);
  res.json({ count: totalCount, success: true });
});

app.get('/', (req, res) => {
  res.json({ 
    status: 'ok', 
    message: 'RealSmile API Server',
    modelsLoaded,
    totalAnalyses: totalCount,
    stats: {
      totalRequests: requestCount,
      cacheHits: cacheHits,
      cacheHitRate: requestCount > 0 ? ((cacheHits / requestCount) * 100).toFixed(1) + '%' : '0%',
      avgProcessingTime: requestCount > 0 ? (totalProcessingTime / requestCount).toFixed(2) + 'ms' : '0ms'
    }
  });
});

app.get('/health', (req, res) => {
  res.json({ status: 'healthy', modelsLoaded });
});

// ═══════════════════════════════════════════════════════
// SMILE ANALYSIS (existing)
// ═══════════════════════════════════════════════════════

app.post('/analyze', async (req, res) => {
  const startTime = Date.now();
  requestCount++;
  
  console.log(`📸 Analysis request #${requestCount} from ${req.ip}`);
  
  try {
    const { image } = req.body;
    
    if (!image) {
      console.log('❌ No image in request body');
      return res.status(400).json({ error: 'No image provided' });
    }
    
    console.log(`📊 Image size: ${Math.round(image.length / 1024)}KB`);
    
    const base64Data = image.replace(/^data:image\/\w+;base64,/, '');
    const imageHash = hashImage(base64Data);
    const cachedResult = cache.get(imageHash);
    
    if (cachedResult) {
      cacheHits++;
      console.log(`✅ Cache hit! (${cacheHits}/${requestCount})`);
      return res.json(cachedResult);
    }
    
    console.log('🔄 Loading models...');
    await loadModels();
    
    console.log('🖼️  Processing image...');
    const buffer = Buffer.from(base64Data, 'base64');
    const img = new Image();
    img.src = buffer;
    
    console.log('🔍 Detecting faces...');
    
    const detections = await faceapi
      .detectAllFaces(img, new faceapi.TinyFaceDetectorOptions({
        inputSize: 512,
        scoreThreshold: 0.5
      }))
      .withFaceLandmarks()
      .withFaceExpressions();
    
    const processingTime = Date.now() - startTime;
    totalProcessingTime += processingTime;
    
    console.log(`✅ Found ${detections.length} face(s) in ${processingTime}ms`);
    
    if (!detections || detections.length === 0) {
      console.log('⚠️  No faces detected');
      return res.json({ people: [] });
    }
    
    const people = detections.map((detection) => {
      const landmarks = detection.landmarks;
      const expressions = detection.expressions;
      const box = detection.detection.box;
      
      const eyeConstriction = calculateEyeConstriction(landmarks);
      const cheekRaise = calculateCheekRaise(landmarks);
      const mouthCurve = calculateMouthCurve(landmarks);
      const symmetry = calculateSymmetry(landmarks);
      
      const happiness = expressions.happy || 0;
      
      const geometricScore = (
        eyeConstriction * 0.25 +
        cheekRaise * 0.25 +
        mouthCurve * 0.35 +
        symmetry * 0.15
      );
      
      const aiScore = happiness * 100;
      
      const blendedScore = (geometricScore * 0.4) + (aiScore * 0.6);
      const finalScore = Math.round(blendedScore);
      
      const isGenuine = finalScore >= 55 || happiness > 0.5;
      
      const verdict = isGenuine 
        ? finalScore >= 75 ? "Genuine Joy! 😄" : "Real Smile 😊"
        : finalScore >= 35 ? "Polite Smile 😐" : "Fake Smile! 😬";
      
      return {
        score: finalScore,
        isGenuine,
        verdict,
        metrics: {
          eyeConstriction,
          cheekRaise,
          mouthCurve,
          symmetry
        },
        boundingBox: {
          x: box.x / img.width,
          y: box.y / img.height,
          width: box.width / img.width,
          height: box.height / img.height
        }
      };
    });
    
    const result = { people };
    
    totalCount += people.length;
    console.log(`📊 Total count: ${totalCount}`);
    
    cache.set(imageHash, result);
    console.log(`💾 Cached result`);
    
    res.json(result);
    
  } catch (error) {
    const processingTime = Date.now() - startTime;
    console.error(`❌ Analysis error after ${processingTime}ms:`, error);
    res.status(500).json({ 
      error: 'Analysis failed', 
      message: error.message 
    });
  }
});

// ═══════════════════════════════════════════════════════
// LANDMARKS EXTRACTION (for looksmaxxing test on mobile)
// ═══════════════════════════════════════════════════════

app.post('/landmarks', async (req, res) => {
  const startTime = Date.now();
  requestCount++;
  
  console.log(`🔬 Landmarks request #${requestCount} from ${req.ip}`);
  
  try {
    const { image } = req.body;
    
    if (!image) {
      return res.status(400).json({ error: 'No image provided' });
    }
    
    console.log(`📊 Image size: ${Math.round(image.length / 1024)}KB`);
    
    const base64Data = image.replace(/^data:image\/\w+;base64,/, '');
    const imageHash = 'lm_' + hashImage(base64Data);
    const cachedResult = cache.get(imageHash);
    
    if (cachedResult) {
      cacheHits++;
      console.log(`✅ Landmarks cache hit!`);
      return res.json(cachedResult);
    }
    
    await loadModels();
    
    const buffer = Buffer.from(base64Data, 'base64');
    const img = new Image();
    img.src = buffer;
    
    const detections = await faceapi
      .detectAllFaces(img, new faceapi.TinyFaceDetectorOptions({
        inputSize: 512,
        scoreThreshold: 0.3
      }))
      .withFaceLandmarks()
      .withFaceExpressions();
    
    const processingTime = Date.now() - startTime;
    totalProcessingTime += processingTime;
    
    if (!detections || detections.length === 0) {
      console.log('⚠️  No faces detected for landmarks');
      return res.json({ landmarks: null, error: 'No face detected' });
    }
    
    // Pick largest face
    const detection = detections.sort((a, b) => 
      b.detection.box.width * b.detection.box.height - a.detection.box.width * a.detection.box.height
    )[0];
    
    // Return raw 68-point landmarks
    const positions = detection.landmarks.positions.map(p => ({
      x: p.x || p._x,
      y: p.y || p._y
    }));
    
    const result = {
      landmarks: positions,
      imageWidth: img.width,
      imageHeight: img.height,
      confidence: detection.detection.score,
      processingTime
    };
    
    cache.set(imageHash, result);
    totalCount++;
    
    console.log(`✅ Landmarks extracted (${positions.length} points) in ${processingTime}ms`);
    res.json(result);
    
  } catch (error) {
    console.error(`❌ Landmarks error:`, error);
    res.status(500).json({ error: 'Landmarks extraction failed', message: error.message });
  }
});

// ═══════════════════════════════════════════════════════
// EMAIL / SUBSCRIBER ENDPOINT
// ═══════════════════════════════════════════════════════

const subscribers = new Map(); // email → { source, score, tier, date }

app.post('/subscribe', async (req, res) => {
  try {
    const { email, source, score, tier } = req.body || {};
    if (!email || !email.includes('@')) {
      return res.status(400).json({ error: 'Invalid email' });
    }

    const isNew = !subscribers.has(email);
    subscribers.set(email, { source, score, tier, date: new Date().toISOString() });
    console.log(`[SUBSCRIBE] ${email} — ${source || 'unknown'} — tier:${tier || 'none'} — score:${score || 'n/a'} — new:${isNew}`);

    // Send email via Resend if configured
    const resendKey = process.env.RESEND_API_KEY;
    if (resendKey && isNew) {
      try {
        const { Resend } = require('resend');
        const resend = new Resend(resendKey);
        const isPurchase = source === 'purchase';
        const isPro = tier === 'pro';

        await resend.emails.send({
          from: 'RealSmile <noreply@send.realsmile.online>',
          to: email,
          subject: isPurchase
            ? (isPro ? '⚡ Your Pro Report is Ready' : '✓ Your Full Report is Unlocked')
            : '📊 Your RealSmile Analysis',
          html: isPurchase ? buildPurchaseEmail(score, isPro) : buildLeadEmail(score),
        });
        console.log(`[EMAIL SENT] ${email} — ${source}`);
      } catch (err) {
        console.error('[RESEND ERROR]', err.message);
      }
    }

    res.json({ success: true });
  } catch (err) {
    console.error('[SUBSCRIBE ERROR]', err);
    res.json({ success: true });
  }
});

app.get('/subscribers/count', (req, res) => {
  res.json({ count: subscribers.size });
});

function buildPurchaseEmail(score, isPro) {
  const color = isPro ? '#4f46e5' : '#10b981';
  const icon = isPro ? '⚡' : '✓';
  const features = ['All 10 facial metrics with scores', 'Percentile rankings vs analyzed faces', 'Personalized glow-up action plan', 'Downloadable PDF report',
    ...(isPro ? ['Unlimited progress rescans', 'Before/after metric comparisons', 'Embed widget access'] : [])];
  return `<!DOCTYPE html><html><head><meta charset="utf-8"></head>
<body style="background:#0a0a0a;color:#fff;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;padding:0">
  <div style="max-width:480px;margin:0 auto;padding:40px 24px">
    <div style="text-align:center;margin-bottom:32px">
      <div style="display:inline-block;background:${color};border-radius:50%;width:48px;height:48px;line-height:48px;font-size:24px;text-align:center">${icon}</div>
      <h1 style="font-size:22px;font-weight:900;margin:16px 0 4px;letter-spacing:-0.03em">${isPro ? 'Pro Report Unlocked' : 'Full Report Unlocked'}</h1>
    </div>
    ${score ? `<div style="background:#111;border:1px solid #222;border-radius:16px;padding:20px;text-align:center;margin-bottom:24px">
      <p style="color:#6b7280;font-size:12px;text-transform:uppercase;letter-spacing:0.1em;margin:0 0 4px">Your Overall Score</p>
      <p style="font-size:48px;font-weight:900;margin:0;color:#fff">${score}</p>
    </div>` : ''}
    ${features.map(f => `<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px"><span style="color:#10b981;font-weight:bold">✓</span><span style="font-size:14px;color:#d1d5db">${f}</span></div>`).join('')}
    <a href="https://realsmile.online/looksmaxxing-test" style="display:block;background:#fff;color:#000;text-align:center;padding:14px 24px;border-radius:50px;font-weight:900;font-size:15px;text-decoration:none;margin:24px 0 16px">
      View Your Full Report →
    </a>
    ${!isPro ? `<div style="background:#1e1b4b;border:1px solid #4f46e5;border-radius:12px;padding:16px;margin-bottom:24px">
      <p style="font-size:13px;color:#c7d2fe;margin:0">Want to track progress? <a href="https://realsmile.online/looksmaxxing-test" style="color:#818cf8;font-weight:700">Upgrade to Pro →</a></p>
    </div>` : ''}
    <p style="font-size:11px;color:#374151;text-align:center">RealSmile · 7-day money-back guarantee</p>
  </div>
</body></html>`;
}

function buildLeadEmail(score) {
  return `<!DOCTYPE html><html><head><meta charset="utf-8"></head>
<body style="background:#0a0a0a;color:#fff;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;padding:0">
  <div style="max-width:480px;margin:0 auto;padding:40px 24px">
    <h1 style="font-size:22px;font-weight:900;text-align:center;margin:0 0 8px;letter-spacing:-0.03em">Your Analysis is Ready</h1>
    ${score ? `<p style="font-size:15px;color:#9ca3af;text-align:center;margin:0 0 24px">You scored <strong style="color:#fff">${score}/100</strong></p>` : ''}
    <a href="https://realsmile.online/looksmaxxing-test" style="display:block;background:#fff;color:#000;text-align:center;padding:14px 24px;border-radius:50px;font-weight:900;font-size:15px;text-decoration:none;margin-bottom:24px">
      See Full Report — from $4.99 →
    </a>
    <p style="font-size:11px;color:#374151;text-align:center">RealSmile · realsmile.online</p>
  </div>
</body></html>`;
}

// ═══════════════════════════════════════════════════════
// START SERVER
// ═══════════════════════════════════════════════════════

console.log('🚀 Starting server...');
loadModels().then(() => {
  console.log('✅ Startup complete');
}).catch(err => {
  console.error('❌ Failed to preload models:', err);
});

app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
  console.log(`💰 Rate limit: 50 requests per 15 minutes per IP`);
  console.log(`💾 Cache enabled: 1 hour TTL`);
  console.log(`📊 Total analyses: ${totalCount}`);
});

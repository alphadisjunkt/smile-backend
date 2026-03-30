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

const subscribers = new Map(); // email → { source, score, tier, date, purchased, nurseTimers }

// Full nurture sequence: day 1, day 3, day 7
function scheduleNurtureSequence(email, score) {
  const resendKey = process.env.RESEND_API_KEY;
  if (!resendKey) return [];

  async function sendIfNotPurchased(subject, html, tag) {
    const sub = subscribers.get(email);
    if (!sub || sub.purchased) return;
    try {
      const { Resend } = require('resend');
      const resend = new Resend(resendKey);
      await resend.emails.send({ from: 'RealSmile <noreply@realsmile.online>', to: email, subject, html });
      console.log(`[NURTURE ${tag}] Sent to ${email}`);
    } catch (err) {
      console.error(`[NURTURE ${tag} ERROR]`, err.message);
    }
  }

  const t1 = setTimeout(() => sendIfNotPurchased(
    '📊 Your looksmax report is still waiting',
    buildAbandonEmail(score),
    'DAY1'
  ), 24 * 60 * 60 * 1000);

  const t3 = setTimeout(() => sendIfNotPurchased(
    `What a ${score ? score + '/100' : 'good'} score actually means for your dating life`,
    buildDay3Email(score),
    'DAY3'
  ), 3 * 24 * 60 * 60 * 1000);

  const t7 = setTimeout(() => sendIfNotPurchased(
    '⏰ Last chance — your $4.99 report offer expires soon',
    buildDay7Email(score),
    'DAY7'
  ), 7 * 24 * 60 * 60 * 1000);

  return [t1, t3, t7];
}

// Legacy alias so existing call sites still work
function scheduleAbandonEmail(email, score) {
  return scheduleNurtureSequence(email, score)[0];
}

function buildAbandonEmail(score) {
  return `<!DOCTYPE html><html><head><meta charset="utf-8"></head>
<body style="background:#0a0a0a;color:#fff;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;padding:0">
  <div style="max-width:480px;margin:0 auto;padding:40px 24px">
    <h1 style="font-size:22px;font-weight:900;text-align:center;margin:0 0 8px;letter-spacing:-0.03em">Your report is still available</h1>
    ${score ? `<p style="font-size:15px;color:#9ca3af;text-align:center;margin:0 0 24px">You scored <strong style="color:#fff">${score}/100</strong> — see exactly what to improve</p>` : ''}
    <div style="background:#111;border:1px solid #1f2937;border-radius:16px;padding:20px;margin-bottom:24px">
      <p style="font-size:13px;color:#9ca3af;margin:0 0 12px">Your full report includes:</p>
      ${['All 10 facial metrics scored and ranked', 'Your percentile vs analyzed faces', 'Which metrics need the most work', 'A personalized glow-up plan', 'Downloadable PDF'].map(f =>
        `<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px"><span style="color:#10b981;font-weight:bold;font-size:12px">✓</span><span style="font-size:13px;color:#d1d5db">${f}</span></div>`
      ).join('')}
    </div>
    <a href="https://realsmile.online/looksmaxxing-test" style="display:block;background:#fff;color:#000;text-align:center;padding:14px 24px;border-radius:50px;font-weight:900;font-size:15px;text-decoration:none;margin-bottom:12px">
      Unlock Full Report — $4.99 →
    </a>
    <p style="font-size:11px;color:#374151;text-align:center;margin:0">7-day money-back guarantee · One-time payment · realsmile.online</p>
  </div>
</body></html>`;
}

function buildDay3Email(score) {
  const tier = score >= 85 ? 'elite' : score >= 70 ? 'above average' : score >= 50 ? 'average' : 'below average';
  const headline = score
    ? `You scored ${score}/100 — here's what that actually means`
    : "Here's what your looksmax score actually means";
  const insight = score >= 70
    ? `A ${score}/100 puts you in the top tier. Most people who score this high see the biggest gains from optimizing <strong style="color:#fff">one specific metric</strong> — not overhauling everything.`
    : `A ${score}/100 means there are 2-3 specific metrics dragging your overall score down. The good news: these are almost always fixable without surgery or major lifestyle changes.`;

  return `<!DOCTYPE html><html><head><meta charset="utf-8"></head>
<body style="background:#0a0a0a;color:#fff;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;padding:0">
  <div style="max-width:480px;margin:0 auto;padding:40px 24px">
    <p style="font-size:12px;color:#4b5563;text-align:center;margin:0 0 24px;text-transform:uppercase;letter-spacing:0.1em">RealSmile · Day 3 Check-in</p>
    <h1 style="font-size:22px;font-weight:900;margin:0 0 16px;letter-spacing:-0.03em;line-height:1.3">${headline}</h1>
    <p style="font-size:14px;color:#9ca3af;margin:0 0 20px;line-height:1.6">${insight}</p>
    <div style="background:#111;border:1px solid #1f2937;border-radius:16px;padding:20px;margin-bottom:24px">
      <p style="font-size:12px;font-weight:700;color:#6b7280;text-transform:uppercase;letter-spacing:0.1em;margin:0 0 12px">The 3 metrics that matter most</p>
      ${[
        { metric: 'Canthal Tilt', impact: 'Perceived confidence & attractiveness. Most improvable with grooming alone.' },
        { metric: 'Jawline Angle', impact: 'Facial structure signal. Responds to body fat % and mastic gum chewing.' },
        { metric: 'Facial Symmetry', impact: 'Baseline attractiveness. Often improvable with posture and skincare.' },
      ].map(m => `<div style="margin-bottom:14px">
        <p style="font-size:13px;font-weight:700;color:#fff;margin:0 0 2px">${m.metric}</p>
        <p style="font-size:12px;color:#6b7280;margin:0">${m.impact}</p>
      </div>`).join('')}
    </div>
    <p style="font-size:14px;color:#9ca3af;margin:0 0 20px;line-height:1.6">Your full report shows where <em>you specifically</em> rank on each metric — and exactly which one to fix first for the biggest result.</p>
    <a href="https://realsmile.online/looksmaxxing-test" style="display:block;background:#fff;color:#000;text-align:center;padding:14px 24px;border-radius:50px;font-weight:900;font-size:15px;text-decoration:none;margin-bottom:12px">
      Get My Full Report — $4.99 →
    </a>
    <p style="font-size:11px;color:#374151;text-align:center;margin:0">7-day money-back guarantee · One-time · realsmile.online</p>
  </div>
</body></html>`;
}

function buildDay7Email(score) {
  return `<!DOCTYPE html><html><head><meta charset="utf-8"></head>
<body style="background:#0a0a0a;color:#fff;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;padding:0">
  <div style="max-width:480px;margin:0 auto;padding:40px 24px">
    <div style="background:#7f1d1d;border:1px solid #991b1b;border-radius:12px;padding:12px 16px;text-align:center;margin-bottom:24px">
      <p style="font-size:12px;font-weight:700;color:#fca5a5;margin:0;text-transform:uppercase;letter-spacing:0.08em">Final notice — offer expires soon</p>
    </div>
    <h1 style="font-size:22px;font-weight:900;margin:0 0 8px;letter-spacing:-0.03em">One last time — your report is $4.99</h1>
    ${score ? `<p style="font-size:15px;color:#9ca3af;margin:0 0 20px">Your score: <strong style="color:#fff;font-size:20px">${score}/100</strong></p>` : ''}
    <p style="font-size:14px;color:#9ca3af;margin:0 0 20px;line-height:1.6">
      Most guys who <em>don't</em> unlock their report end up spending months on the wrong things — hitting the gym hard when it's actually their canthal tilt and brow positioning holding them back.
    </p>
    <p style="font-size:14px;color:#9ca3af;margin:0 0 24px;line-height:1.6">
      $4.99 is less than a coffee. The report shows you exactly what to fix — and what you can skip entirely.
    </p>
    <a href="https://realsmile.online/looksmaxxing-test" style="display:block;background:#fff;color:#000;text-align:center;padding:16px 24px;border-radius:50px;font-weight:900;font-size:16px;text-decoration:none;margin-bottom:12px">
      Unlock My Report — $4.99 →
    </a>
    <p style="font-size:11px;color:#374151;text-align:center;margin:0">7-day money-back guarantee · One-time payment · No recurring charges</p>
    <p style="font-size:10px;color:#1f2937;text-align:center;margin:12px 0 0">This is our final email. You won't hear from us again unless you choose to.</p>
  </div>
</body></html>`;
}

app.post('/subscribe', async (req, res) => {
  try {
    const { email, source, score, tier } = req.body || {};
    if (!email || !email.includes('@')) {
      return res.status(400).json({ error: 'Invalid email' });
    }

    const existing = subscribers.get(email) || {};
    const isNew = !existing.date;
    const isPurchaseNow = source === 'purchase';

    // Cancel all pending nurture timers if this is a purchase
    if (isPurchaseNow && existing.nurtureTimers) {
      existing.nurtureTimers.forEach(t => t && clearTimeout(t));
    }

    const nurtureTimers = (!isPurchaseNow && isNew) ? scheduleNurtureSequence(email, score) : (existing.nurtureTimers || []);
    subscribers.set(email, { source, score, tier, date: new Date().toISOString(), purchased: isPurchaseNow || existing.purchased, nurtureTimers });
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
          from: 'RealSmile <noreply@realsmile.online>',
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
// WIDGET PRO — KEY STORAGE + ENDPOINTS
// ═══════════════════════════════════════════════════════

const fs = require('fs');
const KEYS_FILE = './widget-keys.json';
const widgetKeys = new Map(); // key → { customerId, sessionId, subscriptionId, active, theme, email, createdAt }

// Load persisted keys on startup
try {
  const raw = fs.readFileSync(KEYS_FILE, 'utf8');
  Object.entries(JSON.parse(raw)).forEach(([k, v]) => widgetKeys.set(k, v));
  console.log(`[WIDGET PRO] Loaded ${widgetKeys.size} keys`);
} catch {}

function saveWidgetKeys() {
  try {
    fs.writeFileSync(KEYS_FILE, JSON.stringify(Object.fromEntries(widgetKeys), null, 2));
  } catch (err) {
    console.error('[WIDGET KEYS SAVE ERROR]', err.message);
  }
}

function generateWidgetKey() {
  return `rspro_${crypto.randomBytes(20).toString('hex')}`;
}

// POST /widget-subscribe — create Stripe subscription checkout session
app.post('/widget-subscribe', async (req, res) => {
  try {
    const { plan = 'monthly', email } = req.body || {};
    const stripeKey = process.env.STRIPE_SECRET_KEY;
    if (!stripeKey) return res.status(500).json({ error: 'Stripe not configured' });

    const Stripe = require('stripe');
    const stripe = Stripe(stripeKey);

    const session = await stripe.checkout.sessions.create({
      mode: 'subscription',
      payment_method_types: ['card'],
      customer_email: email || undefined,
      line_items: [{
        price_data: {
          currency: 'usd',
          product_data: {
            name: 'Widget Pro',
            description: 'White-label AI face analysis widgets — no branding, custom themes, analytics',
          },
          unit_amount: plan === 'annual' ? 7900 : 900,
          recurring: { interval: plan === 'annual' ? 'year' : 'month' },
        },
        quantity: 1,
      }],
      success_url: 'https://realsmile.online/widget/pro?session_id={CHECKOUT_SESSION_ID}',
      cancel_url: 'https://realsmile.online/widget',
      metadata: { source: 'widget-pro', plan },
      allow_promotion_codes: true,
    });

    console.log(`[WIDGET PRO] Checkout session created: ${session.id}`);
    res.json({ url: session.url });
  } catch (err) {
    console.error('[WIDGET-SUBSCRIBE ERROR]', err.message);
    res.status(500).json({ error: err.message });
  }
});

// GET /widget-auth?key=xxx — validate key and return theme config
app.get('/widget-auth', (req, res) => {
  const { key } = req.query;
  if (!key) return res.json({ valid: false });

  const cacheHit = cache.get(`wk:${key}`);
  if (cacheHit !== undefined) return res.json(cacheHit);

  const data = widgetKeys.get(key);
  const result = data && data.active
    ? { valid: true, theme: data.theme || {} }
    : { valid: false };

  cache.set(`wk:${key}`, result, 3600);
  res.json(result);
});

// GET /widget-key?session_id=xxx — retrieve or generate key after checkout
app.get('/widget-key', async (req, res) => {
  const { session_id } = req.query;
  if (!session_id) return res.status(400).json({ error: 'missing session_id' });

  // Check if already generated for this session
  for (const [key, data] of widgetKeys.entries()) {
    if (data.sessionId === session_id) {
      return res.json({ key, email: data.email });
    }
  }

  try {
    const stripeKey = process.env.STRIPE_SECRET_KEY;
    if (!stripeKey) return res.status(500).json({ error: 'Stripe not configured' });

    const Stripe = require('stripe');
    const stripe = Stripe(stripeKey);
    const session = await stripe.checkout.sessions.retrieve(session_id);

    if (session.metadata?.source !== 'widget-pro') {
      return res.status(400).json({ error: 'Not a Widget Pro purchase' });
    }
    if (session.status !== 'complete') {
      return res.status(402).json({ error: 'payment_required' });
    }

    // Check if key exists for this customer
    for (const [key, data] of widgetKeys.entries()) {
      if (data.customerId === session.customer) {
        return res.json({ key, email: data.email });
      }
    }

    // Generate new key
    const key = generateWidgetKey();
    const entry = {
      customerId: session.customer,
      sessionId: session_id,
      subscriptionId: session.subscription,
      active: true,
      theme: {},
      email: session.customer_details?.email,
      createdAt: new Date().toISOString(),
    };
    widgetKeys.set(key, entry);
    saveWidgetKeys();
    console.log(`[WIDGET PRO] Key generated: ${key.slice(0, 16)}... for ${entry.email}`);

    res.json({ key, email: entry.email });
  } catch (err) {
    console.error('[WIDGET-KEY ERROR]', err.message);
    res.status(500).json({ error: err.message });
  }
});

// PATCH /widget-theme — update theme for a key
app.patch('/widget-theme', (req, res) => {
  const { key, theme } = req.body || {};
  if (!key || !widgetKeys.has(key)) return res.status(404).json({ error: 'Invalid key' });
  const data = widgetKeys.get(key);
  widgetKeys.set(key, { ...data, theme: { ...data.theme, ...theme } });
  cache.del(`wk:${key}`); // invalidate cache
  saveWidgetKeys();
  res.json({ success: true });
});

// GET /widget-usage?key=xxx — basic usage stats (placeholder for analytics)
app.get('/widget-usage', (req, res) => {
  const { key } = req.query;
  if (!key || !widgetKeys.has(key)) return res.status(404).json({ error: 'Invalid key' });
  const data = widgetKeys.get(key);
  res.json({ key: key.slice(0, 16) + '...', active: data.active, createdAt: data.createdAt, email: data.email });
});

// POST /widget-recover — send key to email (no accounts needed)
app.post('/widget-recover', async (req, res) => {
  const { email } = req.body || {};
  if (!email || !email.includes('@')) return res.status(400).json({ error: 'Invalid email' });

  const resendKey = process.env.RESEND_API_KEY;
  if (!resendKey) return res.status(500).json({ error: 'Email not configured' });

  // Find key by email
  let found = null;
  for (const [key, data] of widgetKeys.entries()) {
    if (data.email === email && data.active) { found = key; break; }
  }

  // Always respond success to prevent email enumeration
  if (found) {
    try {
      const { Resend } = require('resend');
      const resend = new Resend(resendKey);
      await resend.emails.send({
        from: 'RealSmile <noreply@realsmile.online>',
        to: email,
        subject: '🔑 Your Widget Pro key',
        html: `<!DOCTYPE html><html><head><meta charset="utf-8"></head>
<body style="background:#0a0a0a;color:#fff;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;padding:0">
  <div style="max-width:480px;margin:0 auto;padding:40px 24px">
    <h1 style="font-size:20px;font-weight:900;margin:0 0 16px;letter-spacing:-0.02em">Your Widget Pro key</h1>
    <div style="background:#111;border:1px solid #1f2937;border-radius:12px;padding:16px;margin-bottom:20px">
      <p style="font-size:11px;color:#6b7280;margin:0 0 8px;text-transform:uppercase;letter-spacing:0.1em">Widget Key</p>
      <code style="font-size:13px;color:#10b981;word-break:break-all">${found}</code>
    </div>
    <a href="https://realsmile.online/widget/pro" style="display:block;background:#fff;color:#000;text-align:center;padding:14px 24px;border-radius:50px;font-weight:900;font-size:14px;text-decoration:none;margin-bottom:12px">
      Open Widget Dashboard →
    </a>
    <p style="font-size:11px;color:#374151;text-align:center">Keep this key private · realsmile.online</p>
  </div>
</body></html>`,
      });
      console.log(`[WIDGET RECOVER] Sent key to ${email}`);
    } catch (err) {
      console.error('[WIDGET RECOVER ERROR]', err.message);
    }
  } else {
    console.log(`[WIDGET RECOVER] No key found for ${email}`);
  }

  res.json({ success: true }); // always success
});

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

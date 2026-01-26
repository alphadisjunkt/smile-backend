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

// Cache results for 1 hour
const cache = new NodeCache({ stdTTL: 3600, checkperiod: 120 });

// Track daily analyses - resets daily at midnight
let dailyCount = 24 // Start from your current count
let lastResetDate = new Date().toDateString()

// Reset counter daily
const checkDailyReset = () => {
  const today = new Date().toDateString()
  if (today !== lastResetDate) {
    dailyCount = 0
    lastResetDate = today
    console.log('📊 Daily counter reset')
  }
}

app.use(cors({
  origin: '*',
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type'],
  credentials: false
}));

app.options('*', cors());

const limiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 50,
  message: { error: 'Too many requests, please try again later.' },
  standardHeaders: true,
  legacyHeaders: false,
});

app.use('/analyze', limiter);
app.use(express.json({ limit: '10mb' }));

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

// NEW: Get daily count endpoint
app.get('/count', (req, res) => {
  checkDailyReset()
  res.json({ 
    count: dailyCount,
    date: lastResetDate
  })
})

// NEW: Increment count endpoint  
app.post('/count/increment', (req, res) => {
  checkDailyReset()
  dailyCount++
  console.log(`📊 Daily count: ${dailyCount}`)
  res.json({ 
    count: dailyCount,
    date: lastResetDate
  })
})

app.get('/', (req, res) => {
  checkDailyReset()
  res.json({ 
    status: 'ok', 
    message: 'RealSmile API Server',
    modelsLoaded,
    dailyAnalyses: dailyCount,
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
    
    cache.set(imageHash, result);
    console.log(`💾 Cached result for future requests`);
    
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
  console.log(`📊 Daily analyses: ${dailyCount}`);
});

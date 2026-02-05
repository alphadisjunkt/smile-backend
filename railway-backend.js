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

const cache = new NodeCache({ stdTTL: 3600, checkperiod: 120 });

let dailyCount = 450; // Starting baseline
let lastResetDate = new Date().toDateString();

const checkDailyReset = () => {
  const today = new Date().toDateString();
  if (today !== lastResetDate) {
    dailyCount = 450;
    lastResetDate = today;
    console.log('📊 Daily counter reset to baseline');
  }
};

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

app.use('/api/analyze', limiter);
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

function stable(value) {
  return Math.round(value * 10) / 10;
}

function calculateEyeConstriction(landmarks) {
  const leftEye = landmarks.getLeftEye();
  const rightEye = landmarks.getRightEye();
  
  const leftEyeWidth = stable(Math.abs(leftEye[3].x - leftEye[0].x));
  const leftEyeHeight = stable(Math.abs(leftEye[1].y - leftEye[5].y));
  const rightEyeWidth = stable(Math.abs(rightEye[3].x - rightEye[0].x));
  const rightEyeHeight = stable(Math.abs(rightEye[1].y - rightEye[5].y));
  
  const leftAspectRatio = leftEyeWidth / leftEyeHeight;
  const rightAspectRatio = rightEyeWidth / rightEyeHeight;
  const avgAspectRatio = stable((leftAspectRatio + rightAspectRatio) / 2);
  
  const eyeConstriction = Math.max(0, Math.min(1, (4 - avgAspectRatio) / 2));
  
  return Math.round(eyeConstriction * 100);
}

function calculateCheekRaise(landmarks) {
  const nose = landmarks.getNose();
  const leftEye = landmarks.getLeftEye();
  const rightEye = landmarks.getRightEye();
  const mouth = landmarks.getMouth();
  
  const noseTip = nose[3];
  const leftEyeBottom = leftEye[4];
  const rightEyeBottom = rightEye[4];
  const mouthTop = mouth[14];
  
  const eyeToNoseLeft = stable(Math.abs(leftEyeBottom.y - noseTip.y));
  const eyeToNoseRight = stable(Math.abs(rightEyeBottom.y - noseTip.y));
  const noseToMouth = stable(Math.abs(mouthTop.y - noseTip.y));
  
  const avgEyeToNose = stable((eyeToNoseLeft + eyeToNoseRight) / 2);
  const cheekRaise = Math.max(0, Math.min(1, avgEyeToNose / noseToMouth));
  
  return Math.round(cheekRaise * 100);
}

function calculateMouthCurve(landmarks) {
  const mouth = landmarks.getMouth();
  
  const leftCorner = mouth[0];
  const rightCorner = mouth[6];
  const topCenter = mouth[14];
  const bottomCenter = mouth[18];
  
  const mouthWidth = stable(Math.abs(rightCorner.x - leftCorner.x));
  const mouthHeight = stable(Math.abs(bottomCenter.y - topCenter.y));
  const cornerAvgY = stable((leftCorner.y + rightCorner.y) / 2);
  
  const curvature = (topCenter.y - cornerAvgY) / mouthHeight;
  const mouthCurve = Math.max(0, Math.min(1, curvature + 0.5));
  
  return Math.round(mouthCurve * 100);
}

function calculateSymmetry(landmarks) {
  const leftEye = landmarks.getLeftEye();
  const rightEye = landmarks.getRightEye();
  const mouth = landmarks.getMouth();
  
  const leftEyeCenter = leftEye.reduce((acc, point) => ({
    x: acc.x + point.x,
    y: acc.y + point.y
  }), { x: 0, y: 0 });
  leftEyeCenter.x = stable(leftEyeCenter.x / leftEye.length);
  leftEyeCenter.y = stable(leftEyeCenter.y / leftEye.length);
  
  const rightEyeCenter = rightEye.reduce((acc, point) => ({
    x: acc.x + point.x,
    y: acc.y + point.y
  }), { x: 0, y: 0 });
  rightEyeCenter.x = stable(rightEyeCenter.x / rightEye.length);
  rightEyeCenter.y = stable(rightEyeCenter.y / rightEye.length);
  
  const leftMouthCorner = mouth[0];
  const rightMouthCorner = mouth[6];
  
  const eyeYDiff = stable(Math.abs(leftEyeCenter.y - rightEyeCenter.y));
  const eyeXDiff = stable(Math.abs(rightEyeCenter.x - leftEyeCenter.x));
  const eyeSymmetry = 1 - (eyeYDiff / eyeXDiff);
  
  const mouthYDiff = stable(Math.abs(leftMouthCorner.y - rightMouthCorner.y));
  const mouthXDiff = stable(Math.abs(rightMouthCorner.x - leftMouthCorner.x));
  const mouthSymmetry = 1 - (mouthYDiff / mouthXDiff);
  
  const avgSymmetry = stable((eyeSymmetry + mouthSymmetry) / 2);
  return Math.round(Math.max(0, Math.min(1, avgSymmetry)) * 100);
}

function calculateLipCornerElevation(landmarks) {
  const mouth = landmarks.getMouth();
  const nose = landmarks.getNose();
  
  const leftCorner = mouth[0];
  const rightCorner = mouth[6];
  const upperLip = mouth[14];
  const noseTip = nose[3];
  
  const cornerAvgY = stable((leftCorner.y + rightCorner.y) / 2);
  const lipToNoseDistance = stable(Math.abs(upperLip.y - noseTip.y));
  const cornerToNoseDistance = stable(Math.abs(cornerAvgY - noseTip.y));
  
  const elevation = cornerToNoseDistance / lipToNoseDistance;
  return Math.round(Math.max(0, Math.min(1, elevation - 0.5)) * 100);
}

function calculateNoseLipDistance(landmarks) {
  const nose = landmarks.getNose();
  const mouth = landmarks.getMouth();
  
  const noseTip = nose[3];
  const upperLip = mouth[14];
  const noseBase = nose[6];
  
  const tipToLipDistance = stable(Math.abs(upperLip.y - noseTip.y));
  const noseHeight = stable(Math.abs(noseBase.y - noseTip.y));
  
  const compression = 1 - (tipToLipDistance / (noseHeight * 2));
  return Math.round(Math.max(0, Math.min(1, compression)) * 100);
}

function hashImage(base64Data) {
  return crypto.createHash('md5').update(base64Data).digest('hex');
}

app.get('/api/counter', (req, res) => {
  checkDailyReset();
  res.json({ 
    count: dailyCount,
    date: lastResetDate
  });
});

app.get('/', (req, res) => {
  checkDailyReset();
  res.json({ 
    status: 'ok', 
    message: 'RealSmile API Server v2.0',
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

app.post('/api/analyze', async (req, res) => {
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
    
    // IMPROVED: Lower threshold for better detection
    const detections = await faceapi
      .detectAllFaces(img, new faceapi.TinyFaceDetectorOptions({
        inputSize: 416,
        scoreThreshold: 0.15  // Much lower for better detection
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
      const box = detection.detection.box;
      
      // Calculate all 6 metrics
      const eyeConstriction = calculateEyeConstriction(landmarks);
      const cheekRaise = calculateCheekRaise(landmarks);
      const mouthCurve = calculateMouthCurve(landmarks);
      const symmetry = calculateSymmetry(landmarks);
      const lipCornerElevation = calculateLipCornerElevation(landmarks);
      const noseLipDistance = calculateNoseLipDistance(landmarks);
      
      const finalScore = Math.round(
        eyeConstriction * 0.40 +
        cheekRaise * 0.25 +
        mouthCurve * 0.15 +
        symmetry * 0.10 +
        lipCornerElevation * 0.05 +
        noseLipDistance * 0.05
      );
      
      const isGenuine = eyeConstriction > 60 && cheekRaise > 50;
      
      let verdict = '';
      if (finalScore >= 80) verdict = 'Excellent genuine Duchenne smile!';
      else if (finalScore >= 65) verdict = 'Good smile with genuine qualities';
      else if (finalScore >= 50) verdict = 'Moderate smile, could be more natural';
      else if (finalScore >= 35) verdict = 'Somewhat forced smile';
      else verdict = 'Appears to be a posed smile';
      
      return {
        score: finalScore,
        isGenuine,
        verdict,
        metrics: {
          eyeConstriction,
          cheekRaise,
          mouthCurve,
          symmetry,
          lipCornerElevation,
          noseLipDistance
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
    
    // Increment counter by number of people analyzed
    dailyCount += people.length;
    console.log(`📊 Daily count: ${dailyCount} (+${people.length})`);
    
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

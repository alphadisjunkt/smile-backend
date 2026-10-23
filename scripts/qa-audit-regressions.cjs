// Offline API/model/input regression checks; no native model, network or emails.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const quiet={log(){},warn(){},error(){}};
function png(width,height){const b=Buffer.alloc(33);Buffer.from('89504e470d0a1a0a','hex').copy(b);b.writeUInt32BE(13,8);b.write('IHDR',12);b.writeUInt32BE(width,16);b.writeUInt32BE(height,20);return b}
function jpeg(width,height){const b=Buffer.from([255,216,255,224,0,4,0,0,255,194,0,8,8,0,0,0,0,1]);b.writeUInt16BE(height,13);b.writeUInt16BE(width,15);return b}
async function backendTests(file){
 const imageInput=require(path.join(path.dirname(file),'image-input.js'));
 for(const b of [png(1280,960),jpeg(4032,3024)]){const r=imageInput.decodeRequestImage(b.toString('base64'),5*1024*1024);assert.ok(r.width>=32);assert.ok(r.height>=32)}
 assert.throws(()=>imageInput.decodeRequestImage(png(32768,32768).toString('base64'),1024),e=>e.statusCode===413);
 for(const value of [{},'',[],Buffer.alloc(23).toString('base64')])assert.throws(()=>imageInput.decodeRequestImage(value,1024),e=>e.statusCode===400);
 for(let n=0;n<40;n++)assert.doesNotThrow(()=>imageInput.dimensions(Buffer.alloc(n)));
 const webp=(chunk,width,height)=>{const b=Buffer.alloc(30);b.write('RIFF');b.write('WEBP',8);b.write(chunk,12);if(chunk==='VP8X'){b.writeUIntLE(width-1,24,3);b.writeUIntLE(height-1,27,3)}else if(chunk==='VP8L'){b[20]=0x2f;b.writeUInt32LE(((height-1)<<14)|(width-1),21)}else{Buffer.from('9d012a','hex').copy(b,23);b.writeUInt16LE(width,26);b.writeUInt16LE(height,28)}return b};
 for(const kind of ['VP8X','VP8L','VP8 '])assert.deepEqual(imageInput.dimensions(webp(kind,1280,960)),{width:1280,height:960});
 const animated=webp('VP8X',640,480);animated[20]=2;assert.throws(()=>imageInput.decodeRequestImage(animated.toString('base64'),1024),/Animated/);
 const animationChunk=Buffer.alloc(20);animationChunk.writeUInt32BE(8);animationChunk.write('acTL',4);assert.throws(()=>imageInput.decodeRequestImage(Buffer.concat([png(640,480),animationChunk]).toString('base64'),1024),/Animated/);
 const routes=new Map(),limitOptions=[],sent=[];let nativeDecodes=0,modelCalls=0,releaseCore,coreFails=false;
 const deferred=new Promise(resolve=>{releaseCore=resolve});
 const net=()=>({async loadFromUri(){modelCalls++;await deferred;if(coreFails)throw Error('fixture model failure')}});
 const api={env:{monkeyPatch(){}},nets:{tinyFaceDetector:net(),faceLandmark68Net:net(),faceExpressionNet:net(),faceRecognitionNet:net(),ageGenderNet:net()}};
 const app={use(){},set(){},options(){},get(p,...f){routes.set('GET '+p,f)},post(p,...f){routes.set('POST '+p,f)},patch(p,...f){routes.set('PATCH '+p,f)}};
 class Cache{constructor(){this.rows=new Map()}get(k){return this.rows.get(k)}set(k,v){this.rows.set(k,v)}del(k){this.rows.delete(k)}}
 const limiter=options=>{limitOptions.push(options);const counts=new Map();return(req,res,next)=>{const n=(counts.get(req.ip)||0)+1;counts.set(req.ip,n);return n>options.max?res.status(429).json({error:'limited'}):next()}};
 class Image{set src(value){nativeDecodes++}get width(){return 0}get height(){return 0}}
 const modules={dotenv:{config(){}},express:Object.assign(()=>app,{json:()=>()=>{}}),cors:()=>()=>{},'express-rate-limit':limiter,'node-cache':Cache,'@vladmandic/face-api':api,canvas:{Image},'@tensorflow/tfjs-node':{},crypto:require('node:crypto'),'./image-input':imageInput,fs:{existsSync:()=>false,writeFileSync(){},readFileSync(){throw Error('Unexpected read')}},resend:{Resend:class{emails={send:async payload=>{sent.push(payload);return{data:{id:'fixture'}}}}}}};
 const context={global:{},require:id=>{assert.ok(Object.hasOwn(modules,id),'Unexpected dependency '+id);return modules[id]},process:{env:{WIDGET_PROVISION_SECRET:'service-fixture',RESEND_API_KEY:'offline-fixture'},on(){}},console:quiet,Buffer,Map,Set,Date,Promise,setTimeout,clearTimeout};
 const source=fs.readFileSync(file,'utf8').split("console.log('🚀 Starting server...');")[0];vm.runInNewContext(source+'\nglobalThis.audit={loadModels,reset(){modelsLoaded=false}};',context);
 const a=context.audit.loadModels(),b=context.audit.loadModels();releaseCore();await Promise.all([a,b]);assert.equal(modelCalls,5,'concurrent initialization loads each network once');
 context.audit.reset();coreFails=true;const failures=await Promise.allSettled([context.audit.loadModels(),context.audit.loadModels()]);assert.ok(failures.every(r=>r.status==='rejected'));coreFails=false;await context.audit.loadModels();assert.equal(modelCalls,13,'failed startup can retry without duplicating concurrent work');
 async function call(method,route,body={},query={},auth='',ip='fixture'){
  let result,status=200;const req={body,query,headers:{authorization:auth},ip};const res={status(code){status=code;return this},json(value){result=value;return this}};const handlers=routes.get(method+' '+route);const next=async(index=0)=>handlers[index]?.(req,res,()=>next(index+1));await next();return{result,status};
 }
 for(const route of ['/analyze','/landmarks']){const r=await call('POST',route,{image:png(32768,32768).toString('base64')});assert.equal(r.status,413);assert.equal(nativeDecodes,0,'oversized raster must not reach native decoder');assert.equal((await call('POST',route,{image:{}})).status,400)}
 const auth='Bearer service-fixture';const key=(await call('POST','/widget-key-provision',{sessionId:'cs_fixture',customerId:'cus_fixture',email:'Owner@Example.invalid'},{},auth)).result.key;
 assert.equal('email' in (await call('GET','/widget-usage',{}, {key})).result,false);
 assert.equal((await call('PATCH','/widget-theme',{key,theme:{primary:'#fff'}})).status,401);
 assert.equal((await call('PATCH','/widget-theme',{key,theme:{primary:'#fff'}},{},auth)).status,200);
 await call('POST','/widget-recover',{email:' owner@example.invalid '},{},'','one');await call('POST','/widget-recover',{email:'owner@example.invalid'},{},'','two');assert.equal(sent.length,1,'recipient cooldown must survive changing IPs');
 for(let i=0;i<3;i++)assert.equal((await call('POST','/widget-recover',{email:'nobody'+i+'@example.invalid'},{},'','limit-ip')).status,200);
 assert.equal((await call('POST','/widget-recover',{email:'nobody4@example.invalid'},{},'','limit-ip')).status,429);
 assert.equal((await call('POST','/widget-recover',{email:{}},{},'','bad-ip')).status,400);
 assert.ok(limitOptions.some(x=>x.windowMs===3600000&&x.max===3));
 console.log('PASS backend: raster headers before decode, startup single flight, widget privacy/theme auth and recovery budgets');
}
module.exports=backendTests;
if(require.main===module)backendTests(path.resolve(process.argv[2]||path.join(__dirname,'../server.js'))).catch(error=>{console.error(error);process.exitCode=1});

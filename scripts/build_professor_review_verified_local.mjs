/** Build the focused four-slide review without the unrestricted-model comparison. */
import fs from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
process.on('uncaughtException',error=>{console.error(error.message);process.exit(1);});
const root=path.resolve(process.argv[2]);
const runtime=process.env.RUNTIME_NODE_MODULES;
if(!runtime)throw Error('Set RUNTIME_NODE_MODULES to the supplied presentation runtime.');
const requireRuntime=createRequire(path.join(runtime,'__runtime__.cjs'));
const {Presentation,PresentationFile}=await import(pathToFileURL(requireRuntime.resolve('@oai/artifact-tool')).href);
const data=JSON.parse(await fs.readFile(path.join(root,'slide_data.json'),'utf8'));
if(!data.unrestricted || data.extensions.length!==2 || data.features.length<3)
  throw Error('Build the professor deck only after both extensions, core feature screens and unrestricted model complete.');
const p=Presentation.create({slideSize:{width:1280,height:720}});
const C={ink:'#102b36',green:'#007f73',amber:'#c76a16',gray:'#64748b',pale:'#eef5f5',blue:'#2869a9'};
const font='Arial';
const pct=x=>(100*x).toFixed(2)+'%';
const short=x=>x.replace('feature_screen_','').replace('feature_full_','').replace('unrestricted_inception_resnet_v2','InceptionResNet-V2');
/** Use native text boxes for editable interpretation and small provenance footers. */
function text(slide,value,left,top,width,height,size=24,color=C.ink,bold=false){
  const shape=slide.shapes.add({geometry:'textbox',position:{left,top,width,height},fill:'none',line:{fill:'none',width:0}});
  shape.text=value;shape.text.style={typeface:font,fontSize:size,color,bold};return shape;
}
/** Apply the same quiet canvas and validation-population footer to every question. */
function slide(title,index){
  const s=p.slides.add();s.background.fill='#ffffff';text(s,title,48,30,1190,95,42,C.ink,true);
  text(s,index===4?'Payload benchmark: 2,056 recordings (1,697 train / 359 validation). No final-test data.':index===2?'Train 1,709 / validation 361 / held-out test 365. RPW positive. Shared recording allocation.':index===3?'Both routes: same held-out test recordings. Validation-selected thresholds. One daily report per node.':`Validation: ${data.count} recordings | ${data.rpw} RPW, ${data.healthy} healthy | shared recording split`,48,690,1150,18,12,C.gray);
  text(s,String(index),1210,678,25,22,16,C.gray);return s;
}
/** Keep chart data and axes native rather than flattening analytical evidence into an image. */
function chart(slide,type,options){
  const config={chartFill:'#ffffff',plotAreaFill:'#ffffff',
    hasLegend:true,legend:{position:'bottom',textStyle:{typeface:font,fontSize:16,fill:C.ink}},
    xAxis:{textStyle:{typeface:font,fontSize:14,fill:C.gray},line:{fill:'#ccd6dc',width:1}},
    yAxis:{textStyle:{typeface:font,fontSize:14,fill:C.gray},majorGridlines:{fill:'#e4e9ec',width:1}},...options};
  const fix=value=>{if(!value || typeof value!=='object')return; if('fontFamily' in value){value.typeface=value.fontFamily;delete value.fontFamily;} for(const [key,child] of Object.entries(value)){if(typeof child==='number' && Number.isFinite(child))value[key]=Number(child.toPrecision(12));else fix(child);}};
  for(const key of ['xAxis','yAxis'])if(typeof config[key]?.title==='string')config[key].title={text:config[key].title,textStyle:{typeface:font,fontSize:16,fill:C.gray}};
  fix(config);return slide.charts.add(type,config);
}
/** Native tables retain exact displayed values and remain editable in PowerPoint. */
function table(slide,values,left,top,width,height,columnWidths,size=21){
  const t=slide.tables.add({rows:values.length,columns:values[0].length,left,top,width,height,values,columnWidths});
  t.borders.assign({fill:'#d4dfe3',width:.7,style:'solid'});
  for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){
    const cell=t.getCell(r,c);cell.fill=r===0?C.ink:'#ffffff';
    cell.text.style={typeface:font,fontSize:size,color:r===0?'#ffffff':C.ink,bold:r===0};
  }return t;
}
/** Display true counts plus row normalization so RPW misses cannot disappear in overall accuracy. */
function confusion(slide,record,left,top,width,title){
  text(slide,title,left,top,width,35,26,C.ink,true);
  const m=record.confusion,values=[['Actual / predicted','healthy','RPW']];
  for(let i=0;i<2;i++){
    const sum=m[i][0]+m[i][1];values.push([i===0?'healthy':'RPW',...m[i].map(v=>`${v} (${(v/sum*100).toFixed(1)}%)`)]);
  }
  const matrix=table(slide,values,left,top+48,width,180,[width*.36,width*.32,width*.32],23);
  for(let r=1;r<3;r++)for(let c=1;c<3;c++)matrix.getCell(r,c).fill=r===c?'#e7f3f0':'#fff2e8';
}
// 2. Paired heatmaps are actual features from one source WAV; bars use controlled screening runs only.
{
 const s=slide('MFCC vs Mel vs PCEN: What Features Work Best?',1);
 text(s,`RPW training example: ${data.feature_example.duration_seconds.toFixed(2)} s recording, first 10 s shown, no padding`,55,126,1170,26,18,C.gray);
 const image=await fs.readFile(path.join(root,'plots/feature_heatmaps.png'));
 s.images.add({blob:image.buffer.slice(image.byteOffset,image.byteOffset+image.byteLength),contentType:'image/png',alt:'MFCC, log-Mel and PCEN features from the same audio recording',fit:'contain',position:{left:55,top:155,width:1170,height:205}});
 const features=[...data.features].sort((a,b)=>b.f1-a.f1);
 const trials=[...features,...data.feature_full].sort((a,b)=>b.f1-a.f1);
 chart(s,'bar',{position:{left:60,top:368,width:1160,height:215},categories:trials.map(r=>short(r.id)+(r.kind==='feature_full'?' full':'')),
   barOptions:{direction:'column',grouping:'clustered',gapWidth:110},
   series:[{name:'F1',values:trials.map(r=>r.f1),fill:C.green},{name:'RPW recall',values:trials.map(r=>r.recall),fill:C.blue}],
   yAxis:{min:0,max:1,numberFormatCode:'0%',textStyle:{typeface:font,fontSize:15}},
   dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.0%',textStyle:{typeface:font,fontSize:14,fill:C.ink}}});
 const baseline=features.find(r=>r.representation==='mfcc13'),best=features[0];
 text(s,`${short(best.id)} leads screening F1: ${pct(best.f1)} (${((best.f1-baseline.f1)*100).toFixed(2)} pp vs MFCC13)`,55,604,1180,45,27,C.green,true);
 text(s,'Screen: CNN16 / 60 epochs max; full follow-ups: CNN32 / 120 max',58,655,1170,22,14,C.gray);
 s.speakerNotes.text=`The screening CNN, optimizer, seed, recordings and ten-second audio coverage were fixed; full follow-ups use CNN32 and a 120-epoch ceiling. ${short(best.id)} achieved ${pct(best.f1)} F1 and ${pct(best.recall)} RPW recall. The unpadded heatmaps show the first ten seconds of a verified 20.46-second RPW training recording, with MFCC training scaling; sources: feature_comparison.csv, feature_example.json and plots/same_recording_features.npz.`;
}
// 3. A frozen operating point is checked once on reserved recordings, with no test tuning.
{
 const s=slide('Does 100% Validation Recall Hold Up?',2);
 const audit=data.pca_audit,m=audit.metrics;
 const values=[['Split','RPW recall','Accuracy','F1','False alarms','RPW misses']];
 for(const [name,label] of [['train','Train'],['val','Validation'],['test','Held-out test']]){
  const r=m[name];values.push([`${label} (${r.count})`,pct(r.recall),pct(r.accuracy),pct(r.f1),String(r.false_positives),String(r.false_negatives)]);
 }
 table(s,values,55,158,1170,220,[190,190,190,180,215,205],22);
 text(s,`Held-out test: ${m.test.confusion[1][1]}/${m.test.confusion[1][0]+m.test.confusion[1][1]} RPW detected, ${m.test.false_negatives} misses`,55,415,1170,48,31,C.green,true);
 text(s,`Train-only PCA16 retains ${pct(audit.pca_variance_explained)} of feature variance`,55,484,1170,38,26,C.ink);
 text(s,`Reconstruction MSE: train ${audit.reconstruction.train.encoded_mse.toFixed(3)}, test ${audit.reconstruction.test.encoded_mse.toFixed(3)}`,55,533,1170,36,25,C.ink);
 text(s,'PCA16 detected 92 of 96 RPW recordings on the held-out test,\nwith 22 false alarms among 269 healthy recordings.',55,590,1170,58,25,C.green,true);
 text(s,'Same fixed threshold. No duplicate audio across splits. New-tree generalization remains unverified.',55,650,1170,24,16,C.gray);
 s.speakerNotes.text=`We locked the model, PCA16 profile, normalization, windowing and validation-selected threshold ${audit.threshold.toFixed(6)} before reading the test set. Validation recall was 100%, but held-out recall was ${pct(m.test.recall)} with ${m.test.false_negatives} misses and ${m.test.false_positives} false alarms, while reconstruction errors remained similar. PCA refitting on training windows reproduced the saved profile and the train-validation F1 gap was ${(100*audit.train_validation_f1_gap).toFixed(2)} points, but recording-based splits share folders and do not establish independent-tree performance; source pca_audit_2026-10-19/audit_results.json and locked_finalist.json.`;
}
// 4. Explain the feature projection route and the separate local-classification route.
{
 const s=slide('PCA16 Features and Local INT8 Decisions',3);
 const audit=data.pca_audit;
 const local=data.local_test.test;
 text(s,'1. PCA16: classify on the server',55,139,1170,40,30,C.green,true);
 text(s,'PCA learns 16 directions from training MFCCs.\nESP32 projects 390 values into 16 one-byte codes per window.\nServer reconstructs MFCCs and takes the highest window RPW score.',55,191,1170,112,25);
 text(s,'2. INT8 residual CNN: classify on the ESP32',55,328,1170,40,30,C.blue,true);
 text(s,'ESP32 classifies each window locally and aggregates the decisions.\nThe node sends a proposed 24-byte daily summary.',55,380,1170,78,25);
 table(s,[['Route','Evidence','RPW recall','Transmission'],
   ['PCA16 server','Held-out test',pct(audit.metrics.test.recall),'16 B/window (~64 B/recording)'],
   ['Local INT8','Held-out test',pct(local.recall),'24 B/day (proposed summary)']],
   55,488,1170,125,[250,240,190,490],21);
 text(s,'PCA compresses features. The neural network makes the RPW decision.',55,629,1170,37,25,C.green,true);
 text(s,'Local INT8: 95/96 RPW detected, 25 false alarms/269 healthy. Headers extra. ESP32 board measurements pending.',55,666,1170,18,13,C.gray);
 s.speakerNotes.text=`PCA subtracts training-derived means and projects normalized 390-value MFCCs onto sixteen training-fitted principal directions, then encodes each coordinate in one byte and reverses the projection at the server. The server max-pools overlapping-window scores, while the separate ${data.window_deployment.find(r=>r.id==='local_raw_int8').model_bytes}-byte INT8 model can aggregate classifications on the ESP32 and send the proposed daily summary. PCA16's held-out result is ${pct(audit.metrics.test.recall)} recall with ${audit.metrics.test.false_positives} false alarms, and the actual local INT8 export detects 95 of 96 RPW recordings with 25 false alarms among 269 healthy recordings on the same held-out recording split, using its pre-existing validation-selected threshold without retraining or retuning; source local_int8_test_2026-10-19/audit_results.json and board RAM, latency and power remain unmeasured.`;
}
// 5. Savings compare identical five-window PCA codes with the same two-byte codec header.
{
 const s=slide('Lossless Delta Encoding: How Much Can We Save?',4);
 const d=data.delta_savings;
 text(s,'Five windows: 80 bytes of PCA codes + 2-byte codec header = 82-byte raw baseline',55,128,1170,35,22,C.gray);
 chart(s,'bar',{position:{left:45,top:181,width:750,height:270},categories:['Raw baseline','Best case','Average','Worst observed'],
   hasLegend:false,barOptions:{direction:'column',grouping:'clustered',gapWidth:110},
   series:[{name:'Encoded packet bytes',values:[d.baseline_bytes,d.best_bytes,d.mean_bytes,d.worst_observed_bytes],fill:C.blue}],
   yAxis:{min:0,max:100,title:'Bytes for the same five windows',majorUnit:20,textStyle:{typeface:font,fontSize:15}},
   dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.0" B"',textStyle:{typeface:font,fontSize:21,fill:C.ink}}});
 text(s,`${(100*d.fit_64_fraction).toFixed(1)}% fit within 64 bytes`,830,196,390,46,29,C.green,true);
 text(s,`${d.fit_64_count.toLocaleString('en-US')} of ${d.count.toLocaleString('en-US')} recordings\nFive windows, original 8-bit precision`,835,250,390,75,23,C.ink);
 text(s,`${d.four_window_fallback_count} oversized cases fitted\na four-window fallback`,835,336,390,65,23,C.gray);
 text(s,'Future data can still need 82 B\nwith raw fallback: 0% saving',835,412,390,50,18,C.gray);
 const values=[['Case','Output bytes','Bytes saved vs raw','Saving']];
 for(const [label,key] of [['Best observed','best'],['Average (mean)','mean'],['Worst observed','worst_observed']]){
  const bytes=d[key+'_bytes'];values.push([label,Number.isInteger(bytes)?String(bytes):bytes.toFixed(2),Number.isInteger(d[key+'_saving_bytes'])?String(d[key+'_saving_bytes']):d[key+'_saving_bytes'].toFixed(2),(100*d[key+'_saving_fraction']).toFixed(1)+'%']);
 }
 table(s,values,55,480,1170,132,[290,260,340,280],22);
 text(s,'Keep the first vector, encode later differences, reconstruct every PCA byte exactly.',55,630,1170,33,25,C.green,true);
 text(s,'Adaptive delta/Rice packing. Node/time/security/radio overhead extra. Five-window classifier accuracy not re-evaluated.',55,670,1170,19,13,C.gray);
 s.speakerNotes.text=`We store the first 16-byte PCA vector and encode the following window differences with adaptive Rice or escape codes, preserving all original eight-bit values. Across ${d.count} five-window recordings, output ranged from ${d.best_bytes} to ${d.worst_observed_bytes} bytes with a mean of ${d.mean_bytes.toFixed(2)}, saving ${(100*d.mean_saving_fraction).toFixed(1)}% against the same 82-byte raw packet including its two-byte codec header. ${d.fit_64_count} packets fit 64 bytes and ${d.four_window_fallback_count} needed a four-window fallback, while future data can require 82 bytes and five-window classifier accuracy remains untested, as recorded in delta_review_2026-10-19/per_recording.csv and delta_savings.json.`;
}
await fs.mkdir(path.join(root,'deck_previews'),{recursive:true});
for(const [i,s] of p.slides.items.entries()){
 const preview=await p.export({slide:s,format:'png',scale:1});
 await fs.writeFile(path.join(root,'deck_previews',`slide-${i+1}.png`),new Uint8Array(await preview.arrayBuffer()));
}
const stem='TreeVibes_Professor_Quick_Review_2026-10-20';
const candidate=path.join(root,'.deck-build/candidate.pptx');await fs.mkdir(path.dirname(candidate),{recursive:true});
await(await PresentationFile.exportPptx(p)).save(candidate);
// The matching PDF is exported from the finalized PPTX using the supplied LibreOffice runtime.
await fs.writeFile(path.join(root,'.deck-build/deck_manifest.json'),JSON.stringify({slides:4,source:'slide_data.json',cohort:data.cohort,chartsEditable:true},null,2));
console.log(JSON.stringify({candidate,previews:path.join(root,'deck_previews')}));

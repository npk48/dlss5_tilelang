import { chromium } from 'playwright';
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
const base=process.env.D5_TEST_URL || 'http://127.0.0.1:17864';
const token=process.env.D5_TEST_TOKEN || 'studio-smoke-token';
const out=path.resolve(process.env.D5_TEST_OUTPUT || 'test-results/dynamic');
await fs.mkdir(out,{recursive:true});
const browser=await chromium.launch({headless:true,...(process.env.PLAYWRIGHT_CHANNEL?{channel:process.env.PLAYWRIGHT_CHANNEL}:{})});
const page=await browser.newPage({viewport:{width:1440,height:1000}});
const headers={Authorization:`Bearer ${token}`};
const reports=[];const errors=[];page.on('pageerror',e=>errors.push(e.message));
async function json(url,options={}){const r=await page.request.get(base+url,{headers,...options});assert.ok(r.ok(),`${url}: ${r.status()}`);return r.json();}
async function terminal(id){for(let i=0;i<600;i++){const j=await json(`/v1/jobs/${id}`);if(j.status==='failed')throw Error(j.error);if(j.status==='completed')return j;await new Promise(r=>setTimeout(r,500));}throw Error(`Job ${id} timed out`);}
async function picture(width,height,shift=0){return Buffer.from(await page.evaluate(({width,height,shift})=>{
 const canvas=document.createElement('canvas');canvas.width=width;canvas.height=height;const context=canvas.getContext('2d'),im=context.createImageData(width,height);
 for(let y=0;y<height;y++)for(let x=0;x<width;x++){const i=(y*width+x)*4,xx=x-shift;im.data[i]=127+60*Math.sin(xx*.13)+50*Math.cos(y*.17);im.data[i+1]=127+60*Math.sin(xx*.071+y*.099)+45*Math.cos(xx*.21-y*.14);im.data[i+2]=127+64*Math.cos(xx*.11)*Math.sin(y*.19);im.data[i+3]=255;}
 context.putImageData(im,0,0);return canvas.toDataURL('image/png').split(',')[1];
 },{width,height,shift}),'base64');}
try{
 const info=await json('/v1/info');const modelFiles=info.available_models.slice().sort();
 for(const suffix of ['init','step'])assert.ok(modelFiles.includes(`native_guides/vda_small_dynamic_${suffix}.onnx`));
 assert.ok(!modelFiles.some(n=>/vda_small_\d+x\d+_/.test(n)),'Fixed-grid assets are still deployed');
 await page.goto(base);await page.getByRole('button',{name:'Disconnected',exact:true}).waitFor();await page.getByRole('button',{name:'Disconnected',exact:true}).click();
 await page.getByLabel('Bearer token').fill(token);await page.getByRole('button',{name:'Apply & reconnect'}).click();await page.getByRole('button',{name:'Connected',exact:true}).waitFor();
 // Defaults, not a selected fixed NR profile: output tracks each source exactly.
 const cases=[[640,360],[333,333],[321,257],[257,321],[641,287],[640,360]];
 for(const [width,height] of cases){
  const image=await picture(width,height);await page.locator('input[type=file]').setInputFiles({name:`${width}x${height}.png`,mimeType:'image/png',buffer:image});
  await page.getByRole('button',{name:'Full pipeline',exact:true}).click();
  assert.ok(await page.getByRole('button',{name:'Process image',exact:true}).isEnabled(),'Frontend still restricts input to model-size presets');
  const posted=page.waitForResponse(r=>r.request().method()==='POST'&&r.url().endsWith('/v1/jobs'));
  await page.getByRole('button',{name:'Process image',exact:true}).click();const accepted=await(await posted).json();
  const first=await terminal(accepted.id);assert.equal(first.metrics.width,width);assert.equal(first.metrics.height,height);assert.equal(first.metrics.frame,0);assert.equal(first.metrics.history_reset,1);
  await page.getByAltText('Processed result',{exact:true}).waitFor({timeout:15000});await page.waitForFunction(()=>document.querySelector('img[alt="Processed result"]')?.naturalWidth>0);
  const png=await(await page.request.get(base+first.result,{headers})).body();assert.equal(png.readUInt32BE(16),width);assert.equal(png.readUInt32BE(20),height);
  await fs.writeFile(path.join(out,`${width}x${height}.png`),png);
  const submitted=await page.request.post(base+'/v1/jobs',{headers,multipart:{image:{name:'next.png',mimeType:'image/png',buffer:await picture(width,height,1.75)},config:JSON.stringify({...first.config,reset:false})}});
  assert.equal(submitted.status(),202);const second=await terminal((await submitted.json()).id);assert.equal(second.metrics.frame,1);assert.equal(second.metrics.history_reset,0);
  assert.deepEqual((await json('/v1/info')).available_models.slice().sort(),modelFiles,'Per-size assets appeared while processing');
  reports.push({width,height,first,second});console.log(JSON.stringify({width,height,result:'PASS',first_setup_seconds:first.metrics.setup_seconds,step_seconds:second.metrics.process_seconds_including_upload_download}));
 }
 assert.deepEqual(errors,[]);await page.screenshot({path:path.join(out,'studio-dynamic.png'),fullPage:true});
 await fs.writeFile(path.join(out,'report.json'),JSON.stringify({modelFiles,cases:reports,checks:['one dynamic VDA pair','square/wide/portrait/odd/default source sizes','same SDK session across size changes','source-sized PNG output','init and causal step','history reset on geometry change','return to prior geometry','real frontend Full pipeline','no browser exceptions']},null,2));
} catch(e){await page.screenshot({path:path.join(out,'failure.png'),fullPage:true}).catch(()=>{});throw e;} finally{await browser.close();}

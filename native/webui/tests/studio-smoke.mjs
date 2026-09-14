// Real native server only: no mocked requests and no fake job progress.
// Start dlss5_server with --token matching D5_TEST_TOKEN before this test.
import { chromium } from 'playwright';
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
const base = process.env.D5_TEST_URL || 'http://127.0.0.1:17864';
const token = process.env.D5_TEST_TOKEN || 'studio-smoke-token';
const out = path.resolve(process.env.D5_TEST_OUTPUT || 'test-results/studio');
await fs.mkdir(out, { recursive: true });
const browser = await chromium.launch({ headless: true, ...(process.env.PLAYWRIGHT_CHANNEL ? { channel: process.env.PLAYWRIGHT_CHANNEL } : {}) });
const page = await browser.newPage({ viewport: { width: 1440, height: 1100 }, acceptDownloads: true });
const errors = [];
page.on('pageerror', e => errors.push(e.message));
try {
  assert.equal((await page.request.get(`${base}/v1/jobs`)).status(), 401);
  assert.equal((await page.goto(base)).status(), 200);
  await page.getByRole('button', { name: 'Disconnected', exact: true }).waitFor({ timeout: 15000 });
  await page.getByRole('button', { name: 'Disconnected', exact: true }).click();
  await page.getByLabel('Bearer token').fill(token);
  await page.getByRole('button', { name: 'Apply & reconnect' }).click();
  await page.getByRole('button', { name: 'Connected', exact: true }).waitFor({ timeout: 15000 });
  const encoded = await page.evaluate(() => {
    const c = document.createElement('canvas'); c.width = 640; c.height = 360;
    const ctx = c.getContext('2d'); const pixels = ctx.createImageData(640, 360);
    for (let y=0; y<360; y++) for (let x=0; x<640; x++) {
      const i=(y*640+x)*4; pixels.data[i]=x*255/639; pixels.data[i+1]=y*255/359;
      pixels.data[i+2]=80+35*Math.sin(x*.07); pixels.data[i+3]=255;
    }
    ctx.putImageData(pixels, 0, 0); return c.toDataURL('image/png').split(',')[1];
  });
  await page.locator('input[type=file]').setInputFiles({ name:'studio-smoke.png', mimeType:'image/png', buffer:Buffer.from(encoded,'base64') });
  await page.getByRole('button', { name: 'Full pipeline', exact:true }).click();
  await page.getByRole('button', { name: 'Process image', exact:true }).click();
  await page.waitForFunction(() => document.querySelector('.job-row.selected .status.completed, .job-row.selected .status.failed'), null, { timeout:240000 });
  const snapshot=await (await page.request.get(`${base}/v1/jobs`,{headers:{Authorization:`Bearer ${token}`}})).json();
  if(snapshot.jobs[0]?.status==='failed')throw new Error(snapshot.jobs[0].error || 'Native processing failed');
  await page.getByAltText('Processed result', { exact:true }).waitFor();
  await page.waitForFunction(() => document.querySelector('img[alt="Processed result"]')?.naturalWidth > 0);
  const listed = await (await page.request.get(`${base}/v1/jobs`, { headers:{ Authorization:`Bearer ${token}` } })).json();
  const job = listed.jobs[0]; assert.equal(job.status,'completed'); assert.equal(job.metrics.enabled_stages,47);
  assert.equal(job.config.depth,true); assert.equal(job.config.reconstruction,true);
  assert.ok(await page.getByAltText('Original image', { exact:true }).isVisible());
  await page.screenshot({ path:path.join(out,'desktop.png'), fullPage:true });
  const pngPromise = page.waitForEvent('download');
  await page.getByRole('link', { name:'Download PNG', exact:true }).click();
  const pngDownload = await pngPromise; await pngDownload.saveAs(path.join(out,'result.png'));
  assert.equal((await fs.readFile(path.join(out,'result.png'))).subarray(0,8).toString('hex'),'89504e470d0a1a0a');
  const cfgPromise = page.waitForEvent('download');
  await page.getByRole('button', { name:'Config JSON', exact:true }).click();
  await (await cfgPromise).saveAs(path.join(out,'config.json'));
  assert.equal(JSON.parse(await fs.readFile(path.join(out,'config.json'),'utf8')).reconstruction,true);
  await page.getByRole('button', { name:'Wipe comparison', exact:true }).click();
  const slider=page.getByLabel('Comparison divider', { exact:true }); const before=await slider.inputValue();
  await slider.focus(); await slider.press('ArrowLeft'); assert.notEqual(await slider.inputValue(),before);
  await page.getByLabel('Image zoom', { exact:true }).selectOption('1');
  await page.screenshot({ path:path.join(out,'comparison.png'), fullPage:true });
  await page.setViewportSize({ width:390, height:844 });
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth+2), 'Mobile page overflows horizontally');
  await page.screenshot({ path:path.join(out,'mobile.png'), fullPage:true });
  // Pixel correctness as well as HTTP/UI success: disabled processing is identity.
  const auth={Authorization:`Bearer ${token}`};
  const identitySubmission=await page.request.post(`${base}/v1/jobs`,{headers:auth,multipart:{image:{name:'identity.png',mimeType:'image/png',buffer:Buffer.from(encoded,'base64')},config:JSON.stringify({nr:false,mix:0})}});
  assert.equal(identitySubmission.status(),202);const identityId=(await identitySubmission.json()).id;
  let identity;for(let i=0;i<60;i++){identity=await(await page.request.get(`${base}/v1/jobs/${identityId}`,{headers:auth})).json();if(identity.status==='completed'||identity.status==='failed')break;await new Promise(r=>setTimeout(r,250));}
  assert.equal(identity.status,'completed',identity.error);
  const identityPng=await(await page.request.get(`${base}${identity.result}`,{headers:auth})).body();
  const maxPixelError=await page.evaluate(async ([src,dst])=>{
    async function pixels(value){const image=new Image();image.src='data:image/png;base64,'+value;await image.decode();const canvas=document.createElement('canvas');canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;const context=canvas.getContext('2d');context.drawImage(image,0,0);return context.getImageData(0,0,canvas.width,canvas.height).data;}
    const a=await pixels(src),b=await pixels(dst);if(a.length!==b.length)return Infinity;let worst=0;for(let i=0;i<a.length;i++)worst=Math.max(worst,Math.abs(a[i]-b[i]));return worst;
  },[encoded,identityPng.toString('base64')]);
  assert.equal(maxPixelError,0,'Identity processing changed RGB/alpha pixels');
  assert.deepEqual(errors,[]);
  await fs.writeFile(path.join(out,'report.json'),JSON.stringify({ base,job,checks:['public static assets','protected APIs','token connection','upload','full GPU pipeline','source/result pairing','PNG download','effective config download','wipe slider','zoom','mobile layout','no browser exceptions','HTTP PNG identity is pixel-exact'] },null,2));
  console.log(JSON.stringify({ result:'PASS',job:job.id,output:out,metrics:job.metrics }));
} catch(e) { await page.screenshot({path:path.join(out,'failure.png'),fullPage:true}).catch(()=>{}); throw e; } finally { await browser.close(); }

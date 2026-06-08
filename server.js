// ═══════════════════════════════════════════════════════════════
//  ZORA — Open-source AI Chatbot
//  Chat (text) + Image generation (via Free.ai)
// ═══════════════════════════════════════════════════════════════
const express = require('express');
const path = require('path');
const http = require('http');
require('dotenv').config({ path: path.join(__dirname, '.env') });

const app = express();
const PORT = process.env.PORT || 3003;

// ── Configuration ──────────────────────────────────────────────
const CFG = {
  ollamaBase: process.env.OLLAMA_BASE || 'http://localhost:11434',
  groqKey: process.env.GROQ_API_KEY || '',
  deepseekKey: process.env.DEEPSEEK_API_KEY || '',
  glmKey: process.env.GLM_API_KEY || '',
  geminiKey: process.env.GEMINI_API_KEY || '',
  provider: 'none',
  model: '',
};

// ── Available models ────────────────────────────────────────────
const MODELS = {
  ollama: [
    { id: 'deepseek-r1:14b',  name: 'DeepSeek R1 14B',   desc: 'Reasoning model, excellent at complex tasks' },
    { id: 'llama3.2',         name: 'Llama 3.2 3B',       desc: 'Fast, efficient, great for general chat' },
    { id: 'llama3.1',         name: 'Llama 3.1 8B',       desc: 'Balanced quality and speed' },
    { id: 'mistral',          name: 'Mistral 7B',          desc: 'Fast European model, strong reasoning' },
    { id: 'mixtral',          name: 'Mixtral 8x7B',        desc: 'MoE architecture, high quality' },
    { id: 'codellama',        name: 'Code Llama',          desc: 'Specialized for code generation' },
  ],
  groq: [
    { id: 'llama-3.3-70b-versatile', name: 'Llama 3.3 70B',    desc: 'Most powerful, versatile (free tier)' },
    { id: 'llama-3.1-8b-instant',    name: 'Llama 3.1 8B',     desc: 'Fast, instant responses' },
    { id: 'qwen/qwen3-32b',          name: 'Qwen 3 32B',       desc: 'Strong 32B model, high quality' },
    { id: 'meta-llama/llama-4-scout-17b-16e-instruct', name: 'Llama 4 Scout 17B', desc: 'MoE, efficient' },
    { id: 'allam-2-7b',              name: 'Allam 2 7B',       desc: 'Arabic + English, efficient' },
  ],
  deepseek: [
    { id: 'deepseek-v4-flash', name: 'DeepSeek V4 Flash',   desc: '284B MoE, 1M ctx, ultra-cheap, free credits' },
  ],
  glm: [
    { id: 'GLM-5.1',          name: 'GLM 5.1',             desc: 'Top-tier reasoning, Arena 1449, premium' },
    { id: 'GLM-5',            name: 'GLM 5',               desc: 'Flagship, MIT licensed, open-source' },
    { id: 'GLM-4.7',          name: 'GLM 4.7',             desc: 'Best balance of quality & cost' },
    { id: 'GLM-4.7-Flash',    name: 'GLM 4.7 Flash',       desc: 'Free tier, fast, efficient' },
  ],
  gemini: [
    { id: 'gemini-2.5-flash', name: 'Gemini 2.5 Flash',    desc: '1M ctx, free tier, fast, multimodal' },
    { id: 'gemini-2.5-flash-lite', name: 'Gemini 2.5 Flash Lite', desc: 'Faster, lighter, free tier' },
    { id: 'gemini-2.5-pro',   name: 'Gemini 2.5 Pro',      desc: '2M ctx, most powerful Gemini' },
    { id: 'gemini-3-flash',   name: 'Gemini 3 Flash',      desc: 'Latest Gemini, preview, bleeding edge' },
  ],
};

// ═══════════════════════════════════════════════════════════════
// AUTO-DETECT PROVIDER
// ═══════════════════════════════════════════════════════════════
async function detectProvider() {
  const requested = process.env.AI_PROVIDER || 'auto';

  // If user explicitly set a provider, try it
  if (requested !== 'auto') {
    if (requested === 'ollama') {
      const ok = await checkOllama();
      if (ok) { CFG.provider = 'ollama'; CFG.model = 'deepseek-r1:14b'; return; }
      console.warn('  ⚠  Ollama not running. Fallback needed.');
    }
    if (requested === 'groq' && CFG.groqKey) { CFG.provider = 'groq'; CFG.model = 'llama3-70b-8192'; return; }
    if (requested === 'deepseek' && CFG.deepseekKey) { CFG.provider = 'deepseek'; CFG.model = 'deepseek-chat'; return; }
    console.warn('  ⚠  Requested provider not available.');
  }

  // Try Groq first (cloud, fast, intelligent)
  if (CFG.groqKey) {
    CFG.provider = 'groq';
    CFG.model = 'llama-3.3-70b-versatile';
    console.log('  ✓  Using: Groq cloud (Llama 3.3 70B)');
    return;
  }

  // Try DeepSeek (free 5M tokens, V4 Flash)
  if (CFG.deepseekKey) {
    CFG.provider = 'deepseek';
    CFG.model = 'deepseek-v4-flash';
    console.log('  ✓  Using: DeepSeek V4 Flash');
    return;
  }

  // Try GLM (free 5M tokens + free Flash model)
  if (CFG.glmKey) {
    CFG.provider = 'glm';
    CFG.model = 'GLM-4.7-Flash';
    console.log('  ✓  Using: GLM (Zhipu AI)');
    return;
  }

  // Try Gemini (truly free, no credit card)
  if (CFG.geminiKey) {
    CFG.provider = 'gemini';
    CFG.model = 'gemini-2.5-flash';
    console.log('  ✓  Using: Google Gemini');
    return;
  }

  // Fallback: try Ollama (free, local)
  const ollamaModels = await getOllamaModels();
  if (ollamaModels.length > 0) {
    CFG.provider = 'ollama';
    CFG.model = ollamaModels[0];
    console.log(`  ✓  Detected: Ollama → ${CFG.model}`);
    return;
  }

  // Nothing available
  CFG.provider = 'none';
  console.log('  ✗  No AI provider detected. See setup instructions.');
}

async function getOllamaModels() {
  try {
    const res = await fetch(`${CFG.ollamaBase}/api/tags`, { signal: AbortSignal.timeout(2000) });
    if (!res.ok) return [];
    const data = await res.json();
    return (data.models || []).map(m => m.name.replace(':latest', ''));
  } catch {
    return [];
  }
}

async function checkOllama() {
  const models = await getOllamaModels();
  return models.length > 0;
}

// ═══════════════════════════════════════════════════════════════
// MIDDLEWARE
// ═══════════════════════════════════════════════════════════════
app.use(express.json({ limit: '1mb' }));

app.use((req, res, next) => {
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'DENY');
  res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');
  next();
});

app.use((req, res, next) => {
  const start = Date.now();
  res.on('finish', () => {
    const ms = Date.now() - start;
    if (req.path.startsWith('/api')) console.log(`  ${req.method} ${req.path} → ${res.statusCode} (${ms}ms)`);
  });
  next();
});

app.use(express.static(path.join(__dirname, 'public'), {
  maxAge: '1h',
  setHeaders: (res, fp) => { if (fp.endsWith('.html')) res.setHeader('Cache-Control', 'no-cache'); },
}));

// ═══════════════════════════════════════════════════════════════
// API: Status — returns provider info
// ═══════════════════════════════════════════════════════════════
app.get('/api/status', async (req, res) => {
  let models = [];

  // Groq models (always shown if key is configured)
  if (CFG.groqKey) models.push(...MODELS.groq);

  // DeepSeek models
  if (CFG.deepseekKey) models.push(...MODELS.deepseek);

  // GLM models (Zhipu AI)
  if (CFG.glmKey) models.push(...MODELS.glm);

  // Gemini models
  if (CFG.geminiKey) models.push(...MODELS.gemini);

  // Ollama models (fetch actual available)
  const ollamaModels = await getOllamaModels();
  if (ollamaModels.length > 0) {
    models.push(...ollamaModels.map(id => ({ id, name: id, desc: 'Local Ollama model' })));
  }

  // If no models at all, show defaults
  if (models.length === 0) models.push(...MODELS.ollama);

  res.json({
    name: 'Zora',
    provider: CFG.provider,
    model: CFG.model,
    models,
    setup: !CFG.groqKey && !CFG.deepseekKey && CFG.provider === 'none'
      ? { message: 'No AI provider configured.', steps: ['Get a free Groq API key at https://console.groq.com', 'Or get DeepSeek free 5M tokens at https://platform.deepseek.com', 'Or get GLM free credits at https://open.bigmodel.cn', 'Or get free Gemini key at https://aistudio.google.com/apikey', 'Or install Ollama: https://ollama.ai → ollama pull llama3.2'] }
      : null,
  });
});

// ═══════════════════════════════════════════════════════════════
// API: Chat (streaming via SSE)
// ═══════════════════════════════════════════════════════════════
app.post('/api/chat', async (req, res) => {
  const { messages, model } = req.body;

  if (!messages || !Array.isArray(messages) || messages.length === 0) {
    return res.status(400).json({ error: 'messages array is required' });
  }

  if (CFG.provider === 'none') {
    return res.status(400).json({
      error: 'No AI provider configured.\n\n→ Install Ollama: https://ollama.ai\n  Then run: ollama pull deepseek-r1:14b\n\n→ Or get a free Groq API key:\n  1. Sign up at https://console.groq.com\n  2. Create key at https://console.groq.com/keys\n  3. Add to .env: GROQ_API_KEY=gsk_your_key'
    });
  }

  // Set up SSE
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');
  res.flushHeaders();

  const sendEvent = (type, data) => {
    try { res.write(`event: ${type}\ndata: ${JSON.stringify(data)}\n\n`); } catch {}
  };

  try {
    // Route by model ID to determine which provider to use
    const modelId = model || CFG.model;
    const groqIds = MODELS.groq.map(m => m.id);
    const deepseekIds = MODELS.deepseek.map(m => m.id);
    const glmIds = MODELS.glm.map(m => m.id);
    const geminiIds = MODELS.gemini.map(m => m.id);
    if (groqIds.includes(modelId) && CFG.groqKey) await streamGroq(messages, modelId, sendEvent, res);
    else if (deepseekIds.includes(modelId) && CFG.deepseekKey) await streamOpenAI(messages, modelId, sendEvent, res, 'https://api.deepseek.com/v1', CFG.deepseekKey);
    else if (glmIds.includes(modelId) && CFG.glmKey) await streamOpenAI(messages, modelId, sendEvent, res, 'https://open.bigmodel.cn/api/paas/v4', CFG.glmKey);
    else if (geminiIds.includes(modelId) && CFG.geminiKey) await streamOpenAI(messages, modelId, sendEvent, res, 'https://generativelanguage.googleapis.com/v1beta/openai', CFG.geminiKey);
    else if (CFG.groqKey) await streamGroq(messages, modelId, sendEvent, res);
    else await streamOllama(messages, modelId, sendEvent, res);
  } catch (err) {
    console.error('  ✗ Stream error:', err.message);
    if (res.headersSent) {
      sendEvent('error', { message: err.message || 'Stream failed' });
      sendEvent('done', {});
    } else {
      res.status(500).json({ error: err.message || 'Internal error' });
    }
  }
});

// ═══════════════════════════════════════════════════════════════
// STREAM: Ollama (local)
// ═══════════════════════════════════════════════════════════════
async function streamOllama(messages, model, sendEvent, res) {
  const modelName = model || CFG.model || 'llama3.2';
  const response = await fetch(`${CFG.ollamaBase}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: modelName,
      messages: messages.map(m => ({ role: m.role, content: m.content })),
      stream: true,
    }),
    signal: AbortSignal.timeout(120000),
  });

  if (!response.ok) {
    const text = await response.text().catch(() => '');
    sendEvent('error', { message: `Ollama error (${response.status}). Make sure "${modelName}" is pulled: ollama pull ${modelName}` });
    sendEvent('done', {});
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '', fullContent = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const parsed = JSON.parse(line);
          if (parsed.message?.content) {
            fullContent += parsed.message.content;
            sendEvent('chunk', { content: parsed.message.content });
          }
        } catch {}
      }
    }
    sendEvent('done', { fullContent });
  } catch (err) {
    sendEvent('error', { message: 'Connection interrupted' });
    sendEvent('done', { fullContent });
  }
}

// ═══════════════════════════════════════════════════════════════
// STREAM: Groq (cloud, free tier)
// ═══════════════════════════════════════════════════════════════
async function streamGroq(messages, model, sendEvent, res) {
  const modelName = model || CFG.model || 'llama-3.3-70b-versatile';
  const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${CFG.groqKey}`,
    },
    body: JSON.stringify({
      model: modelName,
      messages: messages.map(m => ({ role: m.role, content: m.content })),
      stream: true,
      temperature: 0.7,
      max_tokens: 4096,
    }),
    signal: AbortSignal.timeout(60000),
  });

  if (!response.ok) {
    const errBody = await response.text().catch(() => '');
    let msg = `Groq API error (${response.status})`;
    try { const p = JSON.parse(errBody); msg = p.error?.message || msg; } catch {}
    sendEvent('error', { message: msg });
    sendEvent('done', {});
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '', fullContent = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        const t = line.trim();
        if (!t || !t.startsWith('data: ')) continue;
        const data = t.slice(6);
        if (data === '[DONE]') continue;
        try {
          const parsed = JSON.parse(data);
          const content = parsed.choices?.[0]?.delta?.content || '';
          if (content) { fullContent += content; sendEvent('chunk', { content }); }
        } catch {}
      }
    }
    sendEvent('done', { fullContent });
  } catch (err) {
    sendEvent('error', { message: 'Connection interrupted' });
    sendEvent('done', { fullContent });
  }
}

// ═══════════════════════════════════════════════════════════════
// STREAM: OpenAI-compatible (DeepSeek, GLM, Gemini)
// ═══════════════════════════════════════════════════════════════
async function streamOpenAI(messages, model, sendEvent, res, baseUrl, apiKey) {
  const modelName = model || CFG.model || 'deepseek-v4-flash';
  const response = await fetch(`${baseUrl}/chat/completions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: modelName,
      messages: messages.map(m => ({ role: m.role, content: m.content })),
      stream: true,
      temperature: 0.7,
      max_tokens: 4096,
    }),
    signal: AbortSignal.timeout(60000),
  });

  if (!response.ok) {
    const errBody = await response.text().catch(() => '');
    let msg = `API error (${response.status})`;
    try { const p = JSON.parse(errBody); msg = p.error?.message || msg; } catch {}
    sendEvent('error', { message: msg });
    sendEvent('done', {});
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '', fullContent = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        const t = line.trim();
        if (!t || !t.startsWith('data: ')) continue;
        const data = t.slice(6);
        if (data === '[DONE]') continue;
        try {
          const parsed = JSON.parse(data);
          const content = parsed.choices?.[0]?.delta?.content || '';
          if (content) { fullContent += content; sendEvent('chunk', { content }); }
        } catch {}
      }
    }
    sendEvent('done', { fullContent });
  } catch (err) {
    sendEvent('error', { message: 'Connection interrupted' });
    sendEvent('done', { fullContent });
  }
}

// ═══════════════════════════════════════════════════════════════
// Error handler
// ═══════════════════════════════════════════════════════════════
app.use((err, req, res, next) => {
  console.error('  ✗ Unhandled:', err.message);
  if (!res.headersSent) res.status(500).json({ error: 'Internal server error' });
});

// ═══════════════════════════════════════════════════════════════
// START
// ═══════════════════════════════════════════════════════════════
async function start() {
  console.log('');
  console.log('  ╔══════════════════════════════════════╗');
  console.log('  ║   ◆  ZORA — AI Chatbot              ║');
  console.log('  ║   Open-source · Private · Free       ║');
  console.log('  ╚══════════════════════════════════════╝');
  console.log('');
  console.log('  🔍  Detecting AI provider...');
  await detectProvider();
  console.log('');

  if (CFG.provider === 'none') {
    console.log('  ─────────────────────────────────────────');
    console.log('  ⚠  No AI provider detected.');
    console.log('     To get started:');
    console.log('     1. Install Ollama: https://ollama.ai');
    console.log('     2. Run: ollama pull llama3.2');
    console.log('     3. Keep Ollama running');
    console.log('     ─ or ─');
    console.log('     4. Get free Groq API key:');
    console.log('        https://console.groq.com');
    console.log('     5. Add to .env: GROQ_API_KEY=gsk_...');
    console.log('  ─────────────────────────────────────────');
    console.log('');
  }

  app.listen(PORT, () => {
    console.log(`  🚀  http://localhost:${PORT}`);
    console.log('');
  });
}

start();

// Standard Vertex AI REST caller with multi-hop continuation support.
// Usage: node call.mjs <request.json> <response.json>
// Authentication: standard Google Application Default Credentials / GOOGLE_APPLICATION_CREDENTIALS,
// plus GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION (default: "global").
import { existsSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const defaultNodeModules = resolve(__dirname, "../../viscomposer/pi/node_modules");
const nodeModulesDir = process.env.PI_NODE_MODULES || defaultNodeModules;

let GoogleAuth;
try {
  const reqFromPi = createRequire(nodeModulesDir + "/");
  ({ GoogleAuth } = reqFromPi("google-auth-library"));
} catch {
  const reqLocal = createRequire(import.meta.url);
  ({ GoogleAuth } = reqLocal("google-auth-library"));
}

const [requestPath, responsePath] = process.argv.slice(2);
if (!requestPath || !responsePath) {
  console.error("Usage: node call.mjs <request.json> <response.json>");
  process.exit(2);
}

const req = JSON.parse(readFileSync(requestPath, "utf8"));
const model = process.env.MODEL || "gemini-4-argon";
const project = process.env.GOOGLE_CLOUD_PROJECT;
if (!project) {
  console.error("GOOGLE_CLOUD_PROJECT is required for Vertex AI API calls.");
  process.exit(2);
}
const location = process.env.GOOGLE_CLOUD_LOCATION || "global";
const host = location === "global" ? "aiplatform.googleapis.com" : `${location}-aiplatform.googleapis.com`;
const url = `https://${host}/v1beta1/projects/${project}/locations/${location}/publishers/google/models/${model}:streamGenerateContent?alt=sse`;
const auth = new GoogleAuth({ scopes: ["https://www.googleapis.com/auth/cloud-platform"] });
const MAX_HOPS = Number(process.env.MAX_HOPS || 16);
const TOKEN_CACHE_PATH = `/tmp/.vasp_vx_tok_${project}.json`;

async function getBearerToken() {
  try {
    if (existsSync(TOKEN_CACHE_PATH)) {
      const cached = JSON.parse(readFileSync(TOKEN_CACHE_PATH, "utf8"));
      if (cached.token && cached.expiresAt && Date.now() + 300_000 < cached.expiresAt) {
        return cached.token;
      }
    }
  } catch {}
  const client = await auth.getClient();
  const res = await client.getAccessToken();
  const tok = typeof res === "string" ? res : res?.token;
  const exp = client.credentials?.expiry_date || (Date.now() + 3300_000);
  try {
    const tmp = `${TOKEN_CACHE_PATH}.${process.pid}.tmp`;
    writeFileSync(tmp, JSON.stringify({ token: tok, expiresAt: exp }), { mode: 0o600 });
    renameSync(tmp, TOKEN_CACHE_PATH);
  } catch {}
  return tok;
}

const parts = [{ text: req.user }];
if (Array.isArray(req.images)) {
  for (const img of req.images) {
    if (img && img.data_b64) {
      parts.push({ inlineData: { mimeType: img.mime_type || "image/png", data: img.data_b64 } });
    }
  }
} else if (req.image_b64) {
  parts.push({ inlineData: { mimeType: req.mime_type || "image/png", data: req.image_b64 } });
}

const generationConfig = {
  maxOutputTokens: Number(process.env.MAX_OUT || 65536),
  responseMimeType: "application/json",
};
if (process.env.THINKING) {
  generationConfig.thinkingConfig = { thinkingLevel: process.env.THINKING };
}
if (req.schema) {
  generationConfig.responseJsonSchema = req.schema;
}

const body = {
  contents: [{ role: "user", parts }],
  generationConfig,
};
if (req.system) {
  body.systemInstruction = { parts: [{ text: req.system }] };
}

async function hop(token) {
  const headers = { Authorization: `Bearer ${await getBearerToken()}`, "Content-Type": "application/json" };
  const res = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify(token ? { ...body, continuationToken: token } : body),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${(await res.text()).slice(0, 2000)}`);
  let text = "", usage = null, finish = null, next = null, buffer = "";
  const decoder = new TextDecoder();
  const take = (line) => {
    if (!line.startsWith("data:")) return;
    const chunk = JSON.parse(line.slice(5));
    const cand = chunk.candidates?.[0];
    for (const p of cand?.content?.parts ?? []) if (p.text && !p.thought) text += p.text;
    if (cand?.finishReason) finish = cand.finishReason;
    next = chunk.continuationToken ?? cand?.continuationToken ?? next;
    if (chunk.usageMetadata) usage = chunk.usageMetadata;
  };
  for await (const bytes of res.body) {
    buffer += decoder.decode(bytes, { stream: true });
    let i;
    while ((i = buffer.indexOf("\n")) >= 0) {
      take(buffer.slice(0, i).trim());
      buffer = buffer.slice(i + 1);
    }
  }
  take(buffer.trim());
  return { text, usage, finish, token: next };
}

const t0 = Date.now(), hops = [];
let text = "", token = null;
for (let n = 1; n <= MAX_HOPS; n++) {
  const h = await hop(token);
  text += h.text;
  hops.push({ hop: n, finish: h.finish, usage: h.usage, chars: h.text.length, seconds: (Date.now() - t0) / 1000 });
  console.log(JSON.stringify(hops.at(-1)));
  if (h.finish === "CONTINUATION" && h.token) { token = h.token; continue; }
  if (h.finish === "CONTINUATION") console.log("CONTINUATION without a continuation token; stopping");
  break;
}
const finish = hops.at(-1).finish;
writeFileSync(responsePath, JSON.stringify({ model, seconds: (Date.now() - t0) / 1000, finish, hops, text }, null, 1));

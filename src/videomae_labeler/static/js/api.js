// Thin client for the labeller's JSON API ({success, data, error} envelope).

async function request(method, url, body) {
  let res;
  try {
    res = await fetch(url, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch (err) {
    throw new Error(`無法連線到標註伺服器（${err.message}）`);
  }
  let payload;
  try {
    payload = await res.json();
  } catch {
    throw new Error(`伺服器回應格式錯誤（HTTP ${res.status}）`);
  }
  if (!res.ok || !payload.success) {
    const detail = payload.error ?? payload.detail;
    const msg = typeof detail === "string" ? detail : JSON.stringify(detail);
    throw new Error(msg || `HTTP ${res.status}`);
  }
  return payload.data;
}

const q = (path) => `path=${encodeURIComponent(path)}`;

export const api = {
  media: () => request("GET", "/api/media"),
  pick: (dir = "") => request("POST", "/api/pick", { dir }),
  open: (path) => request("POST", "/api/open", { path }),
  save: (path, segments) => request("POST", "/api/save", { path, segments }),
  modelStatus: (path) => request("GET", `/api/model?${q(path)}`),
  modelStart: (path, model) => request("POST", "/api/model", { path, model }),
  models: () => request("GET", "/api/models"),
  proxyStatus: (path) => request("GET", `/api/proxy?${q(path)}`),
  proxyStart: (path) => request("POST", "/api/proxy", { path }),
  videoUrl: (path, proxy = false) => `/api/video?${q(path)}${proxy ? "&proxy=true" : ""}`,
};

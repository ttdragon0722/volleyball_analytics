// Polling wrappers for the server's background jobs (model pre-labelling, H.264 proxy).
const POLL_MS = 1000;
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const pct = (p) => `${Math.round(p * 100)}%`;

/**
 * @param {{start:(path:string)=>Promise<object>, status:(path:string)=>Promise<object>,
 *          current:()=>string|undefined, progressText:(p:number)=>string,
 *          onProgress:(t:string)=>void, onReady:(s:object)=>void, onError:(m:string)=>void}} o
 */
function createJob({ start, status, current, progressText, onProgress, onReady, onError }) {
  let runningFor = null;
  return {
    async run({ reuseFinished = false } = {}) {
      const path = current();
      if (!path || runningFor === path) return;
      runningFor = path;
      try {
        let st = reuseFinished ? await status(path) : { state: "none" };
        if (!(reuseFinished && st.state === "ready") && st.state !== "running") st = await start(path);
        while (current() === path && st.state === "running") {
          onProgress(progressText(st.progress));
          await wait(POLL_MS);
          st = await status(path);
        }
        if (current() !== path) return;
        if (st.state === "ready") onReady(st);
        else onError(st.error ?? "背景工作失敗");
      } catch (err) {
        if (current() === path) onError(err.message);
      } finally {
        runningFor = null;
      }
    },
  };
}

export function createModelJob({ api, current, model, modelLabel, onProgress, onReady, onError }) {
  return createJob({
    start: (path) => api.modelStart(path, model()), status: api.modelStatus, current, onProgress, onError,
    progressText: (p) => `${modelLabel()} 預標中… ${pct(p)}（可以先開始標註）`,
    onReady: (st) => onReady(st.segments),
  });
}

export function createProxyJob({ api, current, onProgress, onReady, onError }) {
  return createJob({
    start: api.proxyStart, status: api.proxyStatus, current, onProgress, onReady, onError,
    progressText: (p) => `瀏覽器無法直接播放這個影片編碼，正在轉成相容格式… ${pct(p)}\n（只需轉一次，幀號與原影片相同）`,
  });
}

(function () {
  "use strict";

  const input = document.getElementById("reviewInput");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const clearBtn = document.getElementById("clearBtn");
  const charCount = document.getElementById("charCount");
  const exampleChips = document.getElementById("exampleChips");
  const resultEmpty = document.getElementById("resultEmpty");
  const resultBody = document.getElementById("resultBody");
  const latency = document.getElementById("latency");
  const historyBody = document.getElementById("historyBody");
  const historyCount = document.getElementById("historyCount");
  const statusDot = document.getElementById("statusDot");
  const statusText = document.getElementById("statusText");

  let examples = [];
  let history = [];
  let busy = false;

  function escapeHtml(str) {
    return String(str).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function setBusy(value) {
    busy = value;
    analyzeBtn.disabled = value;
    analyzeBtn.querySelector("span").textContent = value ? "分析中…" : "开始分析";
  }

  function updateCount() {
    charCount.textContent = String(input.value.length);
  }

  async function loadExamples() {
    try {
      const response = await fetch("/api/examples");
      examples = await response.json();
      exampleChips.innerHTML = "";
      examples.forEach(function (item, index) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "chip";
        button.textContent = (item.label === "positive" ? "正面示例 " : "负面示例 ") + (index + 1);
        button.addEventListener("click", function () {
          input.value = item.text;
          updateCount();
          input.focus();
        });
        exampleChips.appendChild(button);
      });
    } catch (error) {
      exampleChips.innerHTML = '<span class="keyword-none">示例加载失败</span>';
    }
  }

  function keywordHtml(items) {
    if (!items || !items.length) {
      return '<span class="keyword-none">无显著贡献词</span>';
    }
    return items.map(function (item) {
      const value = item.relative !== undefined ? item.relative : item.score;
      const score = (value >= 0 ? "+" : "") + value.toFixed(2);
      return '<span class="keyword">' + escapeHtml(item.token) + "<b>" + score + "</b></span>";
    }).join("");
  }

  function renderResult(data) {
    resultEmpty.hidden = true;
    resultBody.hidden = false;
    const positive = data.label === "positive";
    const verdict = document.getElementById("verdict");
    verdict.textContent = data.label_cn;
    verdict.classList.toggle("negative", !positive);
    document.getElementById("verdictEn").textContent = positive ? "POSITIVE" : "NEGATIVE";
    document.getElementById("confidence").innerHTML = (data.confidence * 100).toFixed(1) + "<small>%</small>";
    document.getElementById("probPos").textContent = (data.prob_positive * 100).toFixed(1) + "%";
    document.getElementById("probNeg").textContent = (data.prob_negative * 100).toFixed(1) + "%";
    document.getElementById("meterFill").style.width = (data.prob_positive * 100).toFixed(2) + "%";
    document.getElementById("latency").textContent = data.latency_ms.toFixed(0) + " ms";
    document.getElementById("tokenMeta").textContent =
      data.token_count + " tokens" + (data.truncated ? " · truncated" : "");
    document.getElementById("modelMeta").textContent = "BiLSTM-Attention · CPU";

    const explanation = data.explanation || { positive: [], negative: [] };
    document.getElementById("positiveKeywords").innerHTML = keywordHtml(explanation.positive);
    document.getElementById("negativeKeywords").innerHTML = keywordHtml(explanation.negative);
  }

  function addHistory(text, data) {
    history.unshift({
      time: new Date().toLocaleTimeString("zh-CN", { hour12: false }),
      text: text.replace(/\s+/g, " ").trim(),
      label: data.label_cn,
      positive: data.label === "positive",
      confidence: data.confidence,
      latency: data.latency_ms
    });
    history = history.slice(0, 12);
    historyCount.textContent = history.length + " 条";
    historyBody.innerHTML = history.map(function (row) {
      const summary = row.text.length > 92 ? row.text.slice(0, 92) + "…" : row.text;
      return "<tr>" +
        "<td>" + escapeHtml(row.time) + "</td>" +
        "<td>" + escapeHtml(summary) + "</td>" +
        '<td><span class="tag ' + (row.positive ? "" : "negative") + '">' + row.label + "</span></td>" +
        "<td>" + (row.confidence * 100).toFixed(1) + "%</td>" +
        "<td>" + row.latency.toFixed(0) + " ms</td>" +
        "</tr>";
    }).join("");
  }

  async function analyze() {
    const text = input.value.trim();
    if (!text || busy) return;
    setBusy(true);
    document.getElementById("latency").textContent = "—";
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text, explain: true, top_k: 6 })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "请求失败");
      renderResult(data);
      addHistory(text, data);
    } catch (error) {
      resultEmpty.hidden = false;
      resultBody.hidden = true;
      resultEmpty.querySelector("p").textContent = error.message;
      resultEmpty.querySelector("small").textContent = "请检查模型文件或后端服务";
    } finally {
      setBusy(false);
    }
  }

  async function loadModelInfo() {
    try {
      const response = await fetch("/api/model");
      const data = await response.json();
      const metrics = data.metrics || {};
      document.getElementById("statAcc").textContent = metrics.accuracy != null ? (metrics.accuracy * 100).toFixed(2) + "%" : "—";
      document.getElementById("statF1").textContent = metrics.f1 != null ? metrics.f1.toFixed(4) : "—";
      document.getElementById("statParams").textContent = data.parameter_count ? (data.parameter_count / 1e6).toFixed(2) + "M" : "—";
      document.getElementById("statVocab").textContent = data.vocab_size ? (data.vocab_size / 1000).toFixed(1) + "K" : "—";
      if (data.exists) {
        statusDot.classList.add("ready");
        statusText.textContent = "模型已就绪";
      } else {
        statusDot.classList.add("error");
        statusText.textContent = "模型未训练";
      }
    } catch (error) {
      statusDot.classList.add("error");
      statusText.textContent = "服务异常";
    }
  }

  input.addEventListener("input", updateCount);
  input.addEventListener("keydown", function (event) {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      analyze();
    }
  });
  analyzeBtn.addEventListener("click", analyze);
  clearBtn.addEventListener("click", function () {
    input.value = "";
    updateCount();
    input.focus();
  });

  async function initDemo() {
    const params = new URLSearchParams(window.location.search);
    if (!params.has("demo")) return;
    await loadExamples();
    const wanted = params.get("demo") === "negative" ? "negative" : "positive";
    const sample = examples.find(function (item) { return item.label === wanted; });
    if (sample) {
      input.value = sample.text;
      updateCount();
      setTimeout(analyze, 250);
    }
  }

  updateCount();
  loadExamples();
  loadModelInfo();
  initDemo();
})();

"use strict";

const messages = {
  en: {
    skip: "Skip to results", eyebrow: "A benchmark for social interaction", title: "Leaderboard",
    intro: "Evaluating multimodal models in social interaction through audio and video.",
    resources: "Resources", paper: "Paper", code: "Code", dataset: "Dataset", download: "Download results ↓",
    results: "Results", comparison: "",
    search: "Find a model", placeholder: "Model name", loading: "Loading results…",
    tableRegion: "Model results, scroll horizontally for all metrics", caption: "Model scores on a 0–100 scale. Higher is better. Select a column heading to sort.",
    model: "Model", howTitle: "How · Response examples", howIntro: "One conversation, two responses. Compare the answers, three judges’ scores and the supporting evidence.", whoTask: "Speaker identification", whenTask: "Response timing", howTask: "Response quality", jointTask: "Coverage and joint quality", accuracy: "Accuracy", tableHelp: "Click a metric to sort. All scores use a 0–100 scale; higher is better. — means unavailable and is placed last in either sort direction.",
    metricsTitle: "Reading the metrics", jointNote: "QEns_joint = QEns × Cov+ / 100. A high response-quality score alone does not imply reliable decisions about when to speak.",
    sourcesTitle: "Evaluation", sourcesText: "New evaluations use Gemini 3.8 Flash, Qwen3.8-Omni-Flash and GPT-5.6-Sol as judges. Model names link to the corresponding results and evaluation settings.",
    archive: "Historical results and original materials ↗",
    missingText: "Missing or unfinished evaluations are not assigned a score. This page does not combine the six metrics into an overall score.",
    correction: "Report a correction ↗", updated: "Updated", source: "Source", samples: "Samples", positive: "gold-positive", pending: "Evaluation pending", partial: "Quality scoring pending",
    noMatch: "No models match your search.", noData: "Results are being prepared. No scores have been published here yet.",
    loadError: "Results could not be loaded. Please reload the page, or use the Download results link.", count: "models shown",
    metrics: [
      ["Who", "Accuracy in identifying who is speaking from the available audio and video."],
      ["When", "Accuracy of the YES / NO decision to respond at a fixed timestamp, using only the audio and video available up to that time."],
      ["How · QGold", "Mean response quality over all gold-positive items: cases where the reference says a response is appropriate. Generation is forced even when the model predicts NO."],
      ["How · QEns", "Mean quality of non-empty responses on gold-positive items where the model also predicts YES. Each response is scored by the evaluation’s three judges."],
      ["When + How · Cov+", "Coverage of gold-positive items: the percentage for which the model predicts YES and generates a non-empty response."],
      ["When + How · QEns_joint", "Response quality weighted by coverage. Missed gold-positive opportunities contribute zero."]
    ]
  },
  zh: {
    skip: "跳转到结果", eyebrow: "面向社会交互的多模态评测", title: "排行榜",
    intro: "通过音频与视频，评测全模态模型在社会交互中的理解与回应能力。",
    resources: "研究资源", paper: "论文", code: "代码", dataset: "数据集", download: "下载结果 ↓",
    results: "评测结果", comparison: "",
    search: "查找模型", placeholder: "输入模型名称", loading: "正在加载结果…",
    tableRegion: "模型结果，可横向滚动查看全部指标", caption: "模型分数均采用 0–100 标度，越高越好。点击列标题排序。",
    model: "模型", howTitle: "How · 回答对照", howIntro: "同一段对话，两种真实回答。对照回答内容、三位评委的分数与依据。", whoTask: "说话者识别", whenTask: "回应时机", howTask: "回答质量", jointTask: "覆盖率与联合质量", accuracy: "准确率", tableHelp: "点击指标名称排序。所有分数均采用 0–100 标度，越高越好。— 表示缺失值，升序和降序均置于末尾。",
    metricsTitle: "指标说明", jointNote: "QEns_joint = QEns × Cov+ / 100。回答质量高，不一定意味着模型能可靠判断何时应该开口。",
    sourcesTitle: "评测方法", sourcesText: "新增评测使用 Gemini 3.8 Flash、Qwen3.8-Omni-Flash 和 GPT-5.6-Sol 评分。模型名称链接至对应结果与评测设置。",
    archive: "历史结果与原始材料 ↗",
    missingText: "缺失或尚未完成的评测不赋予分数。本站不把六项指标合成为一个总分。",
    correction: "反馈勘误 ↗", updated: "更新日期", source: "来源", samples: "样本", positive: "应回应样本", pending: "评测待完成", partial: "回答质量待评分",
    noMatch: "没有匹配的模型。", noData: "结果整理中，本站暂未发布分数。",
    loadError: "无法加载结果。请刷新页面，或使用“下载结果”链接。", count: "个模型",
    metrics: [
      ["Who", "说话者归属判断的准确率，即根据可见的音视频信息识别谁正在说话。"],
      ["When", "在固定时间点判断是否应回应的准确率，只使用该时间点及之前的音视频信息，输出 YES 或 NO。"],
      ["How · QGold", "所有应回应样本的平均回答质量。应回应样本指标准答案认为应当回应的情况；即使模型预测 NO，也强制生成回答进行评分。"],
      ["How · QEns", "在应回应且模型也预测 YES、生成非空回答的样本中，计算平均回答质量。每条回答由该评测规定的三位评委评分。"],
      ["When + How · Cov+", "应回应样本的覆盖率，即其中模型预测 YES 且生成非空回答的比例。"],
      ["When + How · QEns_joint", "结合覆盖率的回答质量。漏掉的应回应机会计为零分。"]
    ]
  }
};

Object.assign(messages.en, {"navIntro":"Introduction","navResults":"Leaderboard","navExamples":"Dataset examples","subtitle":"Who speaks. When to respond. What to say.","overview":"Social interaction requires more than understanding a video. SocialOmni evaluates whether a model can identify speakers, recognize when it should respond, and generate an appropriate reply from audio and visual context.","level1Title":"Who is speaking?","level1Text":"Connect voices to people using audio and visual cues.","level2Title":"When and how to respond?","level2Text":"Decide whether to speak at a given moment, then produce a context-appropriate response.","filmCaption":"SocialOmni · Project introduction","examplesIntro":"Explore the tasks through selected videos and their reference annotations.","question":"Question","reference":"Reference answer","showAnswer":"Show reference answer","caseSource":"Source annotation ↗"});
Object.assign(messages.zh, {"navIntro":"项目介绍","navResults":"排行榜","navExamples":"视频案例","subtitle":"谁在说话，何时回应，如何回应。","overview":"社会交互不止于理解视频。SocialOmni 评测模型能否结合音视频线索识别说话者、判断何时应当开口，并生成符合语境的回应。","level1Title":"谁在说话？","level1Text":"结合声音与视觉线索，将说话内容与人物对应。","level2Title":"何时回应，如何回应？","level2Text":"在指定时刻判断是否应该开口，并生成符合当前语境的回答。","filmCaption":"SocialOmni · 项目介绍视频","examplesIntro":"通过精选视频和原始参考标注，了解两个层级的评测任务。","question":"题目","reference":"参考答案","showAnswer":"查看参考答案","caseSource":"查看原始标注 ↗"});
function howCasesHTML(data, lang = "en") {
  if (!data) return "";
  const escape = value => String(value ?? "").replace(/[&<>"']/g, char => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"})[char]);
  const text = value => typeof value === "string" ? value : value?.[lang] || value?.en || "";
  const highlight = (value, quotes = []) => {
    const source = String(value || "");
    const matches = [...new Set(quotes.filter(quote => quote && source.includes(quote)))].sort((a, b) => b.length - a.length);
    let output = "", cursor = 0;
    while (cursor < source.length) {
      let next = -1, selected = "";
      for (const quote of matches) {
        const at = source.indexOf(quote, cursor);
        if (at >= 0 && (next < 0 || at < next)) { next = at; selected = quote; }
      }
      if (next < 0) return output + escape(source.slice(cursor));
      output += escape(source.slice(cursor, next)) + `<mark>${escape(selected)}</mark>`;
      cursor = next + selected.length;
    }
    return output;
  };
  const labels = lang === "zh" ? {question:"How 问题", reference:"参考回答", context:"上下文与高亮依据", positive:"正向案例", negative:"反向案例", translation:"中文翻译", explanation:"补充解释", review:"Codex 复核", source:"原始回答与评分 ↗", explanationSource:"评委补充说明 ↗", score:"原评分", evidence:"依据", agree:"与原评分一致", differ:"对原评分保留判断"} : {question:"How question", reference:"Reference answer", context:"Context and highlighted evidence", positive:"Positive example", negative:"Negative example", translation:"Chinese translation", explanation:"Follow-up explanation", review:"Codex review", source:"Original answer and scores ↗", explanationSource:"Judge explanations ↗", score:"Recorded score", evidence:"Evidence", agree:"Supports the recorded score", differ:"Reservation about the recorded score"};
  const evidence = quotes => (quotes || []).map(quote => `<p class="how-evidence"><mark>${escape(quote)}</mark></p>`).join("");
  const responses = data.responses.map(response => {
    const judges = response.judges.map(judge => `<div class="how-judge"><h4><span>${escape(judge.name)}</span><span>${labels.score}: ${escape(judge.score)} / 100</span></h4><p>${escape(text(judge.reason))}</p>${evidence(judge.evidence_quotes)}${judge.supports_recorded_score === false ? `<p class="explanation-label">${labels.differ}</p>` : ""}</div>`).join("");
    const review = response.review ? `<aside class="how-review"><h4>${labels.review} · ${escape(text(response.review.verdict))}</h4><p>${escape(text(response.review.text))}</p>${evidence(response.review.evidence_quotes)}</aside>` : "";
    const answer = typeof response.answer === "string" ? response.answer : response.answer.en;
    const translation = lang === "zh" && response.answer.zh ? `<p class="explanation-label">${labels.translation}</p><p>${escape(response.answer.zh)}</p>` : "";
    const source = /^https?:\/\//.test(response.source_url) ? `<a href="${escape(response.source_url)}">${labels.source}</a>` : "";
    return `<article class="how-response"><span class="how-label">${labels[response.polarity]}</span><h3>${escape(response.model)}</h3><blockquote>${highlight(answer, response.answer_highlights)}</blockquote>${translation}<div class="how-judges">${judges}</div>${review}<div class="how-sources">${source}</div></article>`;
  }).join("");
  const explanationSource = /^https?:\/\//.test(data.explanation_source_url) ? `<a href="${escape(data.explanation_source_url)}">${labels.explanationSource}</a>` : "";
  return `<article class="how-case"><div class="how-context"><div><video controls playsinline preload="none" src="${escape(data.video)}" poster="${escape(data.poster)}" aria-label="${escape(text(data.question))}"></video><div class="how-key-evidence"><h4>${labels.evidence}</h4>${evidence(data.context_highlights)}</div><details><summary>${labels.context}</summary><p class="how-transcript">${highlight(data.context, data.context_highlights)}</p></details></div><div><span class="level-label">LEVEL 2 · ${escape(data.item_id)}</span><h3>${labels.question}</h3><p>${escape(text(data.question))}</p><h4>${labels.reference}</h4><p>${escape(text(data.reference))}</p><p class="method-note">${escape(text(data.explanation_note))}</p><div class="how-sources">${explanationSource}</div></div></div><div class="how-responses">${responses}</div></article>`;
}

Object.assign(messages.en, {authors: "Authors", corresponding: "Corresponding author", citation: "Citation", citeText: "If you use SocialOmni in your research, please cite:", copyCitation: "Copy BibTeX", copiedCitation: "Copied", copyFailed: "Select and copy the citation below", downloadCitation: "Download BibTeX"});
Object.assign(messages.zh, {authors: "作者", corresponding: "通讯作者", citation: "引用", citeText: "如果你的研究使用了 SocialOmni，请引用：", copyCitation: "复制 BibTeX", copiedCitation: "已复制", copyFailed: "请选中下方引用文本复制", downloadCitation: "下载 BibTeX"});

function authorsHTML(publication, lang = "en") {
  const escape = value => String(value).replace(/[&<>"']/g, char => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"})[char]);
  const names = publication.authors.map(author => {
    const url = author.scholar_url;
    const label = author.link_type === "search" ? "Google Scholar search / 谷歌学术检索" : "Google Scholar";
    const searchLabel = author.link_type === "search" ? `<span class="author-search">${lang === "zh" ? "检索" : "search"}</span>` : "";
    const name = url && /^https:\/\/scholar\.google\.com\//.test(url) ? `<a href="${escape(url)}" title="${label}">${escape(author.name)}${searchLabel}</a>` : escape(author.name);
    return `<li>${name}<sup>${author.affiliation}${author.corresponding ? ",†" : ""}</sup></li>`;
  }).join("");
  const affiliations = publication.affiliations.map(item => `<span><sup>${item.id}</sup>${escape(item[lang])}</span>`).join("");
  return `<ul class="author-list" aria-label="${messages[lang].authors}">${names}</ul><p class="affiliations">${affiliations}</p><p class="corresponding-note">† ${messages[lang].corresponding}</p>`;
}

let examples;
let publication;
let howExamples;

const metricKeys = ["who", "when", "qgold", "qens", "cov_plus", "qens_joint"];
let language = "en";
try { language = localStorage.getItem("socialomni-language") === "zh" ? "zh" : "en"; } catch {}
let dataset;
let failed = false;
let sortKey = "qens_joint";
let descending = true;
const search = document.getElementById("search");
const status = document.getElementById("load-status");
const localized = (value) => typeof value === "string" ? value : value?.[language] || value?.en || "";
const score = (record, key) => record.status === "pending" ? null : record.metrics?.[key];
const validScore = (value) => typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 100;

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function sourceLink(source) {
  const link = element("a", localized(source.label));
  link.href = safeUrl(source.url);
  return link;
}

function safeUrl(value) {
  const url = new URL(value, location.href);
  if (url.protocol !== "https:" && url.protocol !== "http:") throw new Error("Unsupported link protocol");
  return url.href;
}

function render() {
  const t = messages[language];
  document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
  document.title = `SocialOmni · ${t.title}`;
  document.querySelectorAll("[data-i18n]").forEach((node) => { node.textContent = t[node.dataset.i18n]; });
  document.querySelectorAll("[data-i18n-aria]").forEach((node) => node.setAttribute("aria-label", t[node.dataset.i18nAria]));
  document.querySelectorAll("[data-i18n-placeholder]").forEach((node) => { node.placeholder = t[node.dataset.i18nPlaceholder]; });
  document.querySelectorAll("[data-lang]").forEach((node) => node.setAttribute("aria-pressed", String(node.dataset.lang === language)));
  document.getElementById("metric-definitions").replaceChildren(...t.metrics.flatMap(([name, definition]) => [element("dt", name), element("dd", definition)]));
  renderCases();
  if (publication) document.getElementById("authors").innerHTML = authorsHTML(publication, language);
  if (howExamples) {
    document.getElementById("how-case-list").innerHTML = howCasesHTML(howExamples, language);
    document.getElementById("how-title").textContent = localized(howExamples.title);
    document.querySelector("[data-i18n=howIntro]").textContent = localized(howExamples.intro);
  }
  if (!dataset) {
    if (!document.getElementById("rows").children.length) status.textContent = failed ? t.loadError : t.loading;
    return;
  }
  document.getElementById("updated").textContent = dataset.updated_at ? `${t.updated} ${dataset.updated_at}` : "";
  renderRows();
}

function renderRows() {
  if (!dataset) return;
  const t = messages[language];
  const query = search.value.trim().toLocaleLowerCase();
  const records = dataset.records.filter((item) => `${item.model} ${item.variant || ""}`.toLocaleLowerCase().includes(query));
  records.sort((a, b) => {
    if (sortKey === "model") return a.model.localeCompare(b.model) * (descending ? -1 : 1);
    const av = score(a, sortKey), bv = score(b, sortKey);
    if (!validScore(av)) return validScore(bv) ? 1 : a.model.localeCompare(b.model);
    if (!validScore(bv)) return -1;
    return (av - bv) * (descending ? -1 : 1) || a.model.localeCompare(b.model);
  });
  document.getElementById("rows").replaceChildren(...records.map((record) => {
    const row = element("tr");
    const name = element("td");
    if (record.source) name.append(sourceLink({ label: record.model, url: record.source.url }));
    else name.textContent = record.model;
    if (record.variant) name.append(element("span", record.variant, "model-meta"));
    if (record.status === "pending") name.append(element("span", t.pending, "model-meta"));
    if (record.status === "partial") name.append(element("span", t.partial, "model-meta"));
    row.append(name);
    metricKeys.forEach((key) => row.append(element("td", validScore(score(record, key)) ? score(record, key).toFixed(2) : "—")));
    return row;
  }));
  document.getElementById("table-wrap").hidden = !records.length;
  status.textContent = records.length ? `${records.length} ${language === "en" && records.length === 1 ? "model shown" : t.count}` : query ? t.noMatch : t.noData;
  document.querySelectorAll("th[data-key]").forEach((header) => {
    const selected = header.dataset.key === sortKey;
    if (selected) header.setAttribute("aria-sort", descending ? "descending" : "ascending");
    else header.removeAttribute("aria-sort");
    header.querySelector(".sort-indicator").textContent = selected ? descending ? " ↓" : " ↑" : " ↕";
  });
}

document.querySelectorAll("[data-lang]").forEach((button) => button.addEventListener("click", () => {
  language = button.dataset.lang;
  try { localStorage.setItem("socialomni-language", language); } catch {}
  render();
}));
document.querySelectorAll("[data-sort]").forEach((button) => button.addEventListener("click", () => {
  const key = button.dataset.sort;
  descending = sortKey === key ? !descending : key !== "model";
  sortKey = key;
  renderRows();
}));
search.addEventListener("input", renderRows);
function useDataset(data) {
  if (!Array.isArray(data.records)) throw new Error("Invalid data format");
  dataset = data;
  Object.entries(data.links || {}).forEach(([key, value]) => {
    const link = document.getElementById(`${key}-link`);
    if (link) link.href = safeUrl(value);
  });
  search.disabled = !data.records.length;
}

const bootstrap = document.getElementById("socialomni-data");
if (bootstrap) {
  try {
    const data = JSON.parse(bootstrap.textContent);
    useDataset(data.results);
    howExamples = data.howExamples;
    publication = data.publication;
    if (Array.isArray(data.examples.cases)) examples = data.examples.cases;
  } catch {
    // Keep the generated HTML visible if embedded data cannot be read.
  }
}
render();
document.getElementById("copy-citation").addEventListener("click", async (event) => {
  const button = event.currentTarget;
  try {
    await navigator.clipboard.writeText(document.getElementById("citation-code").textContent);
    button.textContent = messages[language].copiedCitation;
  } catch {
    button.textContent = messages[language].copyFailed;
  }
});
if (!dataset) {
  fetch("data.json").then((response) => {
    if (!response.ok) throw new Error("Data unavailable");
    return response.json();
  }).then((data) => {
    useDataset(data);
    render();
  }).catch(() => {
    failed = true;
    search.disabled = true;
    if (!document.getElementById("rows").children.length) {
      status.classList.add("error");
      render();
    }
  });
}

function renderCases() {
  if (!examples) return;
  const list = document.getElementById("case-list");
  const t = messages[language];
  list.replaceChildren(...examples.map((item) => {
    const article = element("article", undefined, "case");
    const media = element("video");
    media.controls = true; media.playsInline = true; media.preload = "none";
    media.src = safeUrl(item.video); if (item.poster) media.poster = safeUrl(item.poster);
    media.setAttribute("aria-label", localized(item.title));
    const content = element("div", undefined, "case-content");
    content.append(element("span", "LEVEL " + item.level + " · " + item.id, "level-label"), element("h3", localized(item.title)));
    if (item.description) content.append(element("p", localized(item.description), "case-description"));
    content.append(element("h4", t.question), element("p", language === "zh" && item.question_zh ? item.question_zh : item.question));
    const options = language === "zh" && item.options_zh ? item.options_zh : item.options;
    if (options?.length) { const ul = element("ul", undefined, "case-options"); options.forEach(option => ul.append(element("li", option))); content.append(ul); }
    const details = element("details"); details.append(element("summary", t.showAnswer));
    details.append(element("p", (language === "zh" && item.reference_zh ? item.answer + ". " + item.reference_zh : item.reference)));
    if (item.how_reference) { details.append(element("h4", language === "zh" ? item.how_question_zh : item.how_question), element("p", item.how_reference)); if(language === "zh" && item.how_reference_zh) details.append(element("p", item.how_reference_zh)); }
    content.append(details);
    if(item.source_url) content.append(sourceLink({label: t.caseSource, url: item.source_url}));
    article.append(media, content); return article;
  }));
}
if (!examples) {
  fetch("cases.json").then(response => {
    if (!response.ok) throw new Error("Examples unavailable");
    return response.json();
  }).then(data => {
    if (!Array.isArray(data.cases)) throw new Error("Invalid examples format");
    examples = data.cases;
    renderCases();
  }).catch(() => {});
}

import { readFile, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { runInNewContext } from "node:vm";

const read = (name) => readFile(new URL(name, import.meta.url), "utf8");
const [resultText, caseText, script] = await Promise.all([read("data.json"), read("cases.json"), read("leaderboard.js")]);
const results = JSON.parse(resultText);
const examples = JSON.parse(caseText);
const howExamples = JSON.parse(await read("how-cases.json"));
const {messages, howCasesHTML} = runInNewContext(`${script.slice(0, script.indexOf("let examples;"))}\n({messages, howCasesHTML});`);
const escape = (value = "") => String(value).replace(/[&<>"']/g, char => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"})[char]);
const english = (value) => typeof value === "string" ? value : value?.en || "";
const score = (record, key) => record.status === "pending" ? null : record.metrics?.[key];
const valid = (value) => typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 100;
const keys = ["who", "when", "qgold", "qens", "cov_plus", "qens_joint"];
const link = (url, label) => {
  if (!/^https?:\/\//.test(url)) throw new Error("Invalid source URL");
  return `<a href="${escape(url)}">${escape(label)}</a>`;
};
const records = [...results.records].sort((a, b) => {
  const av = score(a, "qens_joint"), bv = score(b, "qens_joint");
  if (!valid(av)) return valid(bv) ? 1 : a.model.localeCompare(b.model);
  if (!valid(bv)) return -1;
  return bv - av || a.model.localeCompare(b.model);
});
const rows = records.map(record => {
  let name = record.source ? link(record.source.url, record.model) : escape(record.model);
  if (record.variant) name += `<span class="model-meta">${escape(record.variant)}</span>`;
  if (["pending", "partial"].includes(record.status)) name += `<span class="model-meta">${escape(messages.en[record.status])}</span>`;
  return `<tr><td>${name}</td>${keys.map(key => `<td>${valid(score(record, key)) ? score(record, key).toFixed(2) : "—"}</td>`).join("")}</tr>`;
}).join("\n");
const cases = examples.cases.map(item => {
  const options = item.options?.length ? `<ul class="case-options">${item.options.map(option => `<li>${escape(option)}</li>`).join("")}</ul>` : "";
  const response = item.how_reference ? `<h4>${escape(item.how_question)}</h4><p>${escape(item.how_reference)}</p>` : "";
  return `<article class="case"><video controls playsinline preload="none" src="${escape(item.video)}"${item.poster ? ` poster="${escape(item.poster)}"` : ""} aria-label="${escape(english(item.title))}"></video><div class="case-content"><span class="level-label">LEVEL ${escape(item.level)} · ${escape(item.id)}</span><h3>${escape(english(item.title))}</h3>${item.description ? `<p class="case-description">${escape(english(item.description))}</p>` : ""}<h4>${messages.en.question}</h4><p>${escape(item.question)}</p>${options}<details><summary>${messages.en.showAnswer}</summary><p>${escape(item.reference)}</p>${response}</details>${item.source_url ? link(item.source_url, messages.en.caseSource) : ""}</div></article>`;
}).join("\n");
const metrics = messages.en.metrics.map(([name, description]) => `<dt>${escape(name)}</dt><dd>${escape(description)}</dd>`).join("\n");
let html = await read("index.html");
for (const [id, tag, content] of [["how-case-list", "div", howCasesHTML(howExamples, "en")], ["rows", "tbody", rows], ["case-list", "div", cases], ["metric-definitions", "dl", metrics]]) {
  const start = `<!-- generated:${id}:start -->`, end = `<!-- generated:${id}:end -->`;
  if (html.includes(start)) {
    const from = html.indexOf(start), to = html.indexOf(end, from);
    if (to < 0) throw new Error(`Missing end marker: ${id}`);
    html = html.slice(0, from) + start + "\n" + content + "\n" + html.slice(to);
  } else {
    const empty = new RegExp(`(<${tag}\\b[^>]*\\bid="${id}"[^>]*>)\\s*(</${tag}>)`);
    if (!empty.test(html)) throw new Error(`Missing empty container: ${id}`);
    html = html.replace(empty, `$1${start}\n${content}\n${end}$2`);
  }
}
html = html.replace(/(<p id="load-status"[^>]*>)[^<]*(<\/p>)/, `$1${records.length} models shown$2`);
html = html.replace(/(<h2 id="how-title"[^>]*>)[\s\S]*?(<\/h2>)/, `$1${escape(english(howExamples.title))}$2`);
html = html.replace(/(<p[^>]*data-i18n="howIntro"[^>]*>)[\s\S]*?(<\/p>)/, `$1${escape(english(howExamples.intro))}$2`);
const payload = JSON.stringify({results, examples, howExamples}).replace(/</g, "\\u003c");
const bootstrap = `<script id="socialomni-data" type="application/json">${payload}</script>`;
if (html.includes('id="socialomni-data"')) html = html.replace(/<script id="socialomni-data" type="application\/json">[\s\S]*?<\/script>/, () => bootstrap);
else html = html.replace("</body>", `${bootstrap}\n</body>`);
const tokenHash = createHash("sha256").update(await read("tokens.css")).digest("hex").slice(0, 12);
const styles = (await read("styles.css")).replace(/tokens\.css(?:\?[^"']*)?/, `tokens.css?v=${tokenHash}`);
await writeFile(new URL("styles.css", import.meta.url), styles);
for (const asset of ["styles.css", "leaderboard.js"]) {
  const hash = createHash("sha256").update(await read(asset)).digest("hex").slice(0, 12);
  html = html.replace(new RegExp(`(["'])${asset.replace(".", "\\.")}(?:\\?[^"']*)?(["'])`, "g"), `$1${asset}?v=${hash}$2`);
}
await writeFile(new URL("index.html", import.meta.url), html);
console.log(`Generated ${records.length} model rows, ${examples.cases.length} cases and ${messages.en.metrics.length} metric definitions.`);

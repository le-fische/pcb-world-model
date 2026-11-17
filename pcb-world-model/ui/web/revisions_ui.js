/**
 * Multi-Revision Explorer & Showcase Controller (rev1 -> rev15)
 * Handles client-side multi-revision navigation, technical dossiers,
 * benchmark matrix, side-by-side revision comparator, and Rev 15 LLM studio.
 */

// Simple robust Markdown to HTML converter
export function renderMarkdownToHtml(md) {
  if (!md) return "<p><em>No documentation available for this revision.</em></p>";
  
  // Normalize line breaks
  let lines = md.replace(/\r\n/g, "\n").split("\n");
  let html = [];
  let inCode = false;
  let codeLang = "";
  let codeBuffer = [];
  let inTable = false;
  let tableHeader = false;

  for (let i = 0; i < lines.length; i++) {
    let line = lines[i];

    // Fenced code blocks
    if (line.trim().startsWith("```")) {
      if (inCode) {
        html.push(`<pre class="mono"><code class="lang-${codeLang}">${escapeHtml(codeBuffer.join("\n"))}</code></pre>`);
        inCode = false;
        codeBuffer = [];
      } else {
        inCode = true;
        codeLang = line.trim().substring(3).trim();
        codeBuffer = [];
      }
      continue;
    }
    if (inCode) {
      codeBuffer.push(line);
      continue;
    }

    // Tables
    if (line.trim().startsWith("|") && line.trim().endsWith("|")) {
      if (!inTable) {
        html.push('<div class="bench-table-wrap"><table>');
        inTable = true;
        tableHeader = true;
      }
      // Check for separator row |:---|:---|
      if (line.includes("---")) {
        tableHeader = false;
        continue;
      }
      let cells = line.split("|").slice(1, -1).map(c => c.trim());
      let tag = tableHeader ? "th" : "td";
      let rowHtml = cells.map(c => `<${tag}>${formatInline(c)}</${tag}>`).join("");
      html.push(`<tr>${rowHtml}</tr>`);
      continue;
    } else if (inTable) {
      html.push("</table></div>");
      inTable = false;
      tableHeader = false;
    }

    // Headings
    if (line.startsWith("# ")) {
      html.push(`<h1>${formatInline(line.substring(2))}</h1>`);
    } else if (line.startsWith("## ")) {
      html.push(`<h2>${formatInline(line.substring(3))}</h2>`);
    } else if (line.startsWith("### ")) {
      html.push(`<h3>${formatInline(line.substring(4))}</h3>`);
    } else if (line.startsWith("#### ")) {
      html.push(`<h4>${formatInline(line.substring(5))}</h4>`);
    } else if (line.trim().startsWith("> ")) {
      html.push(`<blockquote>${formatInline(line.trim().substring(2))}</blockquote>`);
    } else if (line.trim().startsWith("- ") || line.trim().startsWith("* ")) {
      html.push(`<ul><li>${formatInline(line.trim().substring(2))}</li></ul>`);
    } else if (line.trim() === "---") {
      html.push("<hr>");
    } else if (line.trim().length > 0) {
      html.push(`<p>${formatInline(line)}</p>`);
    }
  }

  if (inCode) {
    html.push(`<pre class="mono"><code>${escapeHtml(codeBuffer.join("\n"))}</code></pre>`);
  }
  if (inTable) {
    html.push("</table></div>");
  }

  return html.join("\n");
}

function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function formatInline(str) {
  return str
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/g, "<em>$1</em>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\$([^$]+)\$/g, '<span class="mono" style="color:#e5c07b;">$1</span>')
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
}

export const FRONTIER_BASELINES = [
  {
    revision: "GPT-6 Astra (Frontier GUI Agent)",
    status: "Baseline",
    capability: "Multimodal Vision-Language Screen Agent (Unconstrained pixel actions)",
    targets: "usb_uart, buck_converter",
    score: "58.40 / 100",
    scoreNum: 58.4,
    time: "48.2s",
    vias: "42 vias (frequent shorts)",
    drc: "14 Errors (Shorts/Clearance)",
    layers: "2L only",
  },
  {
    revision: "Skyfall SKY-001 (RL Graph Placer)",
    status: "Baseline",
    capability: "Deep RL Graph Placer + Conventional Heuristic Maze Router",
    targets: "mcu_sensor_node",
    score: "76.20 / 100",
    scoreNum: 76.2,
    time: "24.5s",
    vias: "31 vias",
    drc: "4 Errors",
    layers: "2L only",
  },
  {
    revision: "Freerouting (Heuristic Baseline)",
    status: "Baseline",
    capability: "Deterministic Ripup-and-Reroute Heuristic (No spatial world model)",
    targets: "B1–B5",
    score: "84.10 / 100",
    scoreNum: 84.1,
    time: "45.0s",
    vias: "48 vias",
    drc: "0 Errors",
    layers: "2L–4L",
  },
];

/* Client logic for the chat page, auth-aware API calls and history panel. */
(function () {
  "use strict";

  const chat = document.getElementById("chat");
  const form = document.getElementById("ask-form");
  const queryInput = document.getElementById("query");
  const reportBtn = document.getElementById("report-btn");
  const historyPanel = document.getElementById("history-panel");
  const historyList = document.getElementById("history-list");
  const historyToggle = document.getElementById("history-toggle");
  const historyClose = document.getElementById("history-close");

  /* ---------- helpers ---------- */

  function scrollBottom() {
    if (chat) chat.scrollTop = chat.scrollHeight;
  }

  /**
   * Render assistant text from Markdown (GitHub-flavoured, tables enabled) and
   * sanitise it before it touches the DOM.
   *
   * Returns null when the vendored libraries are unavailable or parsing fails,
   * so callers fall back to plain text instead of showing raw HTML.
   */
  function renderMarkdown(text) {
    if (typeof window.marked === "undefined" || typeof window.DOMPurify === "undefined") {
      return null;
    }
    try {
      const rawHtml = window.marked.parse(text, { gfm: true, breaks: true });
      return window.DOMPurify.sanitize(rawHtml, {
        USE_PROFILES: { html: true },
        FORBID_TAGS: ["script", "style", "iframe", "form", "input", "button"],
        FORBID_ATTR: ["style", "onerror", "onload", "onclick"],
      });
    } catch (err) {
      console.warn("Markdown rendering failed; falling back to plain text.", err);
      return null;
    }
  }

  function addMessage(text, who) {
    if (!chat) return null;
    const div = document.createElement("div");
    div.className = "msg " + who;

    // Only successful assistant answers are Markdown. User input, loaders and
    // error bubbles stay verbatim so they can never inject markup.
    const html = who === "ai" ? renderMarkdown(text) : null;
    if (html !== null) {
      div.innerHTML = html;
      div.classList.add("markdown");
    } else {
      div.textContent = text;
    }

    chat.appendChild(div);
    scrollBottom();
    return div;
  }

  function addCitations(sections) {
    if (!chat || !sections || sections.length === 0) return;

    const wrap = document.createElement("div");
    wrap.className = "citations";

    const heading = document.createElement("div");
    heading.className = "label";
    heading.textContent = "IPC sections retrieved:";
    wrap.appendChild(heading);

    sections.forEach(function (s) {
      const card = document.createElement("div");
      card.className = "cite";
      card.innerHTML =
        '<span class="sec">Section ' + s.section + " — " + s.section_title + "</span>" +
        '<div class="chap">Chapter ' + s.chapter + ": " + s.chapter_title + "</div>";
      wrap.appendChild(card);
    });

    chat.appendChild(wrap);
    scrollBottom();
  }

  async function readJson(response) {
    try {
      return await response.json();
    } catch (_) {
      return {};
    }
  }

  function handleUnauthorized(response) {
    if (response.status === 401) {
      window.location.href = "/login";
      return true;
    }
    return false;
  }

  /* ---------- ask flow ---------- */

  if (form && queryInput) {
    form.addEventListener("submit", async function (event) {
      event.preventDefault();

      const query = queryInput.value.trim();
      if (!query) return;

      addMessage(query, "user");
      queryInput.value = "";
      const loader = addMessage("Retrieving IPC sections and generating an answer…", "loader");

      let response;
      try {
        response = await fetch("/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json", Accept: "application/json" },
          credentials: "same-origin",
          body: JSON.stringify({ query: query }),
        });
      } catch (err) {
        if (loader) loader.remove();
        addMessage("Network error: " + err.message, "ai error");
        return;
      }

      if (loader) loader.remove();
      if (handleUnauthorized(response)) return;

      if (!response.ok) {
        const data = await readJson(response);
        addMessage("Error: " + (data.error || "Server error (" + response.status + ")."), "ai error");
        return;
      }

      const data = await readJson(response);
      if (data.error) {
        addMessage("Error: " + data.error, "ai error");
        return;
      }

      addMessage(data.answer, "ai");
      addCitations(data.sections);
      if (reportBtn) reportBtn.disabled = false;
    });
  }

  /* ---------- report ---------- */

  if (reportBtn) {
    reportBtn.addEventListener("click", function () {
      window.location.href = "/report";
    });
  }

  /* ---------- history side panel ---------- */

  function renderHistory(items) {
    if (!historyList) return;
    historyList.innerHTML = "";

    if (!items.length) {
      historyList.innerHTML = '<p class="muted">No queries yet.</p>';
      return;
    }

    items.forEach(function (item) {
      const entry = document.createElement("div");
      entry.className = "history-item" + (item.status === "FAILED" ? " failed" : "");

      const q = document.createElement("div");
      q.className = "q";
      q.textContent = item.query_text;

      const meta = document.createElement("div");
      meta.className = "meta";
      meta.innerHTML =
        '<span class="badge ' + (item.status === "FAILED" ? "failed" : "success") + '">' +
        item.status + "</span>" +
        "<span>" + (item.latency_ms != null ? item.latency_ms + " ms" : "—") + "</span>" +
        "<span>" + new Date(item.created_at).toLocaleString() + "</span>";

      entry.appendChild(q);
      entry.appendChild(meta);

      entry.addEventListener("click", function () {
        const body = item.status === "FAILED"
          ? "Error: " + (item.error_message || "Unknown error")
          : item.response_text || "(no response stored)";
        historyList.innerHTML = "";
        const detail = document.createElement("div");
        detail.className = "history-item";
        detail.innerHTML = "<div class='q'></div><pre></pre>";
        detail.querySelector(".q").textContent = item.query_text;
        detail.querySelector("pre").textContent = body;
        historyList.appendChild(detail);
      });

      historyList.appendChild(entry);
    });
  }

  async function loadHistory() {
    if (!historyList) return;
    historyList.innerHTML = '<p class="muted">Loading…</p>';

    const response = await fetch("/api/history?per_page=25", {
      credentials: "same-origin",
      headers: { Accept: "application/json" },
    });

    if (handleUnauthorized(response)) return;
    const data = await readJson(response);
    renderHistory(data.items || []);
  }

  if (historyToggle && historyPanel) {
    historyToggle.addEventListener("click", function () {
      const opening = historyPanel.hidden;
      historyPanel.hidden = !opening;
      if (opening) loadHistory();
    });
  }

  if (historyClose && historyPanel) {
    historyClose.addEventListener("click", function () {
      historyPanel.hidden = true;
    });
  }
})();

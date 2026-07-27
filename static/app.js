const chat = document.getElementById("chat");
const form = document.getElementById("ask-form");
const queryInput = document.getElementById("query");
const reportBtn = document.getElementById("report-btn");

function scrollBottom() {
  chat.scrollTop = chat.scrollHeight;
}

function addMessage(text, who) {
  const div = document.createElement("div");
  div.className = "msg " + who;
  div.textContent = text;
  chat.appendChild(div);
  scrollBottom();
  return div;
}

function addCitations(sections) {
  if (!sections || sections.length === 0) return;
  const wrap = document.createElement("div");
  wrap.className = "citations";

  const heading = document.createElement("div");
  heading.className = "label";
  heading.textContent = "IPC sections retrieved:";
  wrap.appendChild(heading);

  sections.forEach((s) => {
    const card = document.createElement("div");
    card.className = "cite";
    card.innerHTML =
      '<span class="sec">Section ' + s.section + " — " +
      s.section_title + "</span>" +
      '<div class="chap">Chapter ' + s.chapter + ": " + s.chapter_title + "</div>";
    wrap.appendChild(card);
  });

  chat.appendChild(wrap);
  scrollBottom();
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = queryInput.value.trim();
  if (!query) return;

  addMessage(query, "user");
  queryInput.value = "";

  const loader = addMessage("Retrieving IPC sections and generating an answer…", "loader");

  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });

    loader.remove();

    if (!res.ok) {
      let msg = "Server error (" + res.status + ")";
      try {
        const errData = await res.json();
        msg = "Error: " + (errData.error || msg);
      } catch (_) {
        // Render returns an HTML error page on 502; res.json() fails.
        if (res.status === 502) {
          msg = "The AI model is unavailable. Rare or rate-limited. Try again in a minute.";
        }
      }
      addMessage(msg, "ai error");
      return;
    }

    let data;
    try {
      data = await res.json();
    } catch (_) {
      addMessage("The server returned an unexpected response. Please try again.", "ai error");
      return;
    }

    if (data.error) {
      addMessage("Error: " + data.error, "ai error");
      return;
    }

    addMessage(data.answer, "ai");
    addCitations(data.sections);
    reportBtn.disabled = false;
  } catch (err) {
    loader.remove();
    addMessage("Network error: " + err.message, "ai error");
  }
});

reportBtn.addEventListener("click", () => {
  window.location.href = "/report";
});
/* Admin dashboard: stats cards, user management, system logs. */
(function () {
  "use strict";

  const usersBody = document.getElementById("users-body");
  const logsBody = document.getElementById("logs-body");
  const modal = document.getElementById("modal");
  const modalTitle = document.getElementById("modal-title");
  const modalBody = document.getElementById("modal-body");

  function esc(value) {
    return String(value == null ? "" : value).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  async function api(path, options) {
    const response = await fetch(path, Object.assign({
      credentials: "same-origin",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
    }, options || {}));

    if (response.status === 401) {
      window.location.href = "/login";
      return null;
    }
    if (response.status === 403) {
      window.location.href = "/";
      return null;
    }
    try {
      return await response.json();
    } catch (_) {
      return {};
    }
  }

  function openModal(title, html) {
    modalTitle.textContent = title;
    modalBody.innerHTML = html;
    modal.hidden = false;
  }

  document.getElementById("modal-close").addEventListener("click", function () {
    modal.hidden = true;
  });
  modal.addEventListener("click", function (event) {
    if (event.target === modal) modal.hidden = true;
  });

  /* ---------- stats ---------- */

  async function loadStats() {
    const data = await api("/api/admin/stats");
    if (!data) return;
    document.getElementById("stat-users").textContent = data.total_users ?? "—";
    document.getElementById("stat-queries").textContent = data.total_queries ?? "—";
    document.getElementById("stat-success").textContent =
      (data.success_rate != null ? data.success_rate + "%" : "—");
    document.getElementById("stat-latency").textContent =
      (data.avg_latency_ms != null ? data.avg_latency_ms + " ms" : "—");
  }

  /* ---------- users ---------- */

  async function loadUsers() {
    const data = await api("/api/admin/users");
    if (!data) return;

    usersBody.innerHTML = "";
    (data.users || []).forEach(function (user) {
      const tr = document.createElement("tr");
      tr.innerHTML =
        "<td>" + user.id + "</td>" +
        "<td>" + esc(user.username) + "</td>" +
        "<td>" + esc(user.email) + "</td>" +
        '<td><span class="badge ' + (user.role === "admin" ? "admin" : "user") + '">' +
        esc(user.role) + "</span></td>" +
        "<td>" + (user.is_active ? "Active" : "Inactive") + "</td>" +
        "<td>" + user.query_count + "</td>" +
        "<td>" + (user.created_at ? new Date(user.created_at).toLocaleDateString() : "—") + "</td>" +
        '<td><div class="row-actions">' +
        '<button class="mini-btn" data-action="role" data-id="' + user.id + '" data-role="' +
        (user.role === "admin" ? "user" : "admin") + '">Make ' +
        (user.role === "admin" ? "user" : "admin") + "</button>" +
        '<button class="mini-btn danger" data-action="status" data-id="' + user.id + '">' +
        (user.is_active ? "Deactivate" : "Activate") + "</button>" +
        "</div></td>";
      usersBody.appendChild(tr);
    });

    if (!(data.users || []).length) {
      usersBody.innerHTML = '<tr><td colspan="8" class="muted">No users.</td></tr>';
    }
  }

  usersBody.addEventListener("click", async function (event) {
    const button = event.target.closest("button[data-action]");
    if (!button) return;

    const id = button.dataset.id;
    if (button.dataset.action === "role") {
      await api("/api/admin/users/" + id + "/role", {
        method: "PATCH",
        body: JSON.stringify({ role: button.dataset.role }),
      });
    } else if (button.dataset.action === "status") {
      await api("/api/admin/users/" + id + "/toggle-status", { method: "PATCH" });
    }

    await Promise.all([loadUsers(), loadStats()]);
  });

  /* ---------- logs ---------- */

  async function loadLogs() {
    const status = document.getElementById("log-status").value;
    const sort = document.getElementById("log-sort").value;
    const params = new URLSearchParams({ sort: sort });
    if (status) params.set("status", status);

    const data = await api("/api/admin/logs?" + params.toString());
    if (!data) return;

    logsBody.innerHTML = "";
    (data.logs || []).forEach(function (log) {
      const tr = document.createElement("tr");
      tr.innerHTML =
        "<td>" + (log.created_at ? new Date(log.created_at).toLocaleString() : "—") + "</td>" +
        "<td>" + esc(log.user_email || "—") + "</td>" +
        '<td class="wrap">' + esc(log.query_text) + "</td>" +
        '<td><span class="badge ' + (log.status === "FAILED" ? "failed" : "success") + '">' +
        esc(log.status) + "</span></td>" +
        "<td>" + (log.latency_ms != null ? log.latency_ms + " ms" : "—") + "</td>" +
        '<td><button class="mini-btn" data-log="' + log.id + '">View</button></td>';
      logsBody.appendChild(tr);
    });

    if (!(data.logs || []).length) {
      logsBody.innerHTML = '<tr><td colspan="6" class="muted">No logs.</td></tr>';
    }

    logsBody.querySelectorAll("button[data-log]").forEach(function (button) {
      button.addEventListener("click", function () {
        const log = (data.logs || []).find(function (l) {
          return String(l.id) === button.dataset.log;
        });
        if (!log) return;

        const sections = (log.retrieved_sections || [])
          .map(function (s) { return "Section " + s.section + " — " + s.section_title; })
          .join("\n");

        openModal(
          "Query #" + log.id,
          "<div><strong>User:</strong> " + esc(log.user_email) + "</div>" +
          "<div><strong>Status:</strong> " + esc(log.status) +
          " · <strong>Latency:</strong> " + (log.latency_ms ?? "—") + " ms</div>" +
          "<div><strong>Query</strong><pre>" + esc(log.query_text) + "</pre></div>" +
          "<div><strong>Response</strong><pre>" +
          esc(log.response_text || log.error_message || "(none)") + "</pre></div>" +
          "<div><strong>Citations</strong><pre>" + esc(sections || "(none)") + "</pre></div>"
        );
      });
    });
  }

  document.getElementById("log-refresh").addEventListener("click", loadLogs);
  document.getElementById("log-status").addEventListener("change", loadLogs);
  document.getElementById("log-sort").addEventListener("change", loadLogs);

  /* ---------- boot ---------- */
  loadStats();
  loadUsers();
  loadLogs();
})();

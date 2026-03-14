(function () {
  var form = document.getElementById("chat-form");
  var input = document.getElementById("message-input");
  var messagesEl = document.getElementById("messages");
  var sendBtn = document.getElementById("send-btn");
  var API_PREFIX = "/api/v1";

  function formatResponse(data) {
    if (data.error) {
      return { text: "Ошибка: " + data.error, type: "error" };
    }
    var parts = [];
    if (data.is_toxic) {
      var pct = Math.round((data.toxicity_score || 0) * 100);
      parts.push("Токсичность: " + pct + "%");
    }
    if (data.is_spam) {
      var pct = Math.round((data.spam_score || 0) * 100);
      parts.push("Спам: " + pct + "%");
    }
    if (parts.length === 0) {
      return { text: "Нормальное сообщение", type: "normal" };
    }
    if (data.is_toxic && data.is_spam) {
      return { text: parts.join(". "), type: "both" };
    }
    if (data.is_toxic) {
      return { text: parts[0], type: "toxic" };
    }
    return { text: parts[0], type: "spam" };
  }

  function appendMessage(text, className) {
    var div = document.createElement("div");
    div.className = "message " + className;
    div.textContent = text;
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function setLoading(loading) {
    sendBtn.disabled = loading;
  }

  function pollTask(taskId, maxAttempts, intervalMs) {
    return new Promise(function (resolve, reject) {
      var attempts = 0;
      function poll() {
        fetch(API_PREFIX + "/tasks/" + taskId)
          .then(function (res) {
            if (!res.ok) {
              if (res.status === 404) return resolve(null);
              return res.json().then(function (body) {
                reject(new Error(body.detail || res.statusText));
              }).catch(function () { reject(new Error(res.statusText)); });
            }
            return res.json();
          })
          .then(function (data) {
            if (data.status === "completed" && data.results && data.results.length > 0) {
              resolve(data.results[0]);
              return;
            }
            if (data.status === "failed") {
              resolve({ error: data.error || "Classification failed" });
              return;
            }
            attempts++;
            if (attempts >= maxAttempts) {
              resolve({ error: "Timeout waiting for result" });
              return;
            }
            setTimeout(poll, intervalMs);
          })
          .catch(reject);
      }
      poll();
    });
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    var text = (input.value || "").trim();
    if (!text) return;

    input.value = "";
    appendMessage(text, "message-user");
    setLoading(true);

    fetch(API_PREFIX + "/classify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text }),
    })
      .then(function (res) {
        if (!res.ok) {
          return res.json().then(function (body) {
            throw new Error(body.detail || res.statusText);
          }).catch(function () {
            throw new Error(res.statusText);
          });
        }
        return res.json();
      })
      .then(function (data) {
        var taskId = data.task_id;
        if (!taskId) {
          throw new Error("No task_id in response");
        }
        return pollTask(taskId, 60, 500);
      })
      .then(function (data) {
        if (!data) {
          appendMessage("Ошибка: задача не найдена", "message-system error");
          return;
        }
        var result = formatResponse(data);
        appendMessage(result.text, "message-system " + result.type);
      })
      .catch(function (err) {
        appendMessage("Ошибка: " + err.message, "message-system error");
      })
      .finally(function () {
        setLoading(false);
      });
  });
})();

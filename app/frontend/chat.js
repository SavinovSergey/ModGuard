(function () {
  var form = document.getElementById("chat-form");
  var input = document.getElementById("message-input");
  var messagesEl = document.getElementById("messages");
  var sendBtn = document.getElementById("send-btn");

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

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    var text = (input.value || "").trim();
    if (!text) return;

    input.value = "";
    appendMessage(text, "message-user");

    setLoading(true);
    fetch("/api/v1/classify", {
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

// script.js
const chatEl = document.getElementById("chat");
const msgInput = document.getElementById("message");
const sendBtn = document.getElementById("send");

let ws;

function addMessage(text, who="bot"){
  const div = document.createElement("div");
  div.className = "message " + (who === "user" ? "user" : "bot");
  div.textContent = text;
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function connect() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const wsUrl = `${protocol}://${location.host}/ws/chat`;
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    console.log("WebSocket conectado");
  };
  ws.onmessage = (ev) => {
    addMessage(ev.data, "bot");
  };
  ws.onclose = () => {
    console.log("WebSocket cerrado, reintentando en 3s...");
    setTimeout(connect, 3000);
  };
  ws.onerror = (e) => console.error("WebSocket error", e);
}

sendBtn.addEventListener("click", () => {
  const text = msgInput.value.trim();
  if (!text) return;
  addMessage(text, "user");
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(text);
  } else {
    addMessage("Error: WebSocket no está conectado", "bot");
  }
  msgInput.value = "";
});

msgInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    sendBtn.click();
  }
});

// conectar al cargar
connect();

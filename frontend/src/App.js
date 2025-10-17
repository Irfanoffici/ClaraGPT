import React, { useState, useRef, useEffect } from "react";
import "./App.css";

function App() {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState([]);
  const chatEndRef = useRef(null);

  const BACKEND_URL = "https://claragpt.vercel.app/api";

  // Auto-scroll when messages update
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async () => {
    if (!query) return;

    setMessages(prev => [...prev, { sender: "user", text: query }]);

    try {
      const res = await fetch(`${BACKEND_URL}/ask?q=${encodeURIComponent(query)}`);
      const data = await res.json();

      setMessages(prev => [
        ...prev,
        {
          sender: "bot",
          text: data.answer,
          citations: data.citations || []
        }
      ]);
    } catch (error) {
      setMessages(prev => [...prev, { sender: "bot", text: "⚠️ Error fetching response" }]);
    }

    setQuery("");
  };

  return (
    <div className="App">
      <h1>ClaraGPT Chat</h1>
      <div className="chatbox">
        {messages.map((m, i) => (
          <div key={i} className={m.sender === "user" ? "user-msg" : "bot-msg"}>
            <div>{m.text}</div>
            {m.citations && m.citations.length > 0 && (
              <div className="citations">
                {m.citations.map((c, j) => (
                  <div key={j} className="citation">{c}</div>
                ))}
              </div>
            )}
          </div>
        ))}
        <div ref={chatEndRef}></div>
      </div>
      <input
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && sendMessage()}
        placeholder="Ask a medical question..."
      />
      <button onClick={sendMessage}>Send</button>
    </div>
  );
}

export default App;

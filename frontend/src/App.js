import React, { useState } from "react";
import "./App.css";

function App() {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState([]);

  const sendMessage = async () => {
    if (!query) return;

    setMessages(prev => [...prev, { sender: "user", text: query }]);

    try {
     const res = await fetch(`https://claragpt.onrender.com/ask?q=${encodeURIComponent(query)}`);
      const data = await res.json();
      setMessages(prev => [...prev, { sender: "bot", text: data.answer }]);
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
            {m.text}
          </div>
        ))}
      </div>
      <input
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Ask a medical question..."
      />
      <button onClick={sendMessage}>Send</button>
    </div>
  );
}

export default App;

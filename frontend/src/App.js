import React, { useState, useRef, useEffect } from "react";
import "./App.css";

function App() {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState([]);
  const chatEndRef = useRef(null);

  // Scroll to bottom on new message
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async () => {
    if (!query.trim()) return;

    // Add user message
    setMessages((prev) => [...prev, { sender: "user", text: query }]);

    try {
      const res = await fetch(
        `https://claragpt.vercel.app/api/ask?q=${encodeURIComponent(query)}`
      );
      const data = await res.json();

      setMessages((prev) => [...prev, { sender: "bot", text: data.answer }]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: "⚠️ Error fetching response" },
      ]);
      console.error(error);
    }

    setQuery("");
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter") sendMessage();
  };

  return (
    <div className="App">
      <h1>ClaraGPT Chat</h1>
      <div className="chatbox">
        {messages.map((m, i) => (
          <div
            key={i}
            className={m.sender === "user" ? "user-msg" : "bot-msg"}
          >
            {m.text}
          </div>
        ))}
        <div ref={chatEndRef}></div>
      </div>
      <div className="input-area">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Ask a medical question..."
        />
        <button onClick={sendMessage}>Send</button>
      </div>
    </div>
  );
}

export default App;

"use client";

import { useEffect, useRef, useState } from "react";
import Header from "../components/ChatHeader";
import ChatMessage from "../components/ChatMessages";
import InputArea from "../components/ChatInput";
import { Message } from "../types";

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  const API_URL = "http://localhost:8000/chat";

  useEffect(() => {
    if (!localStorage.getItem("session_id")) {
      localStorage.setItem("session_id", "");
    }
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const sendMessage = async () => {
    if (!input.trim()) return;

    setLoading(true);
    const newMessages = [...messages, { role: "user", content: input }];
    setMessages(newMessages);
    setInput("");

    const sessionId = localStorage.getItem("session_id") || undefined;

    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: input, session_id: sessionId || null }),
    });

    const data = await res.json();
    if (data.session_id) localStorage.setItem("session_id", data.session_id);

    setMessages([...newMessages, { role: "agent", content: data.response }]);
    setLoading(false);
  };

  return (
    <main className="flex h-screen items-center justify-center">
      <div className="w-full max-w-3xl h-[90vh] bg-slate-900/70 rounded-2xl shadow-2xl flex flex-col">
        <Header />
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.length === 0 && !loading && (
            <div className="h-full flex items-center justify-center text-center">
              <p className="text-slate-400 text-lg italic">
                What business insight may I help you with today?
              </p>
            </div>
          )}
          {messages.map((msg, i) => (
            <ChatMessage key={i} message={msg} />
          ))}
          {loading && (
            <div className="text-slate-500 text-sm">Agent is thinking…</div>
          )}
          <div ref={bottomRef} />
        </div>
        <InputArea input={input} setInput={setInput} sendMessage={sendMessage} />
      </div>
    </main>
  );
}

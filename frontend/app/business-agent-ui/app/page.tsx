"use client";

import { useEffect, useRef, useState } from "react";
import ChatHeader from "./components/ChatHeader";
import ChatMessages from "./components/ChatMessages";
import ChatInput from "./components/ChatInput";
import { RefObject } from "react";

interface ChatMessagesProps {
  bottomRef: RefObject<HTMLDivElement | null>;
}


// Define Message type strictly
export type Message = {
  role: "user" | "agent";
  content: string;
};

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  // Ref for scrolling to bottom, can be null initially
  const bottomRef = useRef<HTMLDivElement>(null);

  const API_URL = "http://localhost:8000/chat";

  // Initialize session ID in localStorage if not present
  useEffect(() => {
    if (!localStorage.getItem("session_id")) {
      localStorage.setItem("session_id", "");
    }
  }, []);

  // Auto-scroll to bottom when messages or loading state changes
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  /** Sends user message to backend and appends agent response */
  const sendMessage = async () => {
    if (!input.trim()) return;

    setLoading(true);

    // Add user message, typecast role as literal
    const userMessage: Message = { role: "user", content: input };
    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setInput("");

    const sessionId = localStorage.getItem("session_id") || undefined;

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: input,
          session_id: sessionId || null,
        }),
      });

      const data: { response: string; session_id?: string } = await res.json();

      // Store new session ID if provided
      if (data.session_id) {
        localStorage.setItem("session_id", data.session_id);
      }

      // Add agent message, typecast role as literal
      const agentMessage: Message = { role: "agent", content: data.response };
      setMessages([...newMessages, agentMessage]);
    } catch (error) {
      console.error("Error sending message:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="flex h-screen items-center justify-center">
      <div className="w-full max-w-3xl h-[90vh] bg-slate-900/70 rounded-2xl shadow-2xl flex flex-col">
        <ChatHeader />
        <ChatMessages messages={messages} loading={loading} bottomRef={bottomRef} />
        <ChatInput input={input} setInput={setInput} sendMessage={sendMessage} />
      </div>
    </main>
  );
}

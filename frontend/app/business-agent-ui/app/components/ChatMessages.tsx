import React from "react";
import ReactMarkdown from "react-markdown";

type Message = {
  role: "user" | "agent";
  content: string;
};

type ChatMessagesProps = {
  messages: Message[];
  loading: boolean;
  bottomRef: React.RefObject<HTMLDivElement>;
};

/** Displays chat messages, welcome message, and loading state */
export default function ChatMessages({ messages, loading, bottomRef }: ChatMessagesProps) {
  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-4">
      
      {/* Welcome message if no messages yet */}
      {messages.length === 0 && !loading && (
        <div className="h-full flex items-center justify-center text-center">
          <p className="text-slate-400 text-lg italic">
            What business insight may I help you with today?
          </p>
        </div>
      )}

      {/* Render chat messages */}
      {messages.map((msg, i) => (
        <div
          key={i}
          className={`max-w-[80%] px-4 py-3 rounded-2xl text-sm leading-relaxed ${
            msg.role === "user"
              ? "ml-auto bg-emerald-400 text-black"
              : "mr-auto bg-slate-800 text-slate-200"
          }`}
        >
          <div className="prose prose-invert max-w-none">
            <ReactMarkdown>{msg.content}</ReactMarkdown>
          </div>
        </div>
      ))}

      {/* Loading indicator */}
      {loading && <div className="text-slate-500 text-sm">Agent is thinking…</div>}

      {/* Scroll anchor */}
      <div ref={bottomRef} />
    </div>
  );
}

import React from "react";

type ChatInputProps = {
  input: string;
  setInput: React.Dispatch<React.SetStateAction<string>>;
  sendMessage: () => void;
};

/** Input field and send button for chat */
export default function ChatInput({ input, setInput, sendMessage }: ChatInputProps) {
  return (
    <footer className="p-4 border-t border-slate-700 flex gap-3">
      <input
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && sendMessage()}
        placeholder="Ask about transactions, customers, policies…"
        className="
          flex-1
          rounded-xl
          bg-slate-100
          text-slate-900
          placeholder-slate-500
          px-4
          py-3
          text-sm
          shadow-inner
          focus:outline-none
          focus:ring-2
          focus:ring-emerald-400
        "
      />
      <button
        onClick={sendMessage}
        className="px-6 py-3 rounded-xl bg-emerald-500 text-black font-semibold hover:bg-emerald-400 transition"
      >
        Send
      </button>
    </footer>
  );
}

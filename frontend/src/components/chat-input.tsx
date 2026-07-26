"use client";

import { motion } from "motion/react";
import { Send } from "lucide-react";
import { useState, useRef } from "react";

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
  placeholder?: string;
  isLoading?: boolean;
  loadingText?: string;
}

export function ChatInput({
  onSend,
  disabled = false,
  placeholder = "Tell us what happened...",
  isLoading = false,
  loadingText = "Assistant is preparing the response...",
}: ChatInputProps) {
  const [input, setInput] = useState("");
  const [isFocused, setIsFocused] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleSend = () => {
    if (input.trim() && !disabled && !isLoading) {
      onSend(input);
      setInput("");
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <motion.div
      className="relative w-full"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: 0.1 }}
    >
      <div className="chat-glow relative flex items-center gap-2 px-4 py-3 rounded-lg border border-border bg-background focus-within:bg-card transition-all">
        <input
          ref={inputRef}
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          onFocus={() => setIsFocused(true)}
          onBlur={() => setIsFocused(false)}
          placeholder={placeholder}
          disabled={disabled || isLoading}
          className="flex-1 bg-transparent text-white placeholder:text-muted-foreground text-sm outline-none disabled:opacity-50"
        />

        <motion.button
          whileHover={{ scale: isFocused || input.trim() ? 1.05 : 1 }}
          whileTap={{ scale: isFocused || input.trim() ? 0.95 : 1 }}
          onClick={handleSend}
          disabled={!input.trim() || disabled || isLoading}
          className="p-2 rounded-md bg-primary hover:bg-accent text-primary-foreground disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          <Send className="h-4 w-4" />
        </motion.button>
      </div>

      {isLoading && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.2 }}
          className="mt-2 text-xs text-muted-foreground"
          role="status"
          aria-live="polite"
        >
          {loadingText}
        </motion.div>
      )}
    </motion.div>
  );
}

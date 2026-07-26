"use client";

import { useEffect, useState, useRef, useCallback, useMemo } from "react";
import { motion } from "motion/react";
import { Plus, Trash2, MessageSquare, Eye } from "lucide-react";
import { ChatInput } from "@/components/chat-input";
import { UserMessage, AgentMessage, SkeletonLoader, ChatMessageData } from "@/components/chat-message";
import { ComplaintFeaturePanel } from "@/components/complaint/ComplaintFeaturePanel";
import { FIRDraftPreview } from "@/components/complaint/FIRDraftPreview";
import { deriveComplaintInsight, type ComplaintInsight } from "@/lib/complaint-intelligence";
import {
  chatComplaintStream,
  listChatSessions,
  getChatSession,
  deleteChatSession,
  fileComplaintFromChat,
  type ChatSession,
} from "@/lib/types";
import { toast } from "sonner";

const READY_TO_FILE_MARKER = "Great! I have all the information I need. Ready to file your complaint?";

function createSessionId(): string {
  return `session_${Date.now()}_${Math.random().toString(36).slice(2, 11)}`;
}

export default function ChatPage() {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string>("");
  const [messages, setMessages] = useState<ChatMessageData[]>([]);
  const [loading, setLoading] = useState(false);
  const [streamingStatus, setStreamingStatus] = useState("Preparing assistant response");
  const [isInitializing, setIsInitializing] = useState(true);
  const [readyToFile, setReadyToFile] = useState(false);
  const [submittingComplaint, setSubmittingComplaint] = useState(false);
  const [showDraftPreview, setShowDraftPreview] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Derive complaint intelligence from messages
  const lastExtractedFields = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      const msg = messages[i];
      if (msg.role === "agent" && msg.extracted_data && Object.keys(msg.extracted_data).length > 0) {
        return msg.extracted_data;
      }
    }
    return {};
  }, [messages]);

  const complaintInsight: ComplaintInsight = useMemo(
    () => deriveComplaintInsight(lastExtractedFields, messages),
    [lastExtractedFields, messages],
  );

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const createNewSession = useCallback(() => {
    setActiveSessionId(createSessionId());
    setMessages([]);
    setReadyToFile(false);
    setShowDraftPreview(false);
  }, []);

  const loadMessages = useCallback(async (sessionId: string) => {
    try {
      const data = await getChatSession(sessionId);
      const formattedMessages: ChatMessageData[] = data.messages.map((msg) => ({
        role: msg.role,
        content: msg.content,
        timestamp: msg.timestamp,
        extracted_data: msg.extracted_data,
      }));
      setMessages(formattedMessages);

      const lastMessage = formattedMessages[formattedMessages.length - 1];
      setReadyToFile(
        !!lastMessage && lastMessage.role === "agent" && lastMessage.content === READY_TO_FILE_MARKER
      );
    } catch (error) {
      console.error("Failed to load messages:", error);
      toast.error("Failed to load chat history");
    }
  }, []);

  const refreshSessions = useCallback(async () => {
    try {
      const data = await listChatSessions();
      setSessions(data.sessions || []);
      return data.sessions || [];
    } catch (error) {
      console.error("Failed to load sessions:", error);
      return [];
    }
  }, []);

  // Load sessions on mount
  useEffect(() => {
    const init = async () => {
      const loadedSessions = await refreshSessions();
      const mostRecent = loadedSessions.find((s) => !s.is_filed);
      if (mostRecent) {
        setActiveSessionId(mostRecent.id);
        await loadMessages(mostRecent.id);
      } else {
        createNewSession();
      }
      setIsInitializing(false);
    };
    init();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleSendMessage = async (content: string) => {
    if (!activeSessionId || !content.trim()) return;

    setMessages((prev) => [...prev, { role: "user", content, timestamp: new Date().toISOString() }]);
    setStreamingStatus("Preparing assistant response");
    setLoading(true);

    try {
      const chatResponse = await chatComplaintStream(activeSessionId, content, (event) => {
        if (event.event === "status") {
          setStreamingStatus(event.message);
        }
      });

      setMessages((prev) => [
        ...prev,
        {
          role: "agent",
          content: chatResponse.agent_message,
          timestamp: new Date().toISOString(),
          extracted_data: chatResponse.collected_fields,
          suggested_followups: chatResponse.suggested_followups,
        },
      ]);

      setReadyToFile(!!chatResponse.ready_to_file);
      await refreshSessions();
    } catch (error) {
      console.error("Error sending message:", error);
      toast.error("Failed to send message. Please try again.");
    } finally {
      setLoading(false);
      setStreamingStatus("Preparing assistant response");
    }
  };

  const handleFileComplaint = async () => {
    if (!activeSessionId) return;

    setSubmittingComplaint(true);
    try {
      const complaint = await fileComplaintFromChat(activeSessionId);
      toast.success(`Complaint #${complaint.id} filed successfully!`);
      await refreshSessions();
      setShowDraftPreview(false);
      createNewSession();
    } catch (error) {
      console.error("Error filing complaint:", error);
      toast.error("Failed to file complaint. Please try again.");
    } finally {
      setSubmittingComplaint(false);
    }
  };

  const handleDeleteSession = async (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation();

    if (!confirm("Are you sure you want to delete this chat session?")) return;

    try {
      await deleteChatSession(sessionId);
      setSessions((prev) => prev.filter((s) => s.id !== sessionId));
      if (activeSessionId === sessionId) {
        createNewSession();
      }
      toast.success("Chat session deleted");
    } catch (error) {
      console.error("Error deleting session:", error);
      toast.error("Failed to delete session");
    }
  };

  const handleSelectSession = (sessionId: string) => {
    if (sessionId === activeSessionId) return;
    setActiveSessionId(sessionId);
    setMessages([]);
    setReadyToFile(false);
    loadMessages(sessionId);
  };

  if (isInitializing) {
    return (
      <div className="h-screen bg-background flex items-center justify-center">
        <motion.div animate={{ opacity: [0.5, 1, 0.5] }} transition={{ duration: 2, repeat: Infinity }}>
          <MessageSquare className="h-12 w-12 text-accent" />
        </motion.div>
      </div>
    );
  }

  return (
    <div className="h-[calc(100vh-3.5rem)] bg-background flex">
      {/* Sidebar - Session List */}
      <div className="w-64 border-r border-sidebar-border flex flex-col bg-sidebar overflow-hidden">
        <div className="p-4 border-b border-sidebar-border">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={createNewSession}
            className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg bg-primary hover:bg-accent text-primary-foreground text-sm font-medium transition-colors"
          >
            <Plus className="h-4 w-4" />
            New Chat
          </motion.button>
        </div>

        <div className="flex-1 overflow-y-auto space-y-1 p-2">
          {sessions.map((session) => (
            <motion.div
              key={session.id}
              role="button"
              tabIndex={0}
              onClick={() => handleSelectSession(session.id)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault()
                  handleSelectSession(session.id)
                }
              }}
              whileHover={{ x: 4 }}
              className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-all flex items-center justify-between group ${
                activeSessionId === session.id
                  ? "border border-accent bg-primary text-white"
                  : "border border-transparent text-muted-foreground hover:border-sidebar-border hover:bg-sidebar-accent hover:text-white"
              }`}
            >
              <span className="truncate text-xs">
                {session.is_filed
                  ? `✓ #${session.complaint_id ?? session.complaint_id_fk}`
                  : session.title || `Chat ${session.id.slice(0, 8)}`}
              </span>
              {!session.is_filed && (
                <button
                  onClick={(e) => handleDeleteSession(session.id, e)}
                  className="opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  <Trash2 className="h-3 w-3 text-destructive hover:text-destructive/80" />
                </button>
              )}
            </motion.div>
          ))}
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.length === 0 && !loading ? (
            <div className="h-full flex flex-col items-center justify-center text-center">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
              >
                <MessageSquare className="h-16 w-16 text-muted-foreground/30 mx-auto mb-4" />
                <h2 className="text-2xl font-bold text-foreground mb-2">Welcome to Complaint Filing</h2>
                <p className="text-muted-foreground max-w-md">
                  Start by telling us what happened. I&apos;m here to help guide you through the process.
                </p>
              </motion.div>
            </div>
          ) : (
            <>
              {messages.map((msg, idx) =>
                msg.role === "user" ? (
                  <UserMessage key={idx} content={msg.content} timestamp={msg.timestamp} />
                ) : (
                  <AgentMessage
                    key={idx}
                    content={msg.content}
                    timestamp={msg.timestamp}
                    suggested_followups={msg.suggested_followups}
                    collected_fields={msg.extracted_data}
                  />
                )
              )}
              {loading && <SkeletonLoader status={streamingStatus} />}
            </>
          )}

          {/* Complaint Intelligence Panel */}
          {messages.length > 0 && !loading && (
            <ComplaintFeaturePanel insight={complaintInsight} />
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Ready to File Alert */}
        {readyToFile && !loading && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="px-6 py-4 border-t border-border bg-card"
          >
            <div className="flex items-center justify-between gap-4">
              <div>
                <p className="text-sm font-medium text-foreground">All set!</p>
                <p className="text-xs text-muted-foreground">Your complaint is ready to be filed.</p>
              </div>
              <div className="flex items-center gap-2">
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => setShowDraftPreview(true)}
                  className="flex items-center gap-1.5 px-3 py-2 border border-border rounded-lg text-sm text-muted-foreground hover:text-foreground hover:bg-sidebar-accent transition-colors"
                >
                  <Eye className="h-3.5 w-3.5" />
                  Preview Draft
                </motion.button>
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={handleFileComplaint}
                  disabled={submittingComplaint}
                  className="px-4 py-2 bg-primary hover:bg-accent text-primary-foreground rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
                >
                  {submittingComplaint ? "Filing..." : "File Complaint"}
                </motion.button>
              </div>
            </div>
          </motion.div>
        )}

        {/* Draft Preview Drawer */}
        <FIRDraftPreview
          open={showDraftPreview}
          onClose={() => setShowDraftPreview(false)}
          draftData={complaintInsight.draftData}
          canFile={readyToFile}
          isFiling={submittingComplaint}
          onFileComplaint={handleFileComplaint}
        />

        {/* Input Area */}
        <div className="px-6 py-4 border-t border-border bg-sidebar">
          <ChatInput
            onSend={handleSendMessage}
            disabled={loading || submittingComplaint}
            placeholder="Type your message here..."
            isLoading={loading}
            loadingText={streamingStatus}
          />
        </div>
      </div>
    </div>
  );
}

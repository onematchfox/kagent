import React, { useEffect, useMemo, useRef } from "react";
import { Message, TextPart } from "@a2a-js/sdk";
import ToolDisplay, { ToolCallStatus } from "@/components/ToolDisplay";
import AgentCallDisplay, { AgentCallStatus } from "@/components/chat/AgentCallDisplay";
import { isAgentToolName } from "@/lib/utils";
import { ADKMetadata, ProcessedToolResultData, ToolResponseData, normalizeToolResultToText } from "@/lib/messageHandlers";
import { FunctionCall } from "@/types";
import { ToolDecisionType, type ToolDecisionChildContext, KAGENT_HITL_DECISION_TYPE_DENY, getToolApprovalIdFromPart, parseToolApprovalFromString } from "@/lib/hitl";

// Convert ToolCallStatus to AgentCallStatus (pending_approval doesn't apply to agent calls)
const toAgentStatus = (status: ToolCallStatus): AgentCallStatus => {
  if (status === "pending_approval") {
    return "requested";
  }
  return status;
};

interface ToolCallDisplayProps {
  currentMessage: Message;
  allMessages: Message[];
  onToolDecision?: (toolId: string, decision: ToolDecisionType, childContext?: ToolDecisionChildContext) => void;
  decidedTools?: Map<string, ToolDecisionType>;
  isStreaming?: boolean;
}

interface ConfirmationInfo {
  id: string;
  hint?: string;
  /** True when backend is ready for user input (final input-required event received) */
  awaitingInput?: boolean;
}

interface ToolCallState {
  id: string;
  call: FunctionCall;
  result?: {
    content: string;
    is_error?: boolean;
  };
  status: ToolCallStatus;
  /** Confirmation info if this tool is pending approval */
  confirmationInfo?: ConfirmationInfo;
  /** True when backend is ready for user input */
  awaitingInput?: boolean;
}

// Create a global cache to track tool calls across components
const toolCallCache = new Map<string, boolean>();

// Helper functions to work with A2A SDK Messages
const isToolCallRequestMessage = (message: Message): boolean => {
  // Check data parts for kagent_type first
  const hasDataParts = message.parts?.some(part => {
    if (part.kind === "data" && part.metadata) {
      const partMetadata = part.metadata as ADKMetadata;
      return partMetadata?.kagent_type === "function_call";
    }
    return false;
  }) || false;
  
  // Fallback to streaming format check
  if (!hasDataParts) {
    const metadata = message.metadata as ADKMetadata;
    return metadata?.originalType === "ToolCallRequestEvent";
  }
  
  return hasDataParts;
};

const isToolCallExecutionMessage = (message: Message): boolean => {
  const hasDataParts = message.parts?.some(part => {
    if (part.kind === "data" && part.metadata) {
      const partMetadata = part.metadata as ADKMetadata;
      return partMetadata?.kagent_type === "function_response";
    }
    return false;
  }) || false;
  
  // Fallback to streaming format check
  if (!hasDataParts) {
    const metadata = message.metadata as ADKMetadata;
    return metadata?.originalType === "ToolCallExecutionEvent";
  }
  
  return hasDataParts;
};

const isToolCallSummaryMessage = (message: Message): boolean => {
  const metadata = message.metadata as ADKMetadata;
  return metadata?.originalType === "ToolCallSummaryMessage";
};

const extractToolCallRequests = (message: Message): FunctionCall[] => {
  if (!isToolCallRequestMessage(message)) return [];
  
  // Check for stored task format first (data parts)
  const dataParts = message.parts?.filter(part => part.kind === "data") || [];
  const functionCalls: FunctionCall[] = [];
  
  for (const part of dataParts) {
    if (part.metadata) {
      const partMetadata = part.metadata as ADKMetadata;
      if (partMetadata?.kagent_type === "function_call") {
        const data = part.data as unknown as FunctionCall;
        functionCalls.push({
          id: data.id,
          name: data.name,
          args: data.args
        });
      }
    }
  }
  
  // If we found function calls in data parts, return them
  if (functionCalls.length > 0) {
    return functionCalls;
  }
  
  // Try streaming format (metadata or text content)
  const textParts = message.parts?.filter(part => part.kind === "text") || [];
  const content = textParts.map(part => (part as TextPart).text).join("");
  
  try {
    // Tool call data might be stored as JSON in content or metadata
    const metadata = message.metadata as ADKMetadata;
    const toolCallData = metadata?.toolCallData || JSON.parse(content || "[]");
    return Array.isArray(toolCallData) ? toolCallData : [];
  } catch {
    return [];
  }
};

const extractToolCallResults = (message: Message): ProcessedToolResultData[] => {
  if (!isToolCallExecutionMessage(message)) return [];
  
  // Check for stored task format first (data parts)
  const dataParts = message.parts?.filter(part => part.kind === "data") || [];
  const toolResults: ProcessedToolResultData[] = [];
  
  for (const part of dataParts) {
    if (part.metadata) {
      const partMetadata = part.metadata as ADKMetadata;
      if (partMetadata?.kagent_type === "function_response") {
        const data = part.data as unknown as ToolResponseData;
        // Extract normalized content from the result (supports string/object/array)
        const textContent = normalizeToolResultToText(data);
        
        toolResults.push({
          call_id: data.id,
          name: data.name,
          content: textContent,
          is_error: data.response?.isError || false
        });
      }
    }
  }
  
  // If we found tool results in data parts, return them
  if (toolResults.length > 0) {
    return toolResults;
  }
  
  // Try streaming format (metadata or text content)
  const textParts = message.parts?.filter(part => part.kind === "text") || [];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const content = textParts.map(part => (part as any).text).join("");
  
  try {
    const metadata = message.metadata as ADKMetadata;
    const resultData = metadata?.toolResultData || JSON.parse(content || "[]");
    return Array.isArray(resultData) ? resultData : [];
  } catch {
    return [];
  }
};

// Extract confirmation requests from tool_approval interrupt data
// Returns a map of original tool ID -> confirmation info
const extractConfirmationRequests = (messages: Message[]): Map<string, ConfirmationInfo> => {
  const confirmations = new Map<string, ConfirmationInfo>();

  for (const message of messages) {
    const messageMetadata = message.metadata as Record<string, unknown> | undefined;
    const awaitingInput = messageMetadata?.awaiting_input === true;

    for (const part of message.parts ?? []) {
      const id = getToolApprovalIdFromPart(part);
      if (id) {
        const existing = confirmations.get(id);
        confirmations.set(id, {
          id,
          awaitingInput: awaitingInput || existing?.awaitingInput,
        });
      }
    }
  }

  return confirmations;
};

const ToolCallDisplay = ({ 
  currentMessage, 
  allMessages,
  onToolDecision,
  decidedTools = new Map(),
  isStreaming = false,
}: ToolCallDisplayProps) => {
  // Track which call IDs this component instance registered in the cache
  const registeredIdsRef = useRef<Set<string>>(new Set());

  // Compute owned call IDs based on current message (memoized)
  const ownedCallIds = useMemo(() => {
    const currentOwnedIds = new Set<string>();
    if (isToolCallRequestMessage(currentMessage)) {
      const requests = extractToolCallRequests(currentMessage);
      for (const request of requests) {
        if (request.id && !toolCallCache.has(request.id)) {
          currentOwnedIds.add(request.id);
          toolCallCache.set(request.id, true);
        }
      }
    }
    return currentOwnedIds;
  }, [currentMessage]);

  // Update ref and handle cleanup
  useEffect(() => {
    // Store current owned IDs for cleanup
    registeredIdsRef.current = ownedCallIds;

    return () => {
      registeredIdsRef.current.forEach(id => {
        toolCallCache.delete(id);
      });
    };
  }, [ownedCallIds]);

  // Compute tool calls based on all messages and owned IDs (memoized)
  const toolCalls = useMemo(() => {
    if (ownedCallIds.size === 0) {
      return new Map<string, ToolCallState>();
    }

    const newToolCalls = new Map<string, ToolCallState>();
    
    // Extract all confirmation requests to apply to tool calls
    const confirmationRequests = extractConfirmationRequests(allMessages);

    // First pass: collect all tool call requests that this component owns
    for (const message of allMessages) {
      if (isToolCallRequestMessage(message)) {
        const requests = extractToolCallRequests(message);
        for (const request of requests) {
          if (request.id && ownedCallIds.has(request.id)) {
            newToolCalls.set(request.id, {
              id: request.id,
              call: request,
              status: "requested"
            });
          }
        }
      }
    }

    // Second pass: update with execution results
    for (const message of allMessages) {
      if (isToolCallExecutionMessage(message)) {
        const results = extractToolCallResults(message);
        for (const result of results) {
          if (result.call_id && newToolCalls.has(result.call_id)) {
            const existingCall = newToolCalls.get(result.call_id)!;
            
            // If this tool has a pending confirmation request and no decision has been made,
            // any result is an intermediate "awaiting approval" response - skip it
            // EXCEPT for child agent HITL where the result contains the tool_approval data
            const hasPendingConfirmation = confirmationRequests.has(result.call_id);
            const hasDecision = decidedTools.has(result.call_id);
            const hasChildHitl = result.content && parseToolApprovalFromString(result.content);
            
            if (hasPendingConfirmation && !hasDecision && !hasChildHitl) {
              // Don't set result - this is just an intermediate status
              // Keep the tool in "requested" state until proper confirmation data arrives
              continue;
            }
            
            existingCall.result = {
              content: result.content,
              is_error: result.is_error
            };
            existingCall.status = "executing";
          }
        }
      }
    }

    // Third pass: mark completed calls using summary messages
    // We need to be careful here to avoid showing "Completed" before confirmation status arrives.
    // Only auto-complete when:
    // 1. A summary message exists (explicit completion signal), OR
    // 2. Not streaming (loading from history)
    // Even then, skip tools that have pending confirmations or decisions.
    let summaryMessageEncountered = false;
    for (const message of allMessages) {
      if (isToolCallSummaryMessage(message)) {
        summaryMessageEncountered = true;
        break; 
      }
    }

    // Only auto-complete if we have a summary message OR we're loading from history (not streaming)
    // During active streaming without summary, tools stay in "executing" to avoid
    // briefly showing "Completed" before a pending_approval status arrives
    if (summaryMessageEncountered || !isStreaming) {
      newToolCalls.forEach((call, id) => {
        // Only update owned calls that are in 'executing' state and have a result
        if (call.status === "executing" && call.result && ownedCallIds.has(id)) {
          // Don't auto-complete if this tool has a pending confirmation or decision
          // The fourth pass will set the correct status
          const hasPendingConfirmation = confirmationRequests.has(id);
          const hasDecision = decidedTools.has(id);
          if (hasPendingConfirmation || hasDecision) {
            // Skip - fourth pass will handle this
            return;
          }
          call.status = "completed";
        }
      });
    } else if (!isStreaming) {
      // For historical/stored tasks (not actively streaming), auto-complete tool calls that have results
      // During streaming, keep in "executing" until we get explicit completion or confirmation status
      newToolCalls.forEach((call, id) => {
        if (call.status === "executing" && call.result && ownedCallIds.has(id)) {
          // Don't auto-complete if this tool has a pending decision
          // The fourth pass will set the correct status based on the decision type
          const hasDecision = decidedTools.has(id);
          if (hasDecision) {
            // Skip - fourth pass will handle the final status based on approve/deny
            return;
          }
          call.status = "completed";
        }
      });
    }
    // During active streaming without summary messages, tools stay in "executing" state
    // until fourth pass sets pending_approval or streaming ends
    
    // Fourth pass: apply pending_approval or denied status for tools with confirmation requests
    confirmationRequests.forEach((confirmInfo, id) => {
      if (newToolCalls.has(id)) {
        const toolCall = newToolCalls.get(id)!;
        
        // Check if user has already made a decision for this tool
        const decision = decidedTools.get(id);
        
        if (decision) {
          // User made a decision (approve or deny)
          const hasInterruptData = toolCall.result?.content && parseToolApprovalFromString(toolCall.result.content);
          const hasRealResult = toolCall.result?.content && !hasInterruptData;
          
          if (hasRealResult) {
            // Real result arrived - show final status
            toolCall.status = decision === KAGENT_HITL_DECISION_TYPE_DENY ? "denied" : "completed";
          } else if (isStreaming) {
            // Still streaming, waiting for response - show executing
            toolCall.status = "executing";
            if (hasInterruptData) {
              toolCall.result = undefined;
            }
          } else {
            // Loading from history - show final status
            toolCall.status = decision === KAGENT_HITL_DECISION_TYPE_DENY ? "denied" : "completed";
            if (hasInterruptData) {
              toolCall.result = undefined;
            }
          }
          toolCall.confirmationInfo = undefined;
          toolCall.awaitingInput = undefined;
        } else {
          // No decision yet - show pending approval
          toolCall.status = "pending_approval";
          toolCall.confirmationInfo = confirmInfo;
          // Track whether backend is ready for user input
          toolCall.awaitingInput = confirmInfo.awaitingInput;
          // For child agent HITL, preserve the result so AgentCallDisplay can detect tool_approval
          // For regular HITL, clear the stale intermediate "awaiting approval" result
          const hasChildHitl = toolCall.result?.content && parseToolApprovalFromString(toolCall.result.content);
          if (!hasChildHitl) {
            toolCall.result = undefined;
          }
        }
      }
    });

    return newToolCalls;
  }, [allMessages, ownedCallIds, decidedTools, isStreaming]);

  // If no tool calls to display for this message, return null
  const currentDisplayableCalls = Array.from(toolCalls.values()).filter(call => ownedCallIds.has(call.id));
  if (currentDisplayableCalls.length === 0) return null;

  return (
    <div className="space-y-2">
      {currentDisplayableCalls.map(toolCall => (
        isAgentToolName(toolCall.call.name) ? (
          <AgentCallDisplay
            key={toolCall.id}
            call={toolCall.call}
            result={toolCall.result}
            status={toAgentStatus(toolCall.status)}
            isError={toolCall.result?.is_error}
            onToolDecision={onToolDecision}
            awaitingInput={toolCall.awaitingInput}
            isStreaming={isStreaming}
          />
        ) : (
          <ToolDisplay
            key={toolCall.id}
            call={toolCall.call}
            result={toolCall.result}
            status={toolCall.status}
            isError={toolCall.result?.is_error}
            onDecision={(id, decision) => onToolDecision?.(id, decision)}
            confirmationHint={toolCall.confirmationInfo?.hint}
            awaitingInput={toolCall.awaitingInput}
            isStreaming={isStreaming}
          />
        )
      ))}
    </div>
  );
};

export default ToolCallDisplay;

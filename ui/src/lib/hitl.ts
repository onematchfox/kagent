/**
 * Human-in-the-Loop (HITL) constants, types, and utilities for tool approval workflows.
 * These must match the backend constants in kagent-core/a2a/_consts.py
 * 
 * The UI only handles the generic interrupt format (interrupt_type/action_requests).
 * Framework-specific formats (e.g., ADK's adk_request_confirmation) are converted
 * to this generic format at the backend.
 */

// Interrupt type for tool approval requests
export const KAGENT_HITL_INTERRUPT_TYPE_TOOL_APPROVAL = "tool_approval";

// Part metadata key set by backend when a function_response contains tool_approval
// (enables input_required without parsing content). Value from get_kagent_metadata_key("contains_tool_approval").
export const KAGENT_METADATA_CONTAINS_TOOL_APPROVAL = "kagent_contains_tool_approval";

// Decision type key in DataPart
export const KAGENT_HITL_DECISION_TYPE_KEY = "decision_type";

// Decision values
export const KAGENT_HITL_DECISION_TYPE_APPROVE = "approve";
export const KAGENT_HITL_DECISION_TYPE_DENY = "deny";
export const KAGENT_HITL_DECISION_TYPE_REJECT = "reject";

// Type for a single tool approval request (matches backend ToolApprovalRequest)
export interface ToolApprovalRequest {
  name: string;
  args: Record<string, unknown>;
  id?: string;
}

// Type for tool approval interrupt data stored in DataPart
export interface ToolApprovalInterruptData {
  interrupt_type: typeof KAGENT_HITL_INTERRUPT_TYPE_TOOL_APPROVAL;
  action_requests: ToolApprovalRequest[];
  review_configs?: unknown[];
}

// Type for per-tool decisions
export type ToolDecisionType = typeof KAGENT_HITL_DECISION_TYPE_APPROVE | typeof KAGENT_HITL_DECISION_TYPE_DENY;

// Optional context when the decision is for a child agent's tool (multi-agent HITL)
export interface ToolDecisionChildContext {
  childAgentName: string;
  parentCallId: string;
}

// Type for a tool decision sent back to the backend
export interface ToolDecision {
  decision_type: ToolDecisionType;
  tool_id?: string;
}

/**
 * Check if a DataPart contains tool approval interrupt data
 */
export function isToolApprovalInterrupt(data: unknown): data is ToolApprovalInterruptData {
  if (!data || typeof data !== "object") return false;
  const obj = data as Record<string, unknown>;
  return obj.interrupt_type === KAGENT_HITL_INTERRUPT_TYPE_TOOL_APPROVAL && Array.isArray(obj.action_requests);
}

/**
 * Extract tool approval requests from message parts
 */
export function extractToolApprovalRequests(parts: Array<{ kind: string; data?: unknown }>): ToolApprovalRequest[] {
  for (const part of parts) {
    if (part.kind === "data" && isToolApprovalInterrupt(part.data)) {
      return part.data.action_requests;
    }
  }
  return [];
}

/**
 * Check if a DataPart contains a tool decision
 */
export function hasToolDecision(data: unknown): data is ToolDecision {
  if (!data || typeof data !== "object") return false;
  const obj = data as Record<string, unknown>;
  return obj.decision_type === KAGENT_HITL_DECISION_TYPE_APPROVE || obj.decision_type === KAGENT_HITL_DECISION_TYPE_DENY;
}

/**
 * Extract all tool decisions from an array of messages.
 * This is used when loading session history to populate the decidedTools state.
 * Only extracts per-tool decisions (those with tool_id).
 */
export function extractToolDecisionsFromMessages(messages: Array<{ role?: string; parts?: Array<{ kind: string; data?: unknown }> }>): Map<string, ToolDecisionType> {
  const decisions = new Map<string, ToolDecisionType>();
  
  for (const message of messages) {
    if (message.role !== "user") continue; // Only user messages contain tool decisions
    if (!message.parts) continue;
    
    for (const part of message.parts) {
      if (part.kind === "data" && hasToolDecision(part.data) && part.data.tool_id) {
        // For child agent HITL, use parent_call_id as key (if available)
        // This matches how decidedTools is keyed in the UI
        const data = part.data as { parent_call_id?: string; tool_id: string; decision_type: ToolDecisionType };
        const key = data.parent_call_id || data.tool_id;
        decisions.set(key, data.decision_type);
      }
    }
  }
  
  return decisions;
}

// Message type for helper functions (minimal interface for what we need)
type MessageWithParts = { 
  parts?: Array<{ kind: string; data?: unknown; metadata?: unknown }>; 
  metadata?: unknown;
};

/**
 * Get the tool call ID that needs confirmation from a single part, or null.
 * We approve one tool at a time; direct format has action_requests array but we use the first.
 */
export function getToolApprovalIdFromPart(
  part: { kind: string; data?: unknown; metadata?: unknown }
): string | null {
  if (part.kind !== "data" || !part.data) return null;
  const data = part.data as Record<string, unknown>;

  // Direct format: { interrupt_type: "tool_approval", action_requests: [...] }
  if (isToolApprovalInterrupt(data)) {
    const requests = (data.action_requests as ToolApprovalRequest[]) ?? [];
    const first = requests[0]?.id;
    return first ?? null;
  }

  // Wrapped format: backend sets contains_tool_approval when response has tool approval
  const responseData = data as { id?: string; response?: { result?: unknown } };
  if (responseData?.id && responseData?.response?.result) {
    const meta = part.metadata as Record<string, unknown> | undefined;
    if (meta?.[KAGENT_METADATA_CONTAINS_TOOL_APPROVAL] === true) return responseData.id;
  }
  return null;
}

/**
 * Check if a message contains a tool approval request.
 */
export function hasToolApprovalRequest(msg: MessageWithParts): boolean {
  return msg.parts?.some((part) => getToolApprovalIdFromPart(part) !== null) ?? false;
}

/**
 * Check if data parts contain function call or response metadata
 */
export function hasFunctionCallOrResponse(dataParts: Array<{ metadata?: unknown }>): boolean {
  return dataParts.some(part => {
    const meta = part.metadata as { kagent_type?: string } | undefined;
    return meta?.kagent_type === "function_call" || meta?.kagent_type === "function_response";
  });
}

/**
 * Get all invocation IDs from messages that have tool approval requests.
 * Used to filter out intermediate text messages from HITL flows.
 */
export function getToolApprovalInvocationIds(messages: MessageWithParts[]): Set<string> {
  const ids = new Set<string>();
  for (const msg of messages) {
    if (hasToolApprovalRequest(msg)) {
      const meta = msg.metadata as { kagent_invocation_id?: string } | undefined;
      if (meta?.kagent_invocation_id) {
        ids.add(meta.kagent_invocation_id);
      }
    }
  }
  return ids;
}

/**
 * Try to parse tool_approval interrupt data from a string (e.g., AgentTool output).
 * Returns null if the string doesn't contain valid tool_approval data.
 */
export function parseToolApprovalFromString(content: string): ToolApprovalInterruptData | null {
  if (!content) return null;
  
  try {
    // Try direct JSON parse
    const parsed = JSON.parse(content);
    if (isToolApprovalInterrupt(parsed)) {
      return parsed;
    }
  } catch {
    // Not valid JSON, try to find JSON embedded in text
    const jsonMatch = content.match(/\{[\s\S]*"interrupt_type"\s*:\s*"tool_approval"[\s\S]*\}/);
    if (jsonMatch) {
      try {
        const parsed = JSON.parse(jsonMatch[0]);
        if (isToolApprovalInterrupt(parsed)) {
          return parsed;
        }
      } catch {
        // Ignore parse errors
      }
    }
  }
  
  return null;
}

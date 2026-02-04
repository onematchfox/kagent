import { useMemo, useState } from "react";
import { FunctionCall } from "@/types";
import { Card, CardHeader, CardTitle, CardContent, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { convertToUserFriendlyName } from "@/lib/utils";
import { ChevronDown, ChevronUp, MessageSquare, Loader2, AlertCircle, CheckCircle, ShieldAlert, Check, X } from "lucide-react";
import KagentLogo from "../kagent-logo";
import { parseToolApprovalFromString, ToolApprovalRequest, ToolDecisionType, type ToolDecisionChildContext, KAGENT_HITL_DECISION_TYPE_APPROVE, KAGENT_HITL_DECISION_TYPE_DENY } from "@/lib/hitl";

export type AgentCallStatus = "requested" | "executing" | "completed" | "denied";

interface AgentCallDisplayProps {
  call: FunctionCall;
  result?: {
    content: string;
    is_error?: boolean;
    hasChildHitl?: boolean;  // True if this is child agent HITL (set by backend)
  };
  status?: AgentCallStatus;
  isError?: boolean;
  onToolDecision?: (toolId: string, decision: ToolDecisionType, childContext?: ToolDecisionChildContext) => void;
  awaitingInput?: boolean;
  isStreaming?: boolean;
}

const AgentCallDisplay = ({ 
  call, 
  result, 
  status = "requested", 
  isError = false,
  onToolDecision,
  awaitingInput,
  isStreaming = false,
}: AgentCallDisplayProps) => {
  const [areInputsExpanded, setAreInputsExpanded] = useState(false);
  const [areResultsExpanded, setAreResultsExpanded] = useState(false);

  const agentDisplay = useMemo(() => convertToUserFriendlyName(call.name), [call.name]);
  const hasResult = result !== undefined;
  
  // Check if the result contains a child agent's tool_approval request
  // Use hasChildHitl flag from backend if available, otherwise parse content
  const childToolApproval = useMemo(() => {
    if (!result?.content) return null;
    // If hasChildHitl flag is set by backend, trust it and parse
    if (result.hasChildHitl) {
      return parseToolApprovalFromString(result.content);
    }
    // Fallback: try to parse anyway
    return parseToolApprovalFromString(result.content);
  }, [result?.content, result?.hasChildHitl]);
  
  // Get the first action request for display
  const childToolRequest: ToolApprovalRequest | undefined = childToolApproval?.action_requests?.[0];
  
  // Determine if this is awaiting child approval
  // hasChildHitl flag indicates backend detected child HITL in this response
  const isChildApprovalNeeded = (result?.hasChildHitl || childToolApproval !== null) && status !== "completed";
  
  // Button state - disabled during streaming until backend is ready
  const isButtonDisabled = isStreaming && !awaitingInput;

  const getStatusDisplay = () => {
    if (isError && status === "executing") {
      return (
        <>
          <AlertCircle className="w-3 h-3 inline-block mr-2 text-red-500" />
          Error
        </>
      );
    }
    
    // Show child approval status if detected
    if (isChildApprovalNeeded) {
      return (
        <>
          <ShieldAlert className="w-3 h-3 inline-block mr-2 text-amber-500" />
          Child tool awaiting approval
        </>
      );
    }
    
    switch (status) {
      case "executing":
      case "requested":
        return (
          <>
            <KagentLogo className="w-3 h-3 inline-block mr-2 text-blue-500" />
            Delegating
          </>
        );
      case "completed":
        if (isError) {
          return (
            <>
              <AlertCircle className="w-3 h-3 inline-block mr-2 text-red-500" />
              Failed
            </>
          );
        }
        return (
          <>
            <CheckCircle className="w-3 h-3 inline-block mr-2 text-green-500" />
            Completed
          </>
        );
      case "denied":
        return (
          <>
            <X className="w-3 h-3 inline-block mr-2 text-red-500" />
            Denied
          </>
        );
      default:
        return null;
    }
  };

  const childContext: ToolDecisionChildContext | undefined = call.id ? { childAgentName: call.name, parentCallId: call.id } : undefined;

  const handleApprove = () => {
    if (onToolDecision && childToolRequest?.id) {
      onToolDecision(childToolRequest.id, KAGENT_HITL_DECISION_TYPE_APPROVE, childContext);
    }
  };

  const handleDeny = () => {
    if (onToolDecision && childToolRequest?.id) {
      onToolDecision(childToolRequest.id, KAGENT_HITL_DECISION_TYPE_DENY, childContext);
    }
  };

  const isDenied = status === "denied";
  const cardBorderClass = isDenied
    ? 'border-red-400 dark:border-red-600'
    : isChildApprovalNeeded 
      ? 'border-amber-400 dark:border-amber-600' 
      : isError 
        ? 'border-red-300' 
        : '';

  return (
    <Card className={`w-full mx-auto my-1 min-w-full ${cardBorderClass}`}>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-xs flex space-x-5">
          <div className="flex items-center font-medium">
            <KagentLogo className="w-4 h-4 mr-2" />
            {agentDisplay}
          </div>
          <div className="font-light">{call.id}</div>
        </CardTitle>
        <div className="flex justify-center items-center text-xs">
          {getStatusDisplay()}
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-2 mt-2">
          <button className="text-xs flex items-center gap-2" onClick={() => setAreInputsExpanded(!areInputsExpanded)}>
            <MessageSquare className="w-4 h-4" />
            <span>Input</span>
            {areInputsExpanded ? <ChevronUp className="w-4 h-4 ml-1" /> : <ChevronDown className="w-4 h-4 ml-1" />}
          </button>
          {areInputsExpanded && (
            <div className="mt-2 bg-muted/50 p-3 rounded">
              <pre className="text-sm whitespace-pre-wrap break-words">{JSON.stringify(call.args, null, 2)}</pre>
            </div>
          )}
        </div>

        <div className="mt-4 w-full">          
          {/* Show child tool approval details if detected */}
          {isChildApprovalNeeded && childToolRequest && (
            <div className="space-y-2">
              <div className="text-xs text-muted-foreground">
                Child agent requires approval for tool:
              </div>
              <div className="bg-amber-50 dark:bg-amber-950/20 p-3 rounded border border-amber-200 dark:border-amber-800">
                <div className="text-sm font-medium">{childToolRequest.name}</div>
                <div className="mt-2">
                  <div className="text-xs text-muted-foreground">Arguments:</div>
                  <pre className="text-xs whitespace-pre-wrap break-words mt-1">
                    {JSON.stringify(childToolRequest.args, null, 2)}
                  </pre>
                </div>
              </div>
            </div>
          )}
          
          {hasResult && result?.content && !isChildApprovalNeeded && (
            <div className="space-y-2">
              <button className="text-xs flex items-center gap-2" onClick={() => setAreResultsExpanded(!areResultsExpanded)}>
                <MessageSquare className="w-4 h-4" />
                <span>Output</span>
                {areResultsExpanded ? <ChevronUp className="w-4 h-4 ml-1" /> : <ChevronDown className="w-4 h-4 ml-1" />}
              </button>
              {areResultsExpanded && (
                <div className={`mt-2 ${isError ? 'bg-red-50 dark:bg-red-950/10' : 'bg-muted/50'} p-3 rounded`}>
                  <pre className={`text-sm whitespace-pre-wrap break-words ${isError ? 'text-red-600 dark:text-red-400' : ''}`}>
                    {result?.content}
                  </pre>
                </div>
              )}
            </div>
          )}
        </div>
      </CardContent>
      
      {/* Approval buttons for child tool */}
      {isChildApprovalNeeded && childToolRequest && onToolDecision && (
        <CardFooter className="flex flex-col gap-3 pt-0 pb-4">
          <div className="flex justify-end gap-2 w-full">
            <Button
              variant="outline"
              size="sm"
              onClick={handleDeny}
              disabled={isButtonDisabled}
              className="text-red-600 hover:text-red-700 hover:bg-red-50 dark:text-red-400 dark:hover:bg-red-950/20"
            >
              <X className="w-4 h-4 mr-2" />
              Deny
            </Button>
            <Button
              variant="default"
              size="sm"
              onClick={handleApprove}
              disabled={isButtonDisabled}
              className="bg-green-600 hover:bg-green-700 text-white"
            >
              <Check className="w-4 h-4 mr-2" />
              Approve
            </Button>
          </div>
        </CardFooter>
      )}
    </Card>
  );
};

export default AgentCallDisplay;



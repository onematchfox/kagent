import { useMemo, useState } from "react";
import { FunctionCall } from "@/types";
import { Card, CardHeader, CardTitle, CardContent, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { convertToUserFriendlyName } from "@/lib/utils";
import { ChevronDown, ChevronUp, MessageSquare, Loader2, AlertCircle, CheckCircle, ShieldAlert, Check, X } from "lucide-react";
import KagentLogo from "../kagent-logo";
import { parseToolApprovalFromString, ToolApprovalRequest, ToolDecisionType, type ToolDecisionDisplayContext, KAGENT_HITL_DECISION_TYPE_APPROVE, KAGENT_HITL_DECISION_TYPE_DENY } from "@/lib/hitl";

export type AgentCallStatus = "requested" | "executing" | "completed" | "denied";

interface AgentCallDisplayProps {
  call: FunctionCall;
  result?: {
    content: string;
    is_error?: boolean;
    /** True when this response contains a tool_approval (backend or parsed). */
    hasToolApproval?: boolean;
    /** Agent/source name for display when present. */
    interruptAgentName?: string;
  };
  status?: AgentCallStatus;
  isError?: boolean;
  onToolDecision?: (toolId: string, decision: ToolDecisionType, displayContext?: ToolDecisionDisplayContext) => void;
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
  
  // Detect tool_approval in this response (backend flag or parse content)
  const toolApproval = useMemo(() => {
    if (!result?.content) return null;
    if (result.hasToolApproval) return parseToolApprovalFromString(result.content);
    return parseToolApprovalFromString(result.content);
  }, [result?.content, result?.hasToolApproval]);
  
  const toolApprovalRequest: ToolApprovalRequest | undefined = toolApproval?.action_requests?.[0];
  const isApprovalNeeded = (result?.hasToolApproval || toolApproval !== null) && status !== "completed";
  
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
    
    if (isApprovalNeeded) {
      return (
        <>
          <ShieldAlert className="w-3 h-3 inline-block mr-2 text-amber-500" />
          Tool awaiting approval
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

  const agentNameForDisplay = result?.interruptAgentName ?? call.name;
  const displayContext: ToolDecisionDisplayContext | undefined = { agentName: agentNameForDisplay };

  const handleApprove = () => {
    if (onToolDecision && toolApprovalRequest?.id) {
      onToolDecision(toolApprovalRequest.id, KAGENT_HITL_DECISION_TYPE_APPROVE, displayContext);
    }
  };

  const handleDeny = () => {
    if (onToolDecision && toolApprovalRequest?.id) {
      onToolDecision(toolApprovalRequest.id, KAGENT_HITL_DECISION_TYPE_DENY, displayContext);
    }
  };

  const isDenied = status === "denied";
  const cardBorderClass = isDenied
    ? 'border-red-400 dark:border-red-600'
    : isApprovalNeeded
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
          {isApprovalNeeded && toolApprovalRequest && (
            <div className="space-y-2">
              <div className="text-xs text-muted-foreground">
                Approval required for tool:
              </div>
              <div className="bg-amber-50 dark:bg-amber-950/20 p-3 rounded border border-amber-200 dark:border-amber-800">
                <div className="text-sm font-medium">{toolApprovalRequest.name}</div>
                <div className="mt-2">
                  <div className="text-xs text-muted-foreground">Arguments:</div>
                  <pre className="text-xs whitespace-pre-wrap break-words mt-1">
                    {JSON.stringify(toolApprovalRequest.args, null, 2)}
                  </pre>
                </div>
              </div>
            </div>
          )}
          
          {hasResult && result?.content && !isApprovalNeeded && (
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
      
      {isApprovalNeeded && toolApprovalRequest && onToolDecision && (
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



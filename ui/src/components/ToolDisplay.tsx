import { useState, useEffect } from "react";
import { FunctionCall } from "@/types";
import { ScrollArea } from "@radix-ui/react-scroll-area";
import { FunctionSquare, CheckCircle, Clock, Code, ChevronUp, ChevronDown, Loader2, Text, Check, Copy, AlertCircle, ShieldAlert, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent, CardFooter } from "@/components/ui/card";
import { ToolDecisionType, KAGENT_HITL_DECISION_TYPE_APPROVE, KAGENT_HITL_DECISION_TYPE_DENY } from "@/lib/hitl";

export type ToolCallStatus = "requested" | "executing" | "completed" | "pending_approval" | "denied";

interface ToolDisplayProps {
  call: FunctionCall;
  result?: {
    content: string;
    is_error?: boolean;
  };
  status?: ToolCallStatus;
  isError?: boolean;
  /** Called when user makes a decision on tool execution */
  onDecision?: (toolId: string, decision: ToolDecisionType) => void;
  /** Hint message from the confirmation request */
  confirmationHint?: string;
  /** True when backend is ready for user input (final input-required event received) */
  awaitingInput?: boolean;
  /** Whether this is from an active streaming session (vs loaded from history) */
  isStreaming?: boolean;
}

const ToolDisplay = ({ 
  call, 
  result, 
  status = "requested", 
  isError = false,
  onDecision,
  confirmationHint,
  awaitingInput,
  isStreaming = false,
}: ToolDisplayProps) => {
  const [areArgumentsExpanded, setAreArgumentsExpanded] = useState(status === "pending_approval");
  const [areResultsExpanded, setAreResultsExpanded] = useState(false);
  const [isCopied, setIsCopied] = useState(false);

  const hasResult = result !== undefined;
  const isPendingApproval = status === "pending_approval";

  // Auto-expand arguments when tool requires approval so users can see what they're approving
  useEffect(() => {
    if (status === "pending_approval") {
      setAreArgumentsExpanded(true);
    }
  }, [status]);
  
  // For historical sessions (isStreaming=false), buttons are always enabled for pending approvals
  // For live streaming (isStreaming=true), buttons wait for awaitingInput to be true
  const isButtonDisabled = isStreaming && !awaitingInput;

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(result?.content || "");
      setIsCopied(true);
      setTimeout(() => setIsCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy text:", err);
    }
  };

  const handleApprove = () => {
    if (onDecision && call.id) {
      setAreArgumentsExpanded(false);
      onDecision(call.id, KAGENT_HITL_DECISION_TYPE_APPROVE);
    }
  };

  const handleDeny = () => {
    if (onDecision && call.id) {
      setAreArgumentsExpanded(false);
      onDecision(call.id, KAGENT_HITL_DECISION_TYPE_DENY);
    }
  };

  // Define UI elements based on status
  const getStatusDisplay = () => {
    if (isError && status === "executing") {
      return (
        <>
          <AlertCircle className="w-3 h-3 inline-block mr-2 text-red-500" />
          Error
        </>
      );
    }

    switch (status) {
      case "requested":
        return (
          <>
            <Clock className="w-3 h-3 inline-block mr-2 text-blue-500" />
            Call requested
          </>
        );
      case "pending_approval":
        return (
          <>
            <ShieldAlert className="w-3 h-3 inline-block mr-2 text-amber-500" />
            Awaiting approval
          </>
        );
      case "executing":
        return (
          <>
            <Loader2 className="w-3 h-3 inline-block mr-2 text-yellow-500 animate-spin" />
            Executing
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
            Tool call denied
          </>
        );
      default:
        return null;
    }
  };

  const isDenied = status === "denied";
  const cardBorderClass = isDenied
    ? 'border-red-400 dark:border-red-600'
    : isPendingApproval 
      ? 'border-amber-400 dark:border-amber-600' 
      : isError 
        ? 'border-red-300' 
        : '';

  return (
    <Card className={`w-full mx-auto my-1 min-w-full ${cardBorderClass}`}>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-xs flex space-x-5">
          <div className="flex items-center font-medium">
            <FunctionSquare className="w-4 h-4 mr-2" />
            {call.name}
          </div>
          <div className="font-light">{call.id}</div>
        </CardTitle>
        <div className="flex justify-center items-center text-xs">
          {getStatusDisplay()}
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-2 mt-4">
          <Button variant="ghost" size="sm" className="p-0 h-auto justify-start" onClick={() => setAreArgumentsExpanded(!areArgumentsExpanded)}>
            <Code className="w-4 h-4 mr-2" />
            <span className="mr-2">Arguments</span>
            {areArgumentsExpanded ? <ChevronUp className="w-4 h-4 ml-auto" /> : <ChevronDown className="w-4 h-4 ml-auto" />}
          </Button>
          {areArgumentsExpanded && (
            <div className="relative">
              <ScrollArea className="max-h-96 overflow-y-auto p-4 w-full mt-2 bg-muted/50">
                <pre className="text-sm whitespace-pre-wrap break-words">
                  {JSON.stringify(call.args, null, 2)}
                </pre>
              </ScrollArea>
            </div>
          )}
        </div>
        <div className="mt-4 w-full">
          {status === "executing" && !hasResult && (
            <div className="flex items-center p-0 h-auto">
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              <span className="text-sm">Executing...</span>
            </div>
          )}
          {hasResult && (
            <>
              <Button variant="ghost" size="sm" className="p-0 h-auto justify-start" onClick={() => setAreResultsExpanded(!areResultsExpanded)}>
                <Text className="w-4 h-4 mr-2" />
                <span className="mr-2">{isError ? "Error" : "Results"}</span>
                {areResultsExpanded ? <ChevronUp className="w-4 h-4 ml-auto" /> : <ChevronDown className="w-4 h-4 ml-auto" />}
              </Button>
              {areResultsExpanded && (
                <div className="relative">
                  <ScrollArea className={`max-h-96 overflow-y-auto p-4 w-full mt-2 ${isError ? 'bg-red-50 dark:bg-red-950/10' : ''}`}>
                    <pre className={`text-sm whitespace-pre-wrap break-words ${isError ? 'text-red-600 dark:text-red-400' : ''}`}>
                      {result.content}
                    </pre>
                  </ScrollArea>

                  <Button variant="ghost" size="sm" className="absolute top-2 right-2 p-2" onClick={handleCopy}>
                    {isCopied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                  </Button>
                </div>
              )}
            </>
          )}
        </div>
      </CardContent>
      {isPendingApproval && onDecision && (
        <CardFooter className="flex flex-col gap-3 pt-0 pb-4">
          {confirmationHint && (
            <p className="text-xs text-muted-foreground w-full">{confirmationHint}</p>
          )}
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

export default ToolDisplay;

"use client";

import { useState, useEffect } from "react";
import { Share2, Loader2, Copy, Check, Globe, Lock, Eye, EyeOff } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { toast } from "sonner";
import {
  createSessionShare,
  deleteSessionShare,
  listSessionShares,
  type SessionShare,
} from "@/app/actions/sessionShares";

interface ShareButtonProps {
  sessionId: string;
  namespace: string;
  agentName: string;
}

export default function ShareButton({ sessionId, namespace, agentName }: ShareButtonProps) {
  const [open, setOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [share, setShare] = useState<SessionShare | null>(null);
  const [copied, setCopied] = useState(false);
  const [readOnly, setReadOnly] = useState(true);

  const shareUrl = share
    ? `${window.location.origin}/agents/${namespace}/${agentName}/chat/${sessionId}?share=${share.token}`
    : null;

  useEffect(() => {
    let cancelled = false;
    listSessionShares(sessionId).then((result) => {
      if (cancelled) return;
      const existing = result.data?.[0] ?? null;
      setShare(existing);
      if (existing) setReadOnly(existing.read_only);
    }).catch(() => {});
    return () => { cancelled = true; };
  }, [sessionId]);

  const handleOpenChange = async (next: boolean) => {
    setOpen(next);
    if (!next) return;

    setIsLoading(true);
    try {
      const result = await listSessionShares(sessionId);
      const existing = result.data?.[0] ?? null;
      setShare(existing);
      if (existing) {
        setReadOnly(existing.read_only);
      }
    } catch {
      toast.error("Failed to load share status");
    } finally {
      setIsLoading(false);
    }
  };

  const handleCreate = async () => {
    setIsLoading(true);
    try {
      const result = await createSessionShare(sessionId, readOnly);
      if (result.error || !result.data) {
        toast.error(result.error || "Failed to create share link");
        return;
      }
      setShare(result.data);
    } catch {
      toast.error("Something went wrong");
    } finally {
      setIsLoading(false);
    }
  };

  const handleRevoke = async () => {
    setIsLoading(true);
    try {
      const result = await listSessionShares(sessionId);
      const tokens = result.data?.map((s) => s.token) ?? [];
      if (share && !tokens.includes(share.token)) {
        tokens.push(share.token);
      }
      const errors: string[] = [];
      for (const token of tokens) {
        const r = await deleteSessionShare(sessionId, token);
        if (r.error) errors.push(r.error);
      }
      if (errors.length > 0) {
        toast.error("Failed to remove some share links");
        return;
      }
      setShare(null);
      setReadOnly(true);
    } catch {
      toast.error("Something went wrong");
    } finally {
      setIsLoading(false);
    }
  };

  const handleCopy = async () => {
    if (!shareUrl) return;
    try {
      await navigator.clipboard.writeText(shareUrl);
      setCopied(true);
      toast.success("Link copied to clipboard");
      setTimeout(() => setCopied(false), 2000);
    } catch {
      toast.error("Failed to copy link — please copy it manually");
    }
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          title={share ? "Shared — click to manage" : "Share this chat"}
          className={share ? "text-primary" : undefined}
        >
          <Share2 className="h-4 w-4" />
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Share chat</DialogTitle>
        </DialogHeader>

        {isLoading && !share ? (
          <div className="flex items-center justify-center py-6">
            <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
          </div>
        ) : share && shareUrl ? (
          // Active share — show link and revoke option
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Globe className="h-4 w-4 shrink-0 text-primary" />
                <span>Anyone with the link can access this chat.</span>
              </div>
              {share.read_only ? (
                <span className="inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium text-muted-foreground">
                  <EyeOff className="h-3 w-3" /> View only
                </span>
              ) : (
                <span className="inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium text-muted-foreground">
                  <Eye className="h-3 w-3" /> Can interact
                </span>
              )}
            </div>
            <div className="flex items-center gap-2">
              <input
                readOnly
                value={shareUrl}
                className="flex-1 rounded-md border bg-muted px-3 py-2 text-xs text-muted-foreground focus:outline-none"
              />
              <Button variant="outline" size="icon" onClick={handleCopy} title="Copy link">
                {copied ? (
                  <Check className="h-4 w-4 text-green-500" />
                ) : (
                  <Copy className="h-4 w-4" />
                )}
              </Button>
            </div>
            <Button
              variant="outline"
              size="sm"
              className="w-full text-muted-foreground"
              onClick={handleRevoke}
              disabled={isLoading}
            >
              {isLoading ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <Lock className="h-4 w-4 mr-2" />}
              Remove link
            </Button>
          </div>
        ) : (
          // No share yet — configure and create
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <Label htmlFor="read-only-switch" className="text-sm font-medium cursor-pointer">
                  Read-only
                </Label>
                <p className="text-xs text-muted-foreground mt-0.5">
                  {readOnly ? "Visitors can view but not interact" : "Visitors can view and interact"}
                </p>
              </div>
              <Switch
                id="read-only-switch"
                checked={readOnly}
                onCheckedChange={setReadOnly}
                disabled={isLoading}
              />
            </div>
            <Button className="w-full" onClick={handleCreate} disabled={isLoading}>
              {isLoading ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <Globe className="h-4 w-4 mr-2" />}
              Create share link
            </Button>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}

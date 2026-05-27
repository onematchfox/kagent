"use client";
import { use, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import ChatInterface from "@/components/chat/ChatInterface";

function ChatPageViewInner({ params }: { params: Promise<{ name: string; namespace: string; chatId: string }> }) {
  const { name, namespace, chatId } = use(params);
  const searchParams = useSearchParams();
  const shareToken = searchParams.get("share") ?? undefined;

  return <ChatInterface
    selectedAgentName={name}
    selectedNamespace={namespace}
    sessionId={chatId}
    shareToken={shareToken}
  />;
}

export default function ChatPageView({ params }: { params: Promise<{ name: string; namespace: string; chatId: string }> }) {
  return (
    <Suspense>
      <ChatPageViewInner params={params} />
    </Suspense>
  );
}

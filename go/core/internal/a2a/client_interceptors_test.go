package a2a

import (
	"context"
	"testing"

	a2atype "github.com/a2aproject/a2a-go/v2/a2a"
	a2aclient "github.com/a2aproject/a2a-go/v2/a2aclient"
	authimpl "github.com/kagent-dev/kagent/go/core/internal/httpserver/auth"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/trace"
	"k8s.io/apimachinery/pkg/types"
)

func TestUpstreamAuthInterceptor_InjectsTraceContext(t *testing.T) {
	const rawTraceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"

	ctx := propagation.TraceContext{}.Extract(
		context.Background(),
		propagation.MapCarrier{"traceparent": rawTraceparent},
	)

	req := &a2aclient.Request{
		BaseURL:       "http://agent.default:8080",
		ServiceParams: a2aclient.ServiceParams{},
	}
	interceptor := NewUpstreamAuthInterceptor(nil, types.NamespacedName{})
	if _, _, err := interceptor.Before(ctx, req); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	gotValues := req.ServiceParams.Get("traceparent")
	if len(gotValues) == 0 {
		t.Fatal("expected traceparent service param on outgoing request, got none")
	}

	outCtx := propagation.TraceContext{}.Extract(context.Background(), propagation.MapCarrier{"traceparent": gotValues[0]})
	wantTraceID := trace.SpanContextFromContext(ctx).TraceID()
	gotTraceID := trace.SpanContextFromContext(outCtx).TraceID()
	if wantTraceID != gotTraceID {
		t.Errorf("trace ID: want %s, got %s", wantTraceID, gotTraceID)
	}
}

func TestUpstreamAuthInterceptor_NoTraceContext(t *testing.T) {
	req := &a2aclient.Request{
		BaseURL:       "http://agent.default:8080",
		ServiceParams: a2aclient.ServiceParams{},
	}
	interceptor := NewUpstreamAuthInterceptor(nil, types.NamespacedName{})
	if _, _, err := interceptor.Before(context.Background(), req); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if got := req.ServiceParams.Get("traceparent"); len(got) != 0 {
		t.Errorf("expected no traceparent service param, got %q", got)
	}
}

func TestUpstreamAuthInterceptor_ShareUsesOwnerUserID(t *testing.T) {
	sendTo := func(contextID string) *a2atype.SendMessageRequest {
		return &a2atype.SendMessageRequest{Message: &a2atype.Message{ContextID: contextID}}
	}
	tests := []struct {
		name     string
		ctx      context.Context
		payload  any
		wantUser string
	}{
		{name: "no share", ctx: ctxWithUser("visitor"), payload: sendTo("sess-1"), wantUser: "visitor"},
		{name: "share of this session", ctx: ctxWithUserAndShare("visitor"), payload: sendTo("sess-1"), wantUser: "owner-id"},
		{name: "share of another session", ctx: ctxWithUserAndShare("visitor"), payload: sendTo("sess-2"), wantUser: "visitor"},
		{name: "share without context id", ctx: ctxWithUserAndShare("visitor"), payload: sendTo(""), wantUser: "visitor"},
		{name: "share on non-message call", ctx: ctxWithUserAndShare("visitor"), payload: &a2atype.GetTaskRequest{ID: "t1"}, wantUser: "visitor"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &a2aclient.Request{
				BaseURL:       "http://agent.default:8080",
				ServiceParams: a2aclient.ServiceParams{},
				Payload:       tt.payload,
			}
			interceptor := NewUpstreamAuthInterceptor(&authimpl.UnsecureAuthenticator{}, types.NamespacedName{Namespace: "default", Name: "agent"})
			if _, _, err := interceptor.Before(tt.ctx, req); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			got := req.ServiceParams.Get("X-User-Id")
			if len(got) != 1 || got[0] != tt.wantUser {
				t.Errorf("X-User-Id = %q, want [%q]", got, tt.wantUser)
			}
		})
	}
}

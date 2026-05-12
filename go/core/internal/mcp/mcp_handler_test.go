package mcp

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/kagent-dev/kagent/go/core/internal/a2a"
	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	a2aclient "trpc.group/trpc-go/trpc-a2a-go/client"
)

// a2aBackend is a fake A2A server that records whether it was called.
type a2aBackend struct {
	server *httptest.Server
	mu     sync.Mutex
	called bool
}

func (b *a2aBackend) wasCalled() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.called
}

func newA2ABackend(t *testing.T) *a2aBackend {
	t.Helper()
	b := &a2aBackend{}
	b.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b.mu.Lock()
		b.called = true
		b.mu.Unlock()
		resp := map[string]any{
			"jsonrpc": "2.0",
			"id":      "",
			"result": map[string]any{
				"kind":      "message",
				"messageId": "test-msg",
				"role":      "agent",
				"parts":     []any{map[string]any{"kind": "text", "text": "hello from agent"}},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(resp); err != nil {
			t.Errorf("failed to encode fake A2A response: %v", err)
		}
	}))
	t.Cleanup(b.server.Close)
	return b
}

// newTestRegistry builds an AgentClientRegistry with a single agent pre-registered.
func newTestRegistry(t *testing.T, namespace, name, backendURL string) *a2a.AgentClientRegistry {
	t.Helper()
	c, err := a2aclient.NewA2AClient(backendURL + "/" + namespace + "/" + name + "/")
	require.NoError(t, err)
	registry := a2a.NewAgentClientRegistry()
	registry.Register(namespace, name, c)
	return registry
}

// TestInvokeAgent_InvalidAgentRef verifies that invoke_agent returns a tool
// error for agent references that are not exactly "namespace/name".
func TestInvokeAgent_InvalidAgentRef(t *testing.T) {
	for _, ref := range []string{"no-slash", "ns/name/extra", "/name", "ns/"} {
		t.Run(ref, func(t *testing.T) {
			registry := a2a.NewAgentClientRegistry()
			mcpHandler, err := NewMCPHandler(nil, registry, nil)
			require.NoError(t, err)

			mcpServer := httptest.NewServer(mcpHandler)
			t.Cleanup(mcpServer.Close)

			transport := &mcpsdk.StreamableClientTransport{
				Endpoint:             mcpServer.URL,
				DisableStandaloneSSE: true,
			}

			ctx := context.Background()
			cs, err := mcpsdk.NewClient(&mcpsdk.Implementation{Name: "test", Version: "1.0"}, nil).
				Connect(ctx, transport, nil)
			require.NoError(t, err)
			t.Cleanup(func() { cs.Close() })

			result, err := cs.CallTool(ctx, &mcpsdk.CallToolParams{
				Name:      "invoke_agent",
				Arguments: map[string]any{"agent": ref, "task": "say hello"},
			})
			require.NoError(t, err)
			assert.True(t, result.IsError, "expected a tool error for invalid agent ref %q", ref)
		})
	}
}

// TestInvokeAgent_UnregisteredAgent verifies that invoke_agent returns a tool
// error when the requested agent is not present in the AgentClientRegistry.
func TestInvokeAgent_UnregisteredAgent(t *testing.T) {
	registry := a2a.NewAgentClientRegistry() // empty — no agents registered
	mcpHandler, err := NewMCPHandler(nil, registry, nil)
	require.NoError(t, err)

	mcpServer := httptest.NewServer(mcpHandler)
	t.Cleanup(mcpServer.Close)

	transport := &mcpsdk.StreamableClientTransport{
		Endpoint:             mcpServer.URL,
		DisableStandaloneSSE: true,
	}

	ctx := context.Background()
	cs, err := mcpsdk.NewClient(&mcpsdk.Implementation{Name: "test", Version: "1.0"}, nil).
		Connect(ctx, transport, nil)
	require.NoError(t, err)
	t.Cleanup(func() { cs.Close() })

	result, err := cs.CallTool(ctx, &mcpsdk.CallToolParams{
		Name:      "invoke_agent",
		Arguments: map[string]any{"agent": "default/unknown-agent", "task": "say hello"},
	})
	require.NoError(t, err)
	assert.True(t, result.IsError, "expected a tool error for an unregistered agent")
}

// TestInvokeAgent_RoutesViaRegistry verifies that invoke_agent retrieves the
// pre-registered A2A client from AgentClientRegistry and forwards the call.
func TestInvokeAgent_RoutesViaRegistry(t *testing.T) {
	backend := newA2ABackend(t)

	registry := newTestRegistry(t, "default", "test-agent", backend.server.URL)
	mcpHandler, err := NewMCPHandler(nil, registry, nil)
	require.NoError(t, err)

	mcpServer := httptest.NewServer(mcpHandler)
	t.Cleanup(mcpServer.Close)

	transport := &mcpsdk.StreamableClientTransport{
		Endpoint:             mcpServer.URL,
		DisableStandaloneSSE: true,
	}

	ctx := context.Background()
	cs, err := mcpsdk.NewClient(&mcpsdk.Implementation{Name: "test", Version: "1.0"}, nil).
		Connect(ctx, transport, nil)
	require.NoError(t, err)
	t.Cleanup(func() { cs.Close() })

	result, err := cs.CallTool(ctx, &mcpsdk.CallToolParams{
		Name:      "invoke_agent",
		Arguments: map[string]any{"agent": "default/test-agent", "task": "say hello"},
	})
	require.NoError(t, err)
	assert.False(t, result.IsError, "unexpected tool error: %v", result.Content)
	assert.True(t, backend.wasCalled(), "A2A backend should have received the forwarded request")
}

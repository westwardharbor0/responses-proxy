package main

import (
	"bytes"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestModelsEndpoint(t *testing.T) {
	logger := newTestLogger()
	h := newHandler(logger, nil, false)
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	rec := httptest.NewRecorder()

	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}

	var resp modelsResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode models response: %v", err)
	}
	if len(resp.Data) != 1 {
		t.Fatalf("expected 1 model, got %d", len(resp.Data))
	}
	if resp.Data[0].ID == "" {
		t.Fatalf("model id should not be empty")
	}
}

func TestChatProxiesToResponses(t *testing.T) {
	// Upstream stub to capture payload and return a Responses-style body
	var captured struct {
		method  string
		path    string
		headers http.Header
		body    []byte
	}

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured.method = r.Method
		captured.path = r.URL.Path
		captured.headers = r.Header.Clone()
		b, _ := io.ReadAll(r.Body)
		r.Body.Close()
		captured.body = b

		resp := responsesPayload{
			ID:         "resp-123",
			Status:     "succeeded",
			Model:      "gpt-5.3-codex",
			Created:    time.Now().Unix(),
			OutputText: "",
			Output: []struct {
				Content []struct {
					Text string `json:"text"`
				} `json:"content"`
			}{
				{Content: []struct {
					Text string `json:"text"`
				}{
					{Text: "hello"},
				}},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	t.Cleanup(upstream.Close)

	logger := newTestLogger()
	proxy := &responseProxy{
		endpoint:   upstream.URL,
		apiKey:     "test-key",
		apiVersion: "2025-04-01-preview",
		model:      "test-deploy",
		client:     upstream.Client(),
		logger:     logger,
		verbose:    false,
	}

	h := newHandler(logger, proxy, false)

	chatReq := chatCompletionRequest{
		Model: "ignored-by-proxy",
		Messages: []chatMessage{
			{Role: "system", Content: mustRawJSON(`"sys"`)},
			{Role: "user", Content: mustRawJSON(`"hello"`)},
			{Role: "assistant", Content: mustRawJSON(`"world"`)},
		},
		MaxTokens: 42,
	}
	body, _ := json.Marshal(chatReq)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d; body=%s", rec.Code, rec.Body.String())
	}

	var completion chatCompletionResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &completion); err != nil {
		t.Fatalf("decode completion: %v", err)
	}
	if got := completion.Choices[0].Message.Content; got != "hello" {
		t.Fatalf("expected completion content 'hello', got %q", got)
	}
	if got := completion.Model; got != "gpt-5.3-codex" {
		t.Fatalf("expected model passthrough, got %q", got)
	}

	// Verify upstream request mapping
	if captured.method != http.MethodPost {
		t.Fatalf("upstream method = %s", captured.method)
	}
	if captured.path != "/openai/responses" {
		t.Fatalf("upstream path = %s", captured.path)
	}
	var upstreamPayload map[string]any
	if err := json.Unmarshal(captured.body, &upstreamPayload); err != nil {
		t.Fatalf("decode upstream payload: %v", err)
	}
	if upstreamPayload["model"] != "test-deploy" {
		t.Fatalf("expected model 'test-deploy', got %v", upstreamPayload["model"])
	}
	inputs, ok := upstreamPayload["input"].([]any)
	if !ok || len(inputs) != 3 {
		t.Fatalf("expected 3 input items, got %T len=%d", upstreamPayload["input"], len(inputs))
	}
	// user/system => input_text, assistant => output_text
	checkType := func(idx int, expected string) {
		m := inputs[idx].(map[string]any)
		content := m["content"].([]any)
		typ := content[0].(map[string]any)["type"].(string)
		if typ != expected {
			t.Fatalf("input[%d] type expected %s, got %s", idx, expected, typ)
		}
	}
	checkType(0, "input_text")
	checkType(1, "input_text")
	checkType(2, "output_text")
	if maxOut, ok := upstreamPayload["max_output_tokens"].(float64); !ok || int(maxOut) != 42 {
		t.Fatalf("expected max_output_tokens 42, got %v", upstreamPayload["max_output_tokens"])
	}
}

func TestChatSSE(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := responsesPayload{
			ID:      "resp-sse",
			Status:  "succeeded",
			Model:   "gpt-5.3-codex",
			Created: time.Now().Unix(),
			Output: []struct {
				Content []struct {
					Text string `json:"text"`
				} `json:"content"`
			}{
				{Content: []struct {
					Text string `json:"text"`
				}{
					{Text: "chunk"},
				}},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	t.Cleanup(upstream.Close)

	logger := newTestLogger()
	proxy := &responseProxy{
		endpoint:   upstream.URL,
		apiKey:     "k",
		apiVersion: "2025-04-01-preview",
		model:      "m",
		client:     upstream.Client(),
		logger:     logger,
		verbose:    false,
	}

	h := newHandler(logger, proxy, false)
	chatReq := chatCompletionRequest{
		Stream: true,
		Messages: []chatMessage{
			{Role: "user", Content: mustRawJSON(`"hi"`)},
		},
	}
	body, _ := json.Marshal(chatReq)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := newSSERecorder()

	h.ServeHTTP(rec, req)

	if rec.code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.code)
	}
	if ct := rec.Header().Get("Content-Type"); !strings.Contains(ct, "text/event-stream") {
		t.Fatalf("expected event-stream content type, got %q", ct)
	}
	bodyStr := rec.buf.String()
	if !strings.Contains(bodyStr, "data:") || !strings.Contains(bodyStr, "[DONE]") {
		t.Fatalf("expected SSE data and [DONE], got: %s", bodyStr)
	}
}

func TestChatUpstreamErrors(t *testing.T) {
	// Non-200 upstream
	up500 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte("boom"))
	}))
	t.Cleanup(up500.Close)

	logger := newTestLogger()
	proxy := &responseProxy{
		endpoint:   up500.URL,
		apiKey:     "k",
		apiVersion: "2025-04-01-preview",
		model:      "m",
		client:     up500.Client(),
		logger:     logger,
	}
	h := newHandler(logger, proxy, false)

	chatReq := chatCompletionRequest{Messages: []chatMessage{{Role: "user", Content: mustRawJSON(`"hi"`)}}}
	body, _ := json.Marshal(chatReq)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadGateway {
		t.Fatalf("expected 502 for upstream 500, got %d", rec.Code)
	}

	// Upstream decode error
	upBadJSON := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte("not-json"))
	}))
	t.Cleanup(upBadJSON.Close)
	proxy.endpoint = upBadJSON.URL
	proxy.client = upBadJSON.Client()

	req2 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req2.Header.Set("Content-Type", "application/json")
	rec2 := httptest.NewRecorder()
	h.ServeHTTP(rec2, req2)
	if rec2.Code != http.StatusBadGateway {
		t.Fatalf("expected 502 for upstream decode error, got %d", rec2.Code)
	}
}

func TestChatBadJSONRequest(t *testing.T) {
	logger := newTestLogger()
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{}`))
	}))
	t.Cleanup(up.Close)
	proxy := &responseProxy{
		endpoint:   up.URL,
		apiKey:     "k",
		apiVersion: "2025-04-01-preview",
		model:      "m",
		client:     up.Client(),
		logger:     logger,
	}
	h := newHandler(logger, proxy, false)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader([]byte("not-json")))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for bad JSON, got %d", rec.Code)
	}
}

// Helpers
func newTestLogger() *log.Logger {
	return log.New(io.Discard, "", log.LstdFlags)
}

func mustRawJSON(s string) json.RawMessage {
	return json.RawMessage(s)
}

// Minimal ResponseWriter that supports Flush for SSE tests
type sseRecorder struct {
	header http.Header
	buf    bytes.Buffer
	code   int
}

func newSSERecorder() *sseRecorder {
	return &sseRecorder{header: http.Header{}}
}

func (r *sseRecorder) Header() http.Header {
	return r.header
}

func (r *sseRecorder) WriteHeader(status int) {
	r.code = status
}

func (r *sseRecorder) Write(b []byte) (int, error) {
	if r.code == 0 {
		r.code = http.StatusOK
	}
	return r.buf.Write(b)
}

func (r *sseRecorder) Flush() {}

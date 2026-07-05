package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

const (
	defaultAPIVersion = "2025-04-01-preview"
	defaultListenAddr = ":8080"
)

// modelInfo is a minimal model entry for the static /v1/models response.
type modelInfo struct {
	Status          string           `json:"status"`
	Capabilities    map[string]bool  `json:"capabilities"`
	LifecycleStatus string           `json:"lifecycle_status"`
	Deprecation     map[string]int64 `json:"deprecation"`
	ID              string           `json:"id"`
	CreatedAt       int64            `json:"created_at"`
	Object          string           `json:"object"`
}

// modelsResponse mirrors the OpenAI models response shape.
type modelsResponse struct {
	Data []modelInfo `json:"data"`
}

// chatMessage holds a single chat message coming from the OpenAI-style client.
type chatMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

// chatCompletionRequest is the OpenAI-style chat completions request the proxy accepts.
type chatCompletionRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	MaxTokens   int           `json:"max_tokens"`
	Temperature *float64      `json:"temperature"`
	Stream      bool          `json:"stream"`
}

// chatCompletionResponse is the OpenAI-style response returned to the client.
type chatCompletionResponse struct {
	ID      string                     `json:"id"`
	Object  string                     `json:"object"`
	Created int64                      `json:"created"`
	Model   string                     `json:"model"`
	Choices []chatCompletionChoice     `json:"choices"`
	Usage   map[string]json.RawMessage `json:"usage,omitempty"`
}

type chatCompletionChoice struct {
	Index        int           `json:"index"`
	Message      choiceMessage `json:"message"`
	FinishReason string        `json:"finish_reason"`
}

type choiceMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// responsesPayload is the subset of Azure Responses payload we decode.
type responsesPayload struct {
	ID         string `json:"id"`
	Status     string `json:"status"`
	Model      string `json:"model"`
	Created    int64  `json:"created"`
	OutputText string `json:"output_text"`
	Output     []struct {
		Content []struct {
			Text string `json:"text"`
		} `json:"content"`
	} `json:"output"`
}

// responseProxy carries upstream configuration and handles forwarding to Azure Responses.
type responseProxy struct {
	endpoint   string
	apiKey     string
	apiVersion string
	model      string
	client     *http.Client
	logger     *log.Logger
	verbose    bool
}

func main() {
	logger := log.New(os.Stdout, "", log.LstdFlags)
	verbose := parseBool(os.Getenv("LOG_VERBOSE"))
	proxy := newResponseProxy(logger, verbose)
	handler := newHandler(logger, proxy, verbose)
	addr := listenAddr()
	logger.Printf("listening on %s", addr)
	if err := http.ListenAndServe(addr, handler); err != nil {
		logger.Fatalf("server error: %v", err)
	}
}

// listenAddr returns the listen address, defaulting to :8080.
func listenAddr() string {
	if v := strings.TrimSpace(os.Getenv("LISTEN_ADDR")); v != "" {
		return v
	}
	return defaultListenAddr
}

func newHandler(logger *log.Logger, proxy *responseProxy, verbose bool) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			logger.Printf("request %s %s read body err=%v", r.Method, r.URL.Path, err)
			_ = r.Body.Close()
			return
		}
		if verbose {
			logger.Printf("request %s %s body_bytes=%d", r.Method, r.URL.Path, len(body))
			for name, values := range r.Header {
				for _, value := range values {
					logger.Printf("header %s: %s", name, value)
				}
			}
			if len(body) > 0 {
				logger.Printf("body: %s", string(body))
			}
		} else {
			logger.Printf("request %s %s", r.Method, r.URL.Path)
		}
		_ = r.Body.Close()

		if r.URL.Path == "/v1/models" {
			resp := modelsResponse{
				Data: []modelInfo{
					{
						Status: "succeeded",
						Capabilities: map[string]bool{
							"fine_tune":         false,
							"inference":         true,
							"completion":        false,
							"chat_completion":   false,
							"embeddings":        false,
							"global_fine_tune":  false,
							"devtier_fine_tune": false,
							"blossom_fine_tune": false,
						},
						LifecycleStatus: "generally-available",
						Deprecation: map[string]int64{
							"inference": 1803513600,
						},
						ID:        "gpt-5.3-codex-2026-02-24",
						CreatedAt: 1771891200,
						Object:    "model",
					},
				},
			}

			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			if err := json.NewEncoder(w).Encode(resp); err != nil {
				logger.Printf("encode models response: %v", err)
			}
			return
		}

		if r.URL.Path == "/v1/chat/completions" {
			if proxy == nil {
				http.Error(w, "responses proxy not configured; set LLM_ENDPOINT, LLM_API_KEY, and LLM_DEPLOYMENT", http.StatusInternalServerError)
				return
			}
			proxy.handleChatCompletion(r.Context(), body, w)
			return
		}

		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok\n"))
	})
}

func newResponseProxy(logger *log.Logger, verbose bool) *responseProxy {
	endpoint := strings.TrimRight(os.Getenv("LLM_ENDPOINT"), "/")
	apiKey := os.Getenv("LLM_API_KEY")
	apiVersion := os.Getenv("LLM_API_VERSION")
	if apiVersion == "" {
		apiVersion = defaultAPIVersion
	}
	model := os.Getenv("LLM_DEPLOYMENT")
	if model == "" {
		model = os.Getenv("LLM_MODEL")
	}

	if endpoint == "" || apiKey == "" || model == "" {
		logger.Printf("responses proxy disabled: set LLM_ENDPOINT, LLM_API_KEY, and LLM_DEPLOYMENT/LLM_MODEL to enable forwarding")
		return nil
	}

	return &responseProxy{
		endpoint:   endpoint,
		apiKey:     apiKey,
		apiVersion: apiVersion,
		model:      model,
		client:     &http.Client{Timeout: 300 * time.Second},
		logger:     logger,
		verbose:    verbose,
	}
}

func (p *responseProxy) handleChatCompletion(ctx context.Context, body []byte, w http.ResponseWriter) {
	var req chatCompletionRequest
	if err := json.Unmarshal(body, &req); err != nil {
		p.logger.Printf("chat decode: %v", err)
		http.Error(w, "invalid chat request", http.StatusBadRequest)
		return
	}

	model := p.model

	inputs := make([]map[string]any, 0, len(req.Messages))
	for _, msg := range req.Messages {
		text := extractMessageText(msg.Content)
		contentType := "input_text"
		if strings.ToLower(msg.Role) == "assistant" {
			contentType = "output_text"
		}
		inputs = append(inputs, map[string]any{
			"role": msg.Role,
			"content": []map[string]string{
				{
					"type": contentType,
					"text": text,
				},
			},
		})
	}

	payload := map[string]any{
		"model": model,
		"input": inputs,
	}
	if req.MaxTokens > 0 {
		payload["max_output_tokens"] = req.MaxTokens
	}
	if req.Temperature != nil {
		payload["temperature"] = *req.Temperature
	}

	encoded, err := json.Marshal(payload)
	if err != nil {
		p.logger.Printf("encode responses payload: %v", err)
		http.Error(w, "encode payload", http.StatusInternalServerError)
		return
	}

	url := fmt.Sprintf("%s/openai/responses?api-version=%s", p.endpoint, p.apiVersion)
	if p.verbose {
		p.logger.Printf("proxying chat -> responses endpoint=%s model=%s messages=%d", url, model, len(inputs))
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(encoded))
	if err != nil {
		p.logger.Printf("build responses request: %v", err)
		http.Error(w, "build request", http.StatusInternalServerError)
		return
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("api-key", p.apiKey)
	httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)

	resp, err := p.client.Do(httpReq)
	if err != nil {
		p.logger.Printf("responses request: %v", err)
		http.Error(w, "upstream request failed", http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		p.logger.Printf("responses read: %v", err)
		http.Error(w, "read upstream", http.StatusBadGateway)
		return
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		snippet := string(respBody)
		if len(snippet) > 512 {
			snippet = snippet[:512]
		}
		p.logger.Printf("responses status %d: %s", resp.StatusCode, snippet)
		http.Error(w, "upstream error", http.StatusBadGateway)
		return
	}

	if p.verbose {
		p.logger.Printf("responses status %d bytes=%d", resp.StatusCode, len(respBody))
	}

	var parsed responsesPayload
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		p.logger.Printf("responses decode: %v", err)
		http.Error(w, "decode upstream", http.StatusBadGateway)
		return
	}

	text := strings.TrimSpace(parsed.OutputText)
	if text == "" {
		for _, out := range parsed.Output {
			for _, content := range out.Content {
				if strings.TrimSpace(content.Text) != "" {
					text = content.Text
					break
				}
			}
			if text != "" {
				break
			}
		}
	}

	if text == "" {
		p.logger.Printf("responses empty output")
		http.Error(w, "empty upstream output", http.StatusBadGateway)
		return
	}

	created := parsed.Created
	if created == 0 {
		created = time.Now().Unix()
	}
	id := parsed.ID
	if id == "" {
		id = fmt.Sprintf("resp-%d", time.Now().UnixNano())
	}

	// SSE support if client requested streaming
	if req.Stream || strings.Contains(strings.ToLower(strings.Join(w.Header().Values("Accept"), ",")), "text/event-stream") {
		if flusher, ok := w.(http.Flusher); ok {
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			chunk := map[string]any{
				"id":      id,
				"object":  "chat.completion.chunk",
				"created": created,
				"model":   parsed.Model,
				"choices": []map[string]any{
					{
						"index": 0,
						"delta": map[string]any{
							"role":    "assistant",
							"content": text,
						},
						"finish_reason": nil,
					},
				},
			}

			if err := writeSSE(w, chunk); err != nil {
				p.logger.Printf("sse write chunk: %v", err)
				return
			}
			flusher.Flush()

			end := map[string]any{
				"id":      id,
				"object":  "chat.completion.chunk",
				"created": created,
				"model":   parsed.Model,
				"choices": []map[string]any{
					{
						"index":         0,
						"delta":         map[string]any{},
						"finish_reason": "stop",
					},
				},
			}
			if err := writeSSE(w, end); err != nil {
				p.logger.Printf("sse write end: %v", err)
				return
			}
			flusher.Flush()
			if _, err := w.Write([]byte("data: [DONE]\n\n")); err != nil {
				p.logger.Printf("sse write done: %v", err)
			}
			flusher.Flush()
			return
		}
	}

	completion := chatCompletionResponse{
		ID:      id,
		Object:  "chat.completion",
		Created: created,
		Model:   parsed.Model,
		Choices: []chatCompletionChoice{
			{
				Index: 0,
				Message: choiceMessage{
					Role:    "assistant",
					Content: text,
				},
				FinishReason: "stop",
			},
		},
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	if err := json.NewEncoder(w).Encode(completion); err != nil {
		p.logger.Printf("encode chat response: %v", err)
	}
}

func writeSSE(w http.ResponseWriter, payload any) error {
	encoded, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	if _, err := w.Write([]byte("data: " + string(encoded) + "\n\n")); err != nil {
		return err
	}
	return nil
}

func parseBool(val string) bool {
	v, err := strconv.ParseBool(val)
	return err == nil && v
}

func extractMessageText(raw json.RawMessage) string {
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}

	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &parts); err == nil {
		var texts []string
		for _, p := range parts {
			if strings.TrimSpace(p.Text) != "" {
				texts = append(texts, p.Text)
			}
		}
		if len(texts) > 0 {
			return strings.Join(texts, " ")
		}
	}

	return strings.TrimSpace(string(raw))
}

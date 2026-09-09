package conversation

import (
	"errors"
	"io"
	"strings"
	"testing"
)

func TestConvertResponseStreamRejectsMissingTerminalEvent(t *testing.T) {
	for _, operation := range []string{OperationChat, OperationMessages} {
		for _, test := range []struct {
			name   string
			source string
		}{
			{name: "empty"},
			{name: "heartbeat", source: ": keepalive\n\n"},
			{name: "created", source: "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\"}}\n\n"},
			{name: "text delta", source: "data: {\"type\":\"response.output_text.delta\",\"delta\":\"partial answer\"}\n\n"},
			{name: "truncated terminal", source: "data: {\"type\":\"response.completed\",\"response\":"},
		} {
			t.Run(operation+"/"+test.name, func(t *testing.T) {
				stream := ConvertResponseStream(io.NopCloser(strings.NewReader(test.source)), operation)
				defer stream.Close()
				output, err := io.ReadAll(stream)
				if !errors.Is(err, io.ErrUnexpectedEOF) {
					t.Errorf("read error = %v, want io.ErrUnexpectedEOF", err)
				}
				for _, marker := range []string{`"finish_reason":"stop"`, "data: [DONE]", "event: message_stop", `"stop_reason":"end_turn"`} {
					if strings.Contains(string(output), marker) {
						t.Errorf("unterminated stream emitted success marker %q: %s", marker, output)
					}
				}
			})
		}
	}
}

func TestConvertResponseStreamPreservesTerminalEvents(t *testing.T) {
	for _, operation := range []string{OperationChat, OperationMessages} {
		for _, test := range []struct {
			name        string
			terminal    string
			chatMarker  string
			messageMark string
		}{
			{
				name: "completed", terminal: `{"type":"response.completed","response":{"status":"completed"}}`,
				chatMarker: `"finish_reason":"stop"`, messageMark: `"stop_reason":"end_turn"`,
			},
			{
				name: "incomplete", terminal: `{"type":"response.incomplete","response":{"status":"incomplete"}}`,
				chatMarker: `"finish_reason":"length"`, messageMark: `"stop_reason":"max_tokens"`,
			},
			{
				name: "failed", terminal: `{"type":"response.failed","response":{"error":{"message":"upstream failure"}}}`,
				chatMarker: `"error":`, messageMark: "event: error",
			},
			{
				name: "error", terminal: `{"type":"error","error":{"message":"upstream failure"}}`,
				chatMarker: `"error":`, messageMark: "event: error",
			},
			{
				name: "done sentinel", terminal: "[DONE]",
				chatMarker: `"finish_reason":"stop"`, messageMark: `"stop_reason":"end_turn"`,
			},
		} {
			for _, ending := range []string{"\n\n", ""} {
				t.Run(operation+"/"+test.name+"/ending="+ending, func(t *testing.T) {
					source := "data: {\"type\":\"response.output_text.delta\",\"delta\":\"answer\"}\n\n" + "data: " + test.terminal + ending
					stream := ConvertResponseStream(io.NopCloser(strings.NewReader(source)), operation)
					defer stream.Close()
					output, err := io.ReadAll(stream)
					if err != nil {
						t.Fatal(err)
					}
					marker := test.chatMarker
					if operation == OperationMessages {
						marker = test.messageMark
					}
					if !strings.Contains(string(output), marker) {
						t.Fatalf("missing terminal marker %q: %s", marker, output)
					}
				})
			}
		}
	}
}
